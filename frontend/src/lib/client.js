function isObject(value) {
	return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function text(value) {
	return typeof value === 'string' ? value : '';
}

function errorText(body) {
	if (typeof body.detail === 'string' && body.detail) return body.detail;
	const fields = isObject(body.detail) ? body.detail : body;
	for (const candidate of [fields.message, fields.error]) {
		if (typeof candidate === 'string' && candidate) return candidate;
	}
	return 'Request failed';
}

function thrownText(err) {
	if (err instanceof SyntaxError) return 'The backend sent a response that is not valid JSON';
	if (err instanceof Error && err.message) return err.message;
	if (typeof err === 'string' && err) return err;
	return 'Request failed';
}

export async function fetchHealth(api) {
	let payload;
	try {
		payload = await api.get('/health');
	} catch (err) {
		return { ok: false, error: thrownText(err) };
	}

	if (!isObject(payload)) return { ok: false, error: 'Unexpected response from /health' };
	if (payload.success === false || payload.detail !== undefined) {
		return { ok: false, error: errorText(payload) };
	}

	const body = isObject(payload.data) ? payload.data : payload;
	const hasStatus = typeof body.status === 'string';
	const healthy = hasStatus ? body.status === 'ok' : body.ok === true || payload.success === true;
	if (!healthy) {
		return { ok: false, error: hasStatus ? `Backend status: ${body.status}` : 'Unexpected response from /health' };
	}
	return typeof body.version === 'string' ? { ok: true, version: body.version } : { ok: true };
}

export function healthLabel(result) {
	if (!result) return 'Backend: checking…';
	if (!result.ok) return `Backend error: ${result.error ?? 'unknown error'}`;
	return result.version ? `Backend: ok (v${result.version.replace(/^v/i, '')})` : 'Backend: ok';
}

const STATUS_BY_CODE = {
	invalid_request: 400,
	invalid: 400,
	unauthorized: 401,
	forbidden: 403,
	limit_exceeded: 403,
	not_visible_to_requester: 403,
	not_found: 404,
	unknown_source: 404,
	blocked: 409,
	already_available: 409,
	not_open: 409,
	requester_gone: 409,
	rate_limited: 429,
	internal_error: 500,
	source_unavailable: 502
};

const CODE_BY_STATUS = {
	400: 'invalid_request',
	401: 'unauthorized',
	403: 'forbidden',
	404: 'not_found',
	409: 'conflict',
	422: 'invalid_request',
	429: 'rate_limited',
	500: 'internal_error',
	502: 'source_unavailable',
	503: 'unavailable'
};

const AUTH_DETAIL = /not authenticated|could not validate credentials|invalid (?:or expired )?token|token (?:has )?expired/i;

function defaultMessage(status) {
	if (status === 401) return 'Your session has expired: sign in again';
	if (status === 403) return 'You do not have permission to do this';
	if (status === 404) return 'Not found';
	if (status === 429) return 'Too many requests: try again in a moment';
	if (status >= 500) return 'The server could not complete the request';
	return 'Request failed';
}

export function parseRetryAfter(value) {
	if (typeof value === 'number') return Number.isFinite(value) && value >= 0 ? Math.ceil(value) : null;
	if (typeof value !== 'string' || !value.trim()) return null;
	if (/^\s*\d+(?:\.\d+)?\s*$/.test(value)) return Math.ceil(Number(value));
	const match = value.match(/(?:try again|retry)\s+(?:in|after)\s+(\d+(?:\.\d+)?)\s*(?:s\b|sec|second)/i);
	return match ? Math.ceil(Number(match[1])) : null;
}

export function normalizeError(body, status = 0, retryAfterHeader = null) {
	let code = '';
	let message = '';
	let httpStatus = Number.isInteger(status) && status > 0 ? status : 0;
	if (isObject(body)) {
		const { detail } = body;
		if (typeof detail === 'string') {
			message = detail;
			if (!httpStatus && AUTH_DETAIL.test(detail)) httpStatus = 401;
		} else if (Array.isArray(detail)) {
			message = detail
				.map((item) => (isObject(item) && typeof item.msg === 'string' ? item.msg : ''))
				.filter(Boolean)
				.join('; ');
			code = 'invalid_request';
			if (!httpStatus) httpStatus = 422;
		} else {
			const fields = isObject(detail) ? detail : body;
			code = text(fields.error);
			message = text(fields.message);
		}
	}
	if (!httpStatus) httpStatus = STATUS_BY_CODE[code] ?? 0;
	if (!code) code = CODE_BY_STATUS[httpStatus] ?? 'error';
	if (!message) message = defaultMessage(httpStatus);
	const limited = httpStatus === 429 || code === 'rate_limited';
	const retryAfter = limited ? parseRetryAfter(retryAfterHeader) ?? parseRetryAfter(message) : null;
	return { status: httpStatus, code, message, retryAfter };
}

function thrownError(err) {
	if (err instanceof SyntaxError) {
		return { status: 0, code: 'invalid_response', message: 'The server sent a response that is not valid JSON', retryAfter: null };
	}
	if (err instanceof TypeError) {
		return { status: 0, code: 'network_error', message: 'Could not reach the server', retryAfter: null };
	}
	return { status: 0, code: 'network_error', message: thrownText(err), retryAfter: null };
}

export function unwrap(payload, status = 0, retryAfterHeader = null) {
	if (status >= 400) return { ok: false, error: normalizeError(payload, status, retryAfterHeader) };
	if (isObject(payload) && payload.success === true) return { ok: true, data: payload.data };
	if (isObject(payload) && payload.success === undefined && payload.detail === undefined && 'data' in payload) {
		return { ok: true, data: payload.data };
	}
	if (!isObject(payload)) {
		return { ok: false, error: { status, code: 'invalid_response', message: 'Unexpected response from the server', retryAfter: null } };
	}
	return { ok: false, error: normalizeError(payload, status, retryAfterHeader) };
}

export async function callPlugin(api, method, path, body) {
	if (!api || typeof api[method] !== 'function') {
		return { ok: false, error: { status: 0, code: 'no_api', message: 'The page has no connection to the server', retryAfter: null } };
	}
	let payload;
	try {
		payload = body === undefined ? await api[method](path) : await api[method](path, body);
	} catch (err) {
		return { ok: false, error: thrownError(err) };
	}
	return unwrap(payload);
}

function encodeSegment(value) {
	return encodeURIComponent(String(value));
}

export function modelPath(source, id) {
	return `/${encodeSegment(source)}/models/${String(id).split('/').map(encodeSegment).join('/')}`;
}

export function getMe(api) {
	return callPlugin(api, 'get', '/me');
}

export function getSources(api) {
	return callPlugin(api, 'get', '/sources');
}

export function searchModels(api, source, query) {
	return callPlugin(api, 'get', `/${encodeSegment(source)}/search${query ? `?${query}` : ''}`);
}

export function getModel(api, source, id) {
	return callPlugin(api, 'get', modelPath(source, id));
}

export function getNsfwMedia(api, source, id) {
	return callPlugin(api, 'get', `${modelPath(source, id)}/nsfw-media`);
}

export function getInstalled(api, sha256s) {
	return callPlugin(api, 'post', '/installed', { sha256s: [...new Set(sha256s)].slice(0, 200) });
}

export function startDownload(api, body) {
	return callPlugin(api, 'post', '/downloads', body);
}

export function listDownloads(api, { all, limit } = {}) {
	const params = new URLSearchParams();
	if (typeof all === 'boolean') params.set('all', String(all));
	if (typeof limit === 'number') params.set('limit', String(limit));
	const query = params.toString();
	return callPlugin(api, 'get', `/downloads${query ? `?${query}` : ''}`);
}

async function callCore(api, path, { method = 'GET', body, fetchImpl } = {}) {
	const doFetch = fetchImpl ?? globalThis.fetch;
	if (typeof doFetch !== 'function') {
		return { ok: false, error: { status: 0, code: 'no_fetch', message: 'This browser cannot reach the server', retryAfter: null } };
	}
	const token = typeof api?.getToken === 'function' ? api.getToken() : null;
	const headers = { Accept: 'application/json' };
	if (token) headers.Authorization = `Bearer ${token}`;
	const init = { method, headers, credentials: 'include', cache: 'no-store' };
	if (body !== undefined) {
		headers['Content-Type'] = 'application/json';
		init.body = JSON.stringify(body);
	}

	let response;
	try {
		response = await doFetch(path, init);
	} catch (err) {
		return { ok: false, error: thrownError(err) };
	}
	let payload = null;
	try {
		payload = await response.json();
	} catch {
	}
	const retryAfter = typeof response.headers?.get === 'function' ? response.headers.get('Retry-After') : null;
	if (!response.ok) return { ok: false, error: normalizeError(payload, response.status || 0, retryAfter) };
	return unwrap(payload, 0, retryAfter);
}

export function setNsfwMode(api, mode, { key = 'media_nsfw_filter_mode', fetchImpl } = {}) {
	return callCore(api, `/api/settings/${encodeSegment(key)}`, { method: 'PUT', body: { value: mode }, fetchImpl });
}

export async function listRemoteBackends(api, { fetchImpl } = {}) {
	const result = await callCore(api, '/api/backends', { fetchImpl });
	if (!result.ok) return result;
	const backends = Array.isArray(result.data) ? result.data : [];
	return {
		ok: true,
		data: backends
			.filter((backend) => isObject(backend) && backend.enabled !== false && backend.driver === 'native.remote' && backend.id)
			.map((backend) => ({ id: String(backend.id), name: text(backend.name) || String(backend.id) }))
	};
}
