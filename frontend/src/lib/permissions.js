export const AUTH_TOKEN_KEY = 'auth_token';

export const ASSIGN_TO = ['user', 'group'];

const FALLBACK_CAPABILITIES = ['browse', 'request', 'download', 'approve'];
const FALLBACK_SOURCES = ['civitai', 'huggingface', 'openmodeldb'];

const GB_PATTERN = /^(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?$/i;

function isObject(value) {
	return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function stringList(value, fallback) {
	if (!Array.isArray(value)) return [...fallback];
	return [...new Set(value.filter((item) => typeof item === 'string'))];
}

function sortedUnique(list) {
	return [...new Set(list)].sort();
}

export function parseMaxFileGb(text) {
	if (text === null || text === undefined) return { ok: true, value: null };
	if (typeof text === 'number') {
		return Number.isFinite(text) && text > 0 ? { ok: true, value: text } : { ok: false };
	}
	if (typeof text !== 'string') return { ok: false };
	const trimmed = text.trim();
	if (trimmed === '') return { ok: true, value: null };
	if (!GB_PATTERN.test(trimmed)) return { ok: false };
	const value = Number(trimmed);
	return Number.isFinite(value) && value > 0 ? { ok: true, value } : { ok: false };
}

function formatMaxFileGb(value) {
	if (typeof value === 'number') return Number.isFinite(value) && value > 0 ? String(value) : '';
	return typeof value === 'string' ? value.trim() : '';
}

function toRow(group) {
	const params = isObject(group.params) ? group.params : {};
	const id = String(group.group_id);
	return {
		group_id: id,
		name: typeof group.name === 'string' && group.name ? group.name : id,
		is_system: group.is_system === true,
		capabilities: sortedUnique(stringList(group.capabilities, [])),
		params: {
			sources: sortedUnique(stringList(params.sources, FALLBACK_SOURCES)),
			model_types: sortedUnique(stringList(params.model_types, [])),
			max_file_gb: formatMaxFileGb(params.max_file_gb),
			assign_to: typeof params.assign_to === 'string' ? params.assign_to : 'user',
			allow_remote: params.allow_remote === true
		}
	};
}

export function toFormState(payload) {
	const data = isObject(payload) && isObject(payload.data) ? payload.data : payload;
	const source = isObject(data) ? data : {};
	return {
		allowed: {
			capabilities: stringList(source.capabilities, FALLBACK_CAPABILITIES),
			sources: stringList(source.sources, []),
			model_types: stringList(source.model_types, [])
		},
		groups: (Array.isArray(source.groups) ? source.groups : [])
			.filter((group) => isObject(group) && group.group_id !== undefined && group.group_id !== null)
			.map(toRow)
	};
}

export function toPutBody(state) {
	return {
		groups: state.groups.map((group) => {
			const size = parseMaxFileGb(group.params.max_file_gb);
			if (!size.ok) throw new RangeError(`Invalid maximum file size for the group '${group.group_id}'`);
			return {
				group_id: group.group_id,
				capabilities: sortedUnique(group.capabilities),
				params: {
					sources: sortedUnique(group.params.sources),
					model_types: sortedUnique(group.params.model_types),
					max_file_gb: size.value,
					assign_to: group.params.assign_to,
					allow_remote: group.params.allow_remote === true
				}
			};
		})
	};
}

function outside(values, allowed) {
	return values.filter((value) => !allowed.includes(value));
}

export function validateState(state) {
	const allowed = state?.allowed ?? { capabilities: [], sources: [], model_types: [] };
	const problems = [];
	for (const group of state?.groups ?? []) {
		const fail = (message) => problems.push(`${group.name || group.group_id}: ${message}`);

		for (const capability of outside(group.capabilities, allowed.capabilities)) {
			fail(`unknown permission '${capability}'`);
		}
		for (const source of outside(group.params.sources, allowed.sources)) {
			fail(`unknown source '${source}'`);
		}
		for (const type of outside(group.params.model_types, allowed.model_types)) {
			fail(`unknown model type '${type}'`);
		}
		if (!parseMaxFileGb(group.params.max_file_gb).ok) {
			fail('the maximum file size must be a number above 0 (decimals with a dot), or empty for no limit');
		}
		if (!ASSIGN_TO.includes(group.params.assign_to)) {
			fail("'assign to' must be 'user' or 'group'");
		}
	}
	return problems;
}

function canonical(state) {
	return (state?.groups ?? []).map((group) => {
		const size = parseMaxFileGb(group.params.max_file_gb);
		return {
			id: group.group_id,
			capabilities: sortedUnique(group.capabilities),
			sources: sortedUnique(group.params.sources),
			modelTypes: sortedUnique(group.params.model_types),
			maxFileGb: size.ok ? size.value : `invalid:${String(group.params.max_file_gb).trim()}`,
			assignTo: group.params.assign_to,
			allowRemote: group.params.allow_remote === true
		};
	});
}

export function isDirty(initial, state) {
	if (!initial || !state) return initial !== state;
	return JSON.stringify(canonical(initial)) !== JSON.stringify(canonical(state));
}

export function cloneState(state) {
	return JSON.parse(JSON.stringify(state));
}

export function toggleItem(list, id, on) {
	const rest = list.filter((item) => item !== id);
	return sortedUnique(on ? [...rest, id] : rest);
}

export function optionsFor(allowed, selected) {
	return [...allowed, ...outside(selected, allowed)];
}

export function readToken(storage) {
	try {
		const store = storage ?? globalThis.localStorage;
		return store?.getItem(AUTH_TOKEN_KEY) || null;
	} catch {
		return null;
	}
}

function errorText(body, status) {
	if (isObject(body)) {
		const { detail } = body;
		if (typeof detail === 'string' && detail) return detail;
		if (Array.isArray(detail)) {
			const text = detail
				.map((item) => (isObject(item) && typeof item.msg === 'string' ? item.msg : ''))
				.filter(Boolean)
				.join('; ');
			if (text) return text;
		}
		const fields = isObject(detail) ? detail : body;
		for (const candidate of [fields.message, fields.error]) {
			if (typeof candidate === 'string' && candidate) return candidate;
		}
	}
	if (status === 401) return 'Your session has expired: sign in again';
	if (status === 403) return 'Only administrators can edit these permissions';
	return `The server answered with an error (HTTP ${status})`;
}

function thrownText(err) {
	if (err instanceof TypeError) return 'Could not reach the server';
	if (err instanceof Error && err.message) return err.message;
	if (typeof err === 'string' && err) return err;
	return 'Request failed';
}

async function call({ pluginId = 'model-browser-c', method, body, token, fetchImpl }) {
	const bearer = token === undefined ? readToken() : token;
	const headers = { Accept: 'application/json' };
	if (bearer) headers.Authorization = `Bearer ${bearer}`;
	const init = { method, headers, credentials: 'include', cache: 'no-store' };
	if (body !== undefined) {
		headers['Content-Type'] = 'application/json';
		init.body = JSON.stringify(body);
	}
	const url = `/api/plugins/${encodeURIComponent(pluginId)}/permissions`;

	let response;
	try {
		response = await (fetchImpl ?? fetch)(url, init);
	} catch (err) {
		return { ok: false, status: 0, error: thrownText(err) };
	}

	let payload = null;
	try {
		payload = await response.json();
	} catch {
	}

	if (!response.ok) return { ok: false, status: response.status, error: errorText(payload, response.status) };
	if (!isObject(payload) || payload.success === false || !isObject(payload.data) || !Array.isArray(payload.data.groups)) {
		return { ok: false, status: response.status, error: 'Unexpected response from the server' };
	}
	return { ok: true, data: payload.data };
}

export function fetchPermissions(options = {}) {
	return call({ ...options, method: 'GET' });
}

export function savePermissions(body, options = {}) {
	return call({ ...options, method: 'PUT', body });
}
