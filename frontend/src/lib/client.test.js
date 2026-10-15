import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
	getInstalled,
	getMe,
	getModel,
	getNsfwMedia,
	getSources,
	listDownloads,
	listRemoteBackends,
	modelPath,
	normalizeError,
	parseRetryAfter,
	searchModels,
	setNsfwMode,
	startDownload,
	unwrap
} from './client.js';

function fakeHostApi(handler, token = 'secret-token') {
	const calls = [];
	return {
		calls,
		getToken: () => token,
		get: async (path) => {
			calls.push(['get', path]);
			return handler('get', path);
		},
		post: async (path, body) => {
			calls.push(['post', path, body]);
			return handler('post', path, body);
		}
	};
}

const ok = (data) => ({ success: true, data });
const fail = (error, message) => ({ detail: { success: false, error, message } });

test('the plugin calls hit the documented paths and unwrap the envelope', async () => {
	const api = fakeHostApi(() => ok({ hello: 'world' }));
	assert.deepEqual(await getMe(api), { ok: true, data: { hello: 'world' } });
	await getSources(api);
	await searchModels(api, 'civitai', 'q=x&types=LORA&types=Checkpoint&limit=24');
	await searchModels(api, 'civitai', '');
	await getModel(api, 'huggingface', 'latent-consistency/lcm-lora-sdxl');
	await getNsfwMedia(api, 'civitai', '58390');
	await listDownloads(api, { all: false, limit: 30 });
	await listDownloads(api);
	assert.deepEqual(
		api.calls.map((call) => call[1]),
		[
			'/me',
			'/sources',
			'/civitai/search?q=x&types=LORA&types=Checkpoint&limit=24',
			'/civitai/search',
			'/huggingface/models/latent-consistency/lcm-lora-sdxl',
			'/civitai/models/58390/nsfw-media',
			'/downloads?all=false&limit=30',
			'/downloads'
		]
	);
});

test('modelPath encodes each segment of the id on its own', () => {
	assert.equal(modelPath('huggingface', 'org name/repo#1'), '/huggingface/models/org%20name/repo%231');
	assert.equal(modelPath('civitai', 123), '/civitai/models/123');
});

test('POST calls send their body', async () => {
	const api = fakeHostApi((method, path, body) => ok({ path, body }));
	const sha = 'a'.repeat(64);
	await getInstalled(api, [sha, sha, 'b'.repeat(64)]);
	const body = { source: 'civitai', model_id: '1', version_id: '2', file_id: '3' };
	await startDownload(api, body);
	assert.deepEqual(api.calls, [
		['post', '/installed', { sha256s: [sha, 'b'.repeat(64)] }],
		['post', '/downloads', body]
	]);
});

test('an error envelope becomes { status, code, message, retryAfter }', async () => {
	const api = fakeHostApi(() => fail('not_found', 'Model not found'));
	assert.deepEqual(await getModel(api, 'civitai', '1'), {
		ok: false,
		error: { status: 404, code: 'not_found', message: 'Model not found', retryAfter: null }
	});
});

test('a rate limit carries the seconds to wait, read from the message', async () => {
	const api = fakeHostApi(() => fail('rate_limited', 'CivitAI rate limit reached; try again in 30 s'));
	const result = await searchModels(api, 'civitai', '');
	assert.deepEqual(result.error, {
		status: 429,
		code: 'rate_limited',
		message: 'CivitAI rate limit reached; try again in 30 s',
		retryAfter: 30
	});
});

test('rejections of the host api become network or format errors', async () => {
	const offline = fakeHostApi(() => {
		throw new TypeError('Failed to fetch');
	});
	assert.deepEqual((await getMe(offline)).error, { status: 0, code: 'network_error', message: 'Could not reach the server', retryAfter: null });
	const html = fakeHostApi(() => {
		throw new SyntaxError('Unexpected token <');
	});
	assert.equal((await getMe(html)).error.code, 'invalid_response');
	const text = fakeHostApi(() => {
		throw 'offline';
	});
	assert.deepEqual((await getMe(text)).error, { status: 0, code: 'network_error', message: 'offline', retryAfter: null });
	const nothing = fakeHostApi(() => {
		throw undefined;
	});
	assert.equal((await getMe(nothing)).error.message, 'Request failed');
	assert.equal((await getMe(undefined)).error.code, 'no_api');
});

test('normalizeError reads the shapes the backend and the core send', () => {
	assert.deepEqual(normalizeError({ detail: { success: false, error: 'forbidden', message: 'Not your source' } }), {
		status: 403,
		code: 'forbidden',
		message: 'Not your source',
		retryAfter: null
	});
	assert.deepEqual(normalizeError({ detail: 'Not authenticated' }), {
		status: 401,
		code: 'unauthorized',
		message: 'Not authenticated',
		retryAfter: null
	});
	assert.deepEqual(normalizeError({ detail: [{ msg: 'field required' }, { msg: 'too long' }] }), {
		status: 422,
		code: 'invalid_request',
		message: 'field required; too long',
		retryAfter: null
	});
	assert.deepEqual(normalizeError({ success: false, error: 'blocked', message: 'Refused by a plugin' }), {
		status: 409,
		code: 'blocked',
		message: 'Refused by a plugin',
		retryAfter: null
	});
	assert.deepEqual(normalizeError(null, 503), {
		status: 503,
		code: 'unavailable',
		message: 'The server could not complete the request',
		retryAfter: null
	});
	assert.deepEqual(normalizeError({}, 0), { status: 0, code: 'error', message: 'Request failed', retryAfter: null });
	assert.equal(normalizeError({ detail: { error: 'rate_limited', message: 'slow down' } }, 429, '12').retryAfter, 12);
	assert.equal(normalizeError({ detail: { error: 'teapot', message: 'short and stout' } }).status, 0);
});

test('parseRetryAfter reads headers and messages', () => {
	assert.equal(parseRetryAfter('30'), 30);
	assert.equal(parseRetryAfter('2.5'), 3);
	assert.equal(parseRetryAfter(7), 7);
	assert.equal(parseRetryAfter('Hugging Face rate limit reached; try again in 60 s'), 60);
	assert.equal(parseRetryAfter('Please retry after 5 seconds'), 5);
	assert.equal(parseRetryAfter('try again later'), null);
	assert.equal(parseRetryAfter(''), null);
	assert.equal(parseRetryAfter(null), null);
});

test('unwrap accepts the envelope and refuses everything else', () => {
	assert.deepEqual(unwrap({ success: true, data: [1] }), { ok: true, data: [1] });
	assert.deepEqual(unwrap({ data: 'flat' }), { ok: true, data: 'flat' });
	assert.equal(unwrap('nope').ok, false);
	assert.equal(unwrap('nope').error.code, 'invalid_response');
	assert.equal(unwrap({ success: false, error: 'x', message: 'y' }).error.message, 'y');
});

function fakeFetch(reply) {
	const calls = [];
	const impl = async (url, init) => {
		calls.push([url, init]);
		if (typeof reply === 'function') return reply(url, init);
		return reply;
	};
	return { calls, impl };
}

async function withFetch(fetch, run) {
	const saved = globalThis.fetch;
	globalThis.fetch = fetch.impl;
	try {
		return await run();
	} finally {
		globalThis.fetch = saved;
	}
}

function response(status, body, headers = {}) {
	return {
		ok: status >= 200 && status < 300,
		status,
		headers: { get: (name) => headers[name] ?? null },
		json: async () => {
			if (body === undefined) throw new SyntaxError('no body');
			return body;
		}
	};
}

test('setNsfwMode PUTs the core setting with the token in the header only', async () => {
	const fetch = fakeFetch(response(200, { success: true, data: { key: 'media_nsfw_filter_mode', value: 'show' } }));
	const api = fakeHostApi(() => null, 'tok-123');
	const result = await withFetch(fetch, () => setNsfwMode(api, 'show', 'media_nsfw_filter_mode'));
	assert.equal(result.ok, true);
	const [url, init] = fetch.calls[0];
	assert.equal(url, '/api/settings/media_nsfw_filter_mode');
	assert.equal(init.method, 'PUT');
	assert.equal(init.headers.Authorization, 'Bearer tok-123');
	assert.equal(init.headers['Content-Type'], 'application/json');
	assert.deepEqual(JSON.parse(init.body), { value: 'show' });
	assert.ok(!url.includes('tok-123'));
});

test('setNsfwMode uses the key from /me and reports failures', async () => {
	const fetch = fakeFetch(response(403, { detail: 'Not allowed' }));
	const result = await withFetch(fetch, () => setNsfwMode(fakeHostApi(() => null), 'hide', 'custom_key'));
	assert.equal(fetch.calls[0][0], '/api/settings/custom_key');
	assert.deepEqual(result, { ok: false, error: { status: 403, code: 'forbidden', message: 'Not allowed', retryAfter: null } });

	const softFailure = fakeFetch(response(200, { success: false, error: 'invalid_value', message: 'Unknown mode' }));
	const soft = await withFetch(softFailure, () => setNsfwMode(fakeHostApi(() => null), 'hide', 'custom_key'));
	assert.equal(soft.ok, false);
	assert.equal(soft.error.message, 'Unknown mode');

	const down = fakeFetch(() => {
		throw new TypeError('Failed to fetch');
	});
	assert.equal((await withFetch(down, () => setNsfwMode(fakeHostApi(() => null), 'hide', 'custom_key'))).error.code, 'network_error');
});

test('a core call without a token sends no Authorization header', async () => {
	const fetch = fakeFetch(response(200, { success: true, data: [] }));
	await withFetch(fetch, () => listRemoteBackends(fakeHostApi(() => null, null)));
	assert.equal(fetch.calls[0][1].headers.Authorization, undefined);
});

test('listRemoteBackends keeps the enabled native.remote backends', async () => {
	const fetch = fakeFetch(
		response(200, {
			success: true,
			data: [
				{ id: 'local', name: 'This machine', driver: 'native.local', enabled: true },
				{ id: 'r1', name: 'GPU box', driver: 'native.remote', enabled: true },
				{ id: 'r2', name: 'Old box', driver: 'native.remote', enabled: false },
				{ id: 'r3', driver: 'native.remote' },
				{ id: 'c1', name: 'ComfyUI', driver: 'comfyui', enabled: true }
			]
		})
	);
	const result = await withFetch(fetch, () => listRemoteBackends(fakeHostApi(() => null)));
	assert.deepEqual(result, {
		ok: true,
		data: [
			{ id: 'r1', name: 'GPU box' },
			{ id: 'r3', name: 'r3' }
		]
	});
	assert.equal(fetch.calls[0][0], '/api/backends');
	assert.equal(fetch.calls[0][1].method, 'GET');
});

test('a core answer that is not JSON is reported from its status', async () => {
	const fetch = fakeFetch(response(502, undefined, { 'Retry-After': '9' }));
	const result = await withFetch(fetch, () => listRemoteBackends(fakeHostApi(() => null)));
	assert.deepEqual(result.error, {
		status: 502,
		code: 'source_unavailable',
		message: 'The server could not complete the request',
		retryAfter: null
	});
});
