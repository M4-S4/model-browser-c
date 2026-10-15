import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
	cloneState,
	fetchPermissions,
	isDirty,
	optionsFor,
	parseMaxFileGb,
	readToken,
	savePermissions,
	toFormState,
	toggleItem,
	toPutBody,
	validateState
} from './permissions.js';

const DEFAULT_PARAMS = {
	sources: ['civitai', 'huggingface', 'openmodeldb'],
	model_types: [],
	max_file_gb: null,
	assign_to: 'user',
	allow_remote: false
};

function payload() {
	return {
		success: true,
		data: {
			capabilities: ['browse', 'request', 'download', 'approve'],
			sources: ['civitai', 'huggingface', 'openmodeldb', 'civarchive'],
			model_types: ['checkpoint', 'lora', 'embedding'],
			groups: [
				{
					group_id: 'all_admins',
					name: 'All Admins',
					is_system: true,
					capabilities: ['approve', 'browse', 'download', 'request'],
					params: { ...DEFAULT_PARAMS }
				},
				{
					group_id: 'all_users',
					name: 'All Users',
					is_system: true,
					capabilities: ['browse', 'request'],
					params: { ...DEFAULT_PARAMS }
				},
				{
					group_id: 'artists',
					name: 'Artists',
					is_system: false,
					capabilities: ['browse', 'download', 'request'],
					params: {
						sources: ['civitai', 'huggingface'],
						model_types: ['embedding', 'lora'],
						max_file_gb: 6.5,
						assign_to: 'group',
						allow_remote: true
					}
				}
			]
		}
	};
}

function group(state, id) {
	return state.groups.find((entry) => entry.group_id === id);
}

test('toFormState turns the API answer into editable rows', () => {
	const state = toFormState(payload());

	assert.deepEqual(state.allowed, {
		capabilities: ['browse', 'request', 'download', 'approve'],
		sources: ['civitai', 'huggingface', 'openmodeldb', 'civarchive'],
		model_types: ['checkpoint', 'lora', 'embedding']
	});
	assert.deepEqual(
		state.groups.map((entry) => entry.group_id),
		['all_admins', 'all_users', 'artists']
	);
	assert.deepEqual(group(state, 'artists'), {
		group_id: 'artists',
		name: 'Artists',
		is_system: false,
		capabilities: ['browse', 'download', 'request'],
		params: {
			sources: ['civitai', 'huggingface'],
			model_types: ['embedding', 'lora'],
			max_file_gb: '6.5',
			assign_to: 'group',
			allow_remote: true
		}
	});
	assert.equal(group(state, 'all_users').is_system, true);
});

test('toFormState writes a missing size limit as an empty field', () => {
	const state = toFormState(payload());
	assert.equal(group(state, 'all_users').params.max_file_gb, '');
});

test('toFormState accepts the data object alone as well as the whole envelope', () => {
	assert.deepEqual(toFormState(payload().data), toFormState(payload()));
});

test('toFormState sorts the lists of a row and drops duplicates', () => {
	const body = payload();
	body.data.groups[2].capabilities = ['request', 'browse', 'request'];
	body.data.groups[2].params.sources = ['huggingface', 'civitai'];
	const row = group(toFormState(body), 'artists');
	assert.deepEqual(row.capabilities, ['browse', 'request']);
	assert.deepEqual(row.params.sources, ['civitai', 'huggingface']);
});

test('toFormState returns fresh objects on every call', () => {
	const body = payload();
	const first = toFormState(body);
	const second = toFormState(body);
	first.groups[0].capabilities.push('manage');
	first.groups[0].params.sources.push('x');
	assert.deepEqual(second, toFormState(body));
	assert.notEqual(first.groups[0], second.groups[0]);
});

test('toFormState fills in what a row leaves out', () => {
	const state = toFormState({ data: { groups: [{ group_id: 'g1' }] } });
	assert.deepEqual(state.allowed.capabilities, ['browse', 'request', 'download', 'approve']);
	assert.deepEqual(state.groups, [
		{
			group_id: 'g1',
			name: 'g1',
			is_system: false,
			capabilities: [],
			params: {
				sources: ['civitai', 'huggingface', 'openmodeldb'],
				model_types: [],
				max_file_gb: '',
				assign_to: 'user',
				allow_remote: false
			}
		}
	]);
});

test('toFormState copes with answers it cannot interpret', () => {
	for (const body of [null, undefined, 'ok', 42, [], {}, { data: null }, { data: { groups: 'x' } }]) {
		const state = toFormState(body);
		assert.deepEqual(state.groups, [], JSON.stringify(body));
		assert.ok(Array.isArray(state.allowed.capabilities));
	}
	const state = toFormState({ data: { groups: [null, 'x', {}, { group_id: null }, { group_id: 'ok' }] } });
	assert.deepEqual(
		state.groups.map((entry) => entry.group_id),
		['ok']
	);
});

test('toFormState ignores a size that is not a positive number', () => {
	for (const size of [0, -1, null, undefined, true, {}]) {
		const body = payload();
		body.data.groups[2].params.max_file_gb = size;
		assert.equal(group(toFormState(body), 'artists').params.max_file_gb, '', String(size));
	}
});

test('toPutBody builds the request body: numbers parsed, lists sorted, name and flags left out', () => {
	const state = toFormState(payload());
	const row = group(state, 'artists');
	row.capabilities = ['request', 'download', 'browse'];
	row.params.sources = ['huggingface', 'civitai'];
	row.params.model_types = ['lora', 'embedding'];
	row.params.max_file_gb = ' 12.50 ';

	const body = toPutBody(state);

	assert.deepEqual(body.groups[2], {
		group_id: 'artists',
		capabilities: ['browse', 'download', 'request'],
		params: {
			sources: ['civitai', 'huggingface'],
			model_types: ['embedding', 'lora'],
			max_file_gb: 12.5,
			assign_to: 'group',
			allow_remote: true
		}
	});
	assert.deepEqual(Object.keys(body), ['groups']);
	assert.equal(typeof body.groups[2].params.max_file_gb, 'number');
});

test('toPutBody sends an empty size as null', () => {
	const state = toFormState(payload());
	for (const text of ['', '   ']) {
		group(state, 'artists').params.max_file_gb = text;
		assert.equal(toPutBody(state).groups[2].params.max_file_gb, null, JSON.stringify(text));
	}
});

test('toPutBody lists every group, keeping the parameters of a group without download', () => {
	const state = toFormState(payload());
	const body = toPutBody(state);
	assert.deepEqual(
		body.groups.map((entry) => entry.group_id),
		['all_admins', 'all_users', 'artists']
	);
	assert.deepEqual(body.groups[1].params, DEFAULT_PARAMS);
});

test('toPutBody undoes toFormState', () => {
	const original = payload();
	const body = toPutBody(toFormState(original));
	const expected = original.data.groups.map(({ group_id, capabilities, params }) => ({ group_id, capabilities, params }));
	assert.deepEqual(body.groups, expected);
});

test('toPutBody refuses a size it cannot read instead of sending "no limit"', () => {
	const state = toFormState(payload());
	for (const text of ['abc', '0', '-3', '1e', '4,5', 'Infinity']) {
		group(state, 'artists').params.max_file_gb = text;
		assert.throws(() => toPutBody(state), RangeError, text);
	}
});

test('toPutBody gives the same body when called twice and does not touch the form', () => {
	const state = toFormState(payload());
	const before = cloneState(state);
	assert.deepEqual(toPutBody(state), toPutBody(state));
	assert.deepEqual(state, before);
});

test('parseMaxFileGb reads positive numbers written with a dot', () => {
	assert.deepEqual(parseMaxFileGb('4'), { ok: true, value: 4 });
	assert.deepEqual(parseMaxFileGb('4.5'), { ok: true, value: 4.5 });
	assert.deepEqual(parseMaxFileGb('.5'), { ok: true, value: 0.5 });
	assert.deepEqual(parseMaxFileGb('5.'), { ok: true, value: 5 });
	assert.deepEqual(parseMaxFileGb('1e3'), { ok: true, value: 1000 });
	assert.deepEqual(parseMaxFileGb('  12 '), { ok: true, value: 12 });
	assert.deepEqual(parseMaxFileGb(7), { ok: true, value: 7 });
});

test('parseMaxFileGb reads an empty field as no limit', () => {
	for (const text of ['', '  ', null, undefined]) {
		assert.deepEqual(parseMaxFileGb(text), { ok: true, value: null }, String(text));
	}
});

test('parseMaxFileGb rejects everything else', () => {
	for (const text of ['0', '0.0', '-1', '+1', 'abc', '1e', '1,5', '1 000', '0x10', 'Infinity', 'NaN', '.', '1..2']) {
		assert.deepEqual(parseMaxFileGb(text), { ok: false }, text);
	}
	for (const value of [0, -2, NaN, Infinity, true, {}, []]) {
		assert.deepEqual(parseMaxFileGb(value), { ok: false }, String(value));
	}
});

test('validateState has nothing to say about a form the API produced', () => {
	assert.deepEqual(validateState(toFormState(payload())), []);
});

test('validateState rejects a capability the API does not list', () => {
	const state = toFormState(payload());
	group(state, 'artists').capabilities.push('manage');
	assert.deepEqual(validateState(state), ["Artists: unknown permission 'manage'"]);
});

test('validateState rejects a source or model type the API does not list', () => {
	const state = toFormState(payload());
	group(state, 'artists').params.sources.push('modelscope');
	group(state, 'artists').params.model_types.push('hypernetwork');
	assert.deepEqual(validateState(state), ["Artists: unknown source 'modelscope'", "Artists: unknown model type 'hypernetwork'"]);
});

test('validateState rejects a size that is not above 0', () => {
	const state = toFormState(payload());
	for (const text of ['0', '-5', 'many', '1e', '4,5']) {
		group(state, 'all_users').params.max_file_gb = text;
		const problems = validateState(state);
		assert.equal(problems.length, 1, text);
		assert.match(problems[0], /^All Users: the maximum file size must be a number above 0/, text);
	}
});

test('validateState accepts a size above 0 or an empty one', () => {
	const state = toFormState(payload());
	for (const text of ['', ' ', '0.25', '8', '1e2', '.5']) {
		group(state, 'all_users').params.max_file_gb = text;
		assert.deepEqual(validateState(state), [], JSON.stringify(text));
	}
});

test('validateState rejects an assign_to other than user or group', () => {
	const state = toFormState(payload());
	group(state, 'artists').params.assign_to = 'team';
	assert.deepEqual(validateState(state), ["Artists: 'assign to' must be 'user' or 'group'"]);
	group(state, 'artists').params.assign_to = 'user';
	assert.deepEqual(validateState(state), []);
});

test('validateState reports every problem, each with the name of its group', () => {
	const state = toFormState(payload());
	group(state, 'all_users').capabilities.push('manage');
	group(state, 'artists').params.max_file_gb = 'x';
	group(state, 'artists').params.assign_to = '';
	const problems = validateState(state);
	assert.equal(problems.length, 3);
	assert.ok(problems[0].startsWith('All Users:'));
	assert.ok(problems[1].startsWith('Artists:'));
	assert.ok(problems[2].startsWith('Artists:'));
});

test('validateState falls back to the group id for a group without a name', () => {
	const state = toFormState({ data: { capabilities: ['browse'], sources: [], model_types: [], groups: [{ group_id: 'g1', capabilities: ['x'] }] } });
	const problems = validateState(state);
	assert.ok(problems.every((message) => message.startsWith('g1:')));
	assert.ok(problems.includes("g1: unknown permission 'x'"));
});

test('validateState copes with a missing state', () => {
	assert.deepEqual(validateState(null), []);
	assert.deepEqual(validateState({}), []);
});

test('isDirty is false for an untouched copy', () => {
	const initial = toFormState(payload());
	assert.equal(isDirty(initial, toFormState(payload())), false);
	assert.equal(isDirty(initial, cloneState(initial)), false);
});

test('isDirty sees every kind of edit', () => {
	const edits = {
		'a capability on': (state) => state.groups[1].capabilities.push('download'),
		'a capability off': (state) => state.groups[1].capabilities.pop(),
		'a source': (state) => state.groups[1].params.sources.push('civarchive'),
		'a model type': (state) => state.groups[1].params.model_types.push('lora'),
		'the size': (state) => (state.groups[1].params.max_file_gb = '2'),
		'assign_to': (state) => (state.groups[1].params.assign_to = 'group'),
		'allow_remote': (state) => (state.groups[1].params.allow_remote = true),
		'unreadable size text': (state) => (state.groups[1].params.max_file_gb = 'abc')
	};
	for (const [name, edit] of Object.entries(edits)) {
		const state = toFormState(payload());
		edit(state);
		assert.equal(isDirty(toFormState(payload()), state), true, name);
	}
});

test('isDirty is false again once an edit is undone', () => {
	const state = toFormState(payload());
	state.groups[1].capabilities = toggleItem(state.groups[1].capabilities, 'download', true);
	assert.equal(isDirty(toFormState(payload()), state), true);
	state.groups[1].capabilities = toggleItem(state.groups[1].capabilities, 'download', false);
	assert.equal(isDirty(toFormState(payload()), state), false);
});

test('isDirty ignores the order of a list and the way a number is written', () => {
	const state = toFormState(payload());
	const artists = group(state, 'artists');
	artists.capabilities.reverse();
	artists.params.sources.reverse();
	artists.params.max_file_gb = '6.50';
	assert.equal(isDirty(toFormState(payload()), state), false);
	artists.params.max_file_gb = ' 6.5 ';
	assert.equal(isDirty(toFormState(payload()), state), false);
});

test('isDirty ignores the group names', () => {
	const state = toFormState(payload());
	state.groups[0].name = 'Renamed elsewhere';
	assert.equal(isDirty(toFormState(payload()), state), false);
});

test('isDirty compares a missing form with a form', () => {
	const state = toFormState(payload());
	assert.equal(isDirty(null, null), false);
	assert.equal(isDirty(null, state), true);
	assert.equal(isDirty(state, null), true);
});

test('toggleItem adds and removes an id, sorted, without touching the list it was given', () => {
	const list = ['lora', 'checkpoint'];
	assert.deepEqual(toggleItem(list, 'embedding', true), ['checkpoint', 'embedding', 'lora']);
	assert.deepEqual(toggleItem(list, 'lora', false), ['checkpoint']);
	assert.deepEqual(toggleItem(list, 'lora', true), ['checkpoint', 'lora']);
	assert.deepEqual(toggleItem(list, 'vae', false), ['checkpoint', 'lora']);
	assert.deepEqual(list, ['lora', 'checkpoint']);
});

test('optionsFor lists the allowed ids, then the selected ones the API does not know', () => {
	assert.deepEqual(optionsFor(['b', 'a'], ['a']), ['b', 'a']);
	assert.deepEqual(optionsFor(['b', 'a'], ['gone', 'a']), ['b', 'a', 'gone']);
	assert.deepEqual(optionsFor([], ['x']), ['x']);
});

test('cloneState copies deeply', () => {
	const state = toFormState(payload());
	const copy = cloneState(state);
	copy.groups[0].params.sources.push('x');
	assert.notDeepEqual(copy, state);
	assert.deepEqual(state, toFormState(payload()));
});

async function withGlobal(name, value, fn) {
	const saved = Object.getOwnPropertyDescriptor(globalThis, name);
	Object.defineProperty(globalThis, name, { configurable: true, writable: true, value });
	try {
		return await fn();
	} finally {
		if (saved) Object.defineProperty(globalThis, name, saved);
		else delete globalThis[name];
	}
}

function storageWith(token) {
	return { getItem: (key) => (key === 'auth_token' ? token : null) };
}

function withCall(fetchImpl, fn, token = 't') {
	return withGlobal('localStorage', storageWith(token), () => withGlobal('fetch', fetchImpl, fn));
}

test('readToken reads auth_token from localStorage', async () => {
	const asked = [];
	const storage = {
		getItem: (key) => {
			asked.push(key);
			return 'tok-123';
		}
	};
	assert.equal(await withGlobal('localStorage', storage, () => readToken()), 'tok-123');
	assert.deepEqual(asked, ['auth_token']);
});

test('readToken returns null without a token, with a storage that throws, or without a storage', async () => {
	assert.equal(await withGlobal('localStorage', { getItem: () => null }, () => readToken()), null);
	assert.equal(await withGlobal('localStorage', { getItem: () => '' }, () => readToken()), null);
	const throwing = {
		getItem: () => {
			throw new Error('SecurityError');
		}
	};
	assert.equal(await withGlobal('localStorage', throwing, () => readToken()), null);
	assert.equal(await withGlobal('localStorage', {}, () => readToken()), null);
	assert.equal(await withGlobal('localStorage', undefined, () => readToken()), null);
});

function fakeFetch(reply) {
	const calls = [];
	const fn = async (url, init) => {
		calls.push({ url, init });
		return reply(url, init);
	};
	fn.calls = calls;
	return fn;
}

function jsonResponse(status, body) {
	return { ok: status >= 200 && status < 300, status, json: async () => body };
}

test('fetchPermissions sends a GET with the bearer token and the cookies, and returns the data', async () => {
	const fetchImpl = fakeFetch(() => jsonResponse(200, payload()));

	const result = await withCall(fetchImpl, () => fetchPermissions({ pluginId: 'model-browser-c' }), 'tok-123');

	assert.deepEqual(result, { ok: true, data: payload().data });
	assert.equal(fetchImpl.calls.length, 1);
	const { url, init } = fetchImpl.calls[0];
	assert.equal(url, '/api/plugins/model-browser-c/permissions');
	assert.equal(init.method, 'GET');
	assert.equal(init.credentials, 'include');
	assert.equal(init.headers.Authorization, 'Bearer tok-123');
	assert.equal(init.body, undefined);
	assert.ok(!url.includes('tok-123'), 'the token must stay out of the URL');
});

test('fetchPermissions asks the plugin it is given and sends no header without a token', async () => {
	const fetchImpl = fakeFetch(() => jsonResponse(200, payload()));
	await withCall(fetchImpl, () => fetchPermissions({ pluginId: 'other-plugin' }), null);
	assert.equal(fetchImpl.calls[0].url, '/api/plugins/other-plugin/permissions');
	assert.equal('Authorization' in fetchImpl.calls[0].init.headers, false);
});

test('savePermissions sends a PUT with the JSON body and returns the stored data', async () => {
	const body = toPutBody(toFormState(payload()));
	const fetchImpl = fakeFetch(() => jsonResponse(200, payload()));

	const result = await withCall(fetchImpl, () => savePermissions(body, { pluginId: 'model-browser-c' }), 'tok-123');

	assert.deepEqual(result, { ok: true, data: payload().data });
	const { url, init } = fetchImpl.calls[0];
	assert.equal(url, '/api/plugins/model-browser-c/permissions');
	assert.equal(init.method, 'PUT');
	assert.equal(init.headers['Content-Type'], 'application/json');
	assert.equal(init.headers.Authorization, 'Bearer tok-123');
	assert.equal(init.credentials, 'include');
	assert.deepEqual(JSON.parse(init.body), body);
});

test('a failure body from the plugin gives its message', async () => {
	const fetchImpl = fakeFetch(() =>
		jsonResponse(400, {
			detail: { success: false, error: 'unknown_group', message: "There is no user group with the id 'x'" }
		})
	);
	assert.deepEqual(await withCall(fetchImpl, () => savePermissions({ groups: [] }, { pluginId: 'model-browser-c' })), {
		ok: false,
		status: 400,
		error: "There is no user group with the id 'x'"
	});
});

function fetchedWith(fetchImpl) {
	return withCall(fetchImpl, () => fetchPermissions({ pluginId: 'model-browser-c' }));
}

test('a plain FastAPI detail gives its text', async () => {
	const fetchImpl = fakeFetch(() => jsonResponse(403, { detail: 'Administrator privileges required' }));
	assert.deepEqual(await fetchedWith(fetchImpl), {
		ok: false,
		status: 403,
		error: 'Administrator privileges required'
	});
});

test('a validation list gives the messages joined', async () => {
	const fetchImpl = fakeFetch(() =>
		jsonResponse(422, { detail: [{ loc: ['body'], msg: 'Field required' }, { loc: ['body', 'x'], msg: 'Bad value' }] })
	);
	const result = await withCall(fetchImpl, () => savePermissions({}, { pluginId: 'model-browser-c' }));
	assert.deepEqual(result, { ok: false, status: 422, error: 'Field required; Bad value' });
});

test('a failure without a readable body is explained from its status', async () => {
	const none = (status) => fakeFetch(() => ({ ok: false, status, json: async () => Promise.reject(new SyntaxError('Unexpected token <')) }));

	assert.equal((await fetchedWith(none(401))).error, 'Your session has expired: sign in again');
	assert.equal((await fetchedWith(none(403))).error, 'Only administrators can edit these permissions');
	assert.equal((await fetchedWith(none(502))).error, 'The server answered with an error (HTTP 502)');
});

test('an unreachable server is reported with status 0', async () => {
	const fetchImpl = fakeFetch(() => {
		throw new TypeError('Failed to fetch');
	});
	assert.deepEqual(await fetchedWith(fetchImpl), { ok: false, status: 0, error: 'Could not reach the server' });

	const other = fakeFetch(() => {
		throw new Error('boom');
	});
	assert.deepEqual(await fetchedWith(other), { ok: false, status: 0, error: 'boom' });

	const odd = fakeFetch(() => {
		throw undefined;
	});
	assert.deepEqual(await fetchedWith(odd), { ok: false, status: 0, error: 'Request failed' });
});

test('a success status with a body that is not a permissions answer is an error', async () => {
	const bodies = [null, 'ok', [], {}, { success: true }, { success: true, data: {} }, { success: false, data: payload().data }];
	for (const body of bodies) {
		const fetchImpl = fakeFetch(() => jsonResponse(200, body));
		assert.deepEqual(
			await fetchedWith(fetchImpl),
			{ ok: false, status: 200, error: 'Unexpected response from the server' },
			JSON.stringify(body)
		);
	}
	const html = fakeFetch(() => ({ ok: true, status: 200, json: async () => Promise.reject(new SyntaxError('Unexpected token <')) }));
	assert.equal((await fetchedWith(html)).error, 'Unexpected response from the server');
});

test('the calls never reject', async () => {
	const result = await fetchedWith(() => Promise.reject(new Error('offline')));
	assert.equal(result.ok, false);
});
