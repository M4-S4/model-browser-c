import { test } from 'node:test';
import assert from 'node:assert/strict';
import { getModel, getNsfwMedia } from './client.js';
import { hasPartialVersions, isPartial, mediaModelId, mergeVersion, versionRef } from './versions.js';

function full(id, extra = {}) {
	return {
		id,
		name: `Version ${id}`,
		base_model: 'SDXL 1.0',
		trigger_words: [`trigger ${id}`],
		created_at: '2024-05-01',
		early_access: false,
		nsfw: false,
		images: [{ id: Number(id), url: `https://example.test/${id}.jpg`, nsfw: false }],
		files: [{ id: `file-${id}`, name: `model-${id}.safetensors`, primary: true }],
		partial: false,
		...extra
	};
}

function shell(id, extra = {}) {
	return {
		id,
		name: `Version ${id}`,
		base_model: null,
		trigger_words: [],
		created_at: '2023-01-01',
		early_access: false,
		nsfw: false,
		images: [],
		files: [],
		partial: true,
		...extra
	};
}

function drawerDetail() {
	return {
		source: 'civarchive',
		id: '78291',
		name: 'A model',
		description_html: '<p>About</p>',
		gated: false,
		versions: [full('300'), shell('200'), shell('100')]
	};
}

function answerFor(id) {
	const others = ['300', '200', '100'].filter((other) => other !== id);
	return { ...drawerDetail(), versions: [full(id), ...others.map((other) => shell(other))] };
}

test('only partial: true makes a version a shell', () => {
	assert.equal(isPartial(shell('1')), true);
	for (const version of [full('1'), { id: '1' }, { id: '1', partial: false }, { id: '1', partial: 'true' }, { id: '1', partial: 1 }, null, undefined, 'x', []]) {
		assert.equal(isPartial(version), false);
	}
	assert.equal(hasPartialVersions(drawerDetail()), true);
	assert.equal(hasPartialVersions({ versions: [{ id: '1', files: [], images: [] }, { id: '2', partial: false }] }), false);
	assert.equal(hasPartialVersions({ versions: [] }), false);
	assert.equal(hasPartialVersions({}), false);
	assert.equal(hasPartialVersions(null), false);
});

test('versionRef and mediaModelId: a version of its own id only when the versions come one by one', () => {
	assert.equal(versionRef('78291', '83095'), '78291:83095');
	assert.equal(versionRef(78291, 83095), '78291:83095');
	const detail = drawerDetail();
	assert.equal(mediaModelId(detail, detail.versions[0], true), '78291:300');
	assert.equal(mediaModelId(detail, detail.versions[1], true), '78291:200');
	assert.equal(mediaModelId(detail, detail.versions[1], false), '78291');
	assert.equal(mediaModelId({ id: 58390 }, { id: '1' }, false), 58390);
	assert.equal(mediaModelId(detail, null, true), '78291');
});

test('mergeVersion puts the full version in the place of its shell and keeps everything else', () => {
	const detail = drawerDetail();
	const snapshot = structuredClone(detail);
	const result = mergeVersion(detail, '200', answerFor('200'));
	assert.equal(result.ok, true);
	const merged = result.detail;
	assert.deepEqual(
		merged.versions.map((version) => version.id),
		['300', '200', '100']
	);
	assert.deepEqual(merged.versions[1], full('200'));
	assert.equal(merged.versions[1].partial, false);
	assert.equal(merged.versions[1].files.length, 1);
	assert.equal(merged.versions[1].images.length, 1);
	assert.equal(merged.versions[0], detail.versions[0]);
	assert.equal(merged.versions[2], detail.versions[2]);
	assert.equal(isPartial(merged.versions[0]), false);
	assert.equal(isPartial(merged.versions[2]), true);
	assert.equal(merged.name, 'A model');
	assert.equal(merged.description_html, '<p>About</p>');
	assert.notEqual(merged, detail);
	assert.notEqual(merged.versions, detail.versions);
	assert.deepEqual(detail, snapshot);
});

test('mergeVersion clears partial, even when the full version carries no flag', () => {
	const answer = answerFor('200');
	delete answer.versions[0].partial;
	const merged = mergeVersion(drawerDetail(), '200', answer).detail;
	assert.equal(merged.versions[1].partial, false);
	assert.equal(isPartial(merged.versions[1]), false);
});

test('versions merged one after the other all stay merged', () => {
	let detail = drawerDetail();
	for (const id of ['100', '200']) {
		const result = mergeVersion(detail, id, answerFor(id));
		assert.equal(result.ok, true);
		detail = result.detail;
	}
	assert.deepEqual(
		detail.versions.map((version) => [version.id, isPartial(version), version.files.length]),
		[
			['300', false, 1],
			['200', false, 1],
			['100', false, 1]
		]
	);
	assert.equal(hasPartialVersions(detail), false);
	assert.equal(mediaModelId(detail, detail.versions[2], true), '78291:100');
});

test('mergeVersion compares ids as text', () => {
	const result = mergeVersion(drawerDetail(), 200, { ...answerFor('200'), id: 78291 });
	assert.equal(result.ok, true);
	assert.equal(result.detail.versions[1].partial, false);
});

test('mergeVersion refuses an answer that does not hold the version in full, and changes nothing', () => {
	const detail = drawerDetail();
	const snapshot = structuredClone(detail);
	const refusal = (result) => {
		assert.equal(result.ok, false);
		assert.deepEqual(Object.keys(result.error).sort(), ['code', 'message', 'retryAfter', 'status']);
		assert.equal(result.error.status, 0);
		assert.equal(result.error.code, 'invalid_response');
		assert.ok(result.error.message);
		return result.error.message;
	};
	assert.match(refusal(mergeVersion(detail, '200', answerFor('300'))), /without its files/);
	assert.match(refusal(mergeVersion(detail, '200', { ...answerFor('300'), versions: [full('300')] })), /did not send/);
	assert.match(refusal(mergeVersion(detail, '200', { ...answerFor('200'), versions: [] })), /did not send/);
	assert.match(refusal(mergeVersion(detail, '200', { ...answerFor('200'), id: '99' })), /another model/);
	for (const answer of [null, undefined, 'text', 7, [], {}, { versions: 'x' }]) {
		assert.match(refusal(mergeVersion(detail, '200', answer)), /Unexpected answer/);
	}
	assert.match(refusal(mergeVersion(detail, '999', answerFor('999'))), /not in the list/);
	assert.match(refusal(mergeVersion(detail, '', answerFor('200'))), /not in the list/);
	assert.match(refusal(mergeVersion(null, '200', answerFor('200'))), /not in the list/);
	assert.deepEqual(detail, snapshot);
});

test('a shell is asked for by its composite id on the detail route, and its previews on the same id', async () => {
	const asked = [];
	const api = {
		get: async (path) => {
			asked.push(path);
			return { success: true, data: path.endsWith('/nsfw-media') ? { images: [] } : answerFor('200') };
		}
	};
	const detail = drawerDetail();
	const byVersion = hasPartialVersions(detail);
	const shown = await getModel(api, detail.source, versionRef(detail.id, '200'));
	assert.equal(shown.ok, true);
	const merged = mergeVersion(detail, '200', shown.data);
	assert.equal(merged.ok, true);
	await getNsfwMedia(api, detail.source, mediaModelId(merged.detail, merged.detail.versions[1], byVersion));
	assert.deepEqual(asked, ['/civarchive/models/78291%3A200', '/civarchive/models/78291%3A200/nsfw-media']);
});

test('a detail without shells (CivitAI, Hugging Face, OpenModelDB) keeps its old ways', () => {
	const detail = {
		source: 'civitai',
		id: 58390,
		versions: [
			{ id: '1', files: [{ id: 'a' }], images: [] },
			{ id: '2', files: [], images: [], partial: false }
		]
	};
	assert.equal(hasPartialVersions(detail), false);
	assert.equal(detail.versions.some(isPartial), false);
	assert.equal(mediaModelId(detail, detail.versions[1], hasPartialVersions(detail)), 58390);
});
