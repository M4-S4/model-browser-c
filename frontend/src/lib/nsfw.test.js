import { test } from 'node:test';
import assert from 'node:assert/strict';
import { normalizeMe } from './capabilities.js';
import {
	createRevealCache,
	effectiveMode,
	effectiveModes,
	isRevealable,
	mediaState,
	modeHint,
	nsfwInfo
} from './nsfw.js';

const ALLOWED_MODES = { allowed: ['blur', 'show', 'hide'], blur: ['blur', 'hide'], blocked: ['hide'] };

function me(ceiling, mode) {
	return normalizeMe({
		capabilities: ['browse'],
		nsfw: { ceiling, mode, allowed_modes: ALLOWED_MODES[ceiling], max_level: 2, hide_poi: false, setting_key: 'media_nsfw_filter_mode' }
	});
}

const BLURRED = { id: '42', url: null, blurhash: 'LEHV6nWB2yk8pyo0adR*.7kCMdnj', nsfw: true, nsfw_level: 8, kind: 'image' };
const SAFE = { id: '7', url: 'https://image.civitai.com/a/b/original=true/7.jpeg', nsfw: false, kind: 'image' };
const SHOWN_NSFW = { ...BLURRED, url: 'https://image.civitai.com/a/b/original=true/42.jpeg' };

test('effectiveModes follows the matrix: every mode under allowed', () => {
	assert.deepEqual(effectiveModes(me('allowed', 'blur')), [
		{ id: 'blur', label: 'Blur', disabled: false },
		{ id: 'show', label: 'Show', disabled: false },
		{ id: 'hide', label: 'Hide', disabled: false }
	]);
});

test('effectiveModes follows the matrix: no Show under a blur ceiling', () => {
	assert.deepEqual(
		effectiveModes(me('blur', 'blur')).map((mode) => [mode.id, mode.disabled]),
		[
			['blur', false],
			['show', true],
			['hide', false]
		]
	);
});

test('effectiveModes follows the matrix: nothing to choose under a blocked ceiling', () => {
	assert.deepEqual(effectiveModes(me('blocked', 'hide')), []);
	assert.equal(nsfwInfo(me('blocked', 'show')).locked, true);
});

test('effectiveMode bounds the preference by the ceiling, like the server', () => {
	assert.equal(effectiveMode(me('allowed', 'show')), 'show');
	assert.equal(effectiveMode(me('allowed', 'hide')), 'hide');
	assert.equal(effectiveMode({ nsfw: { ceiling: 'blur', mode: 'show' } }), 'blur');
	assert.equal(effectiveMode({ nsfw: { ceiling: 'blocked', mode: 'show' } }), 'hide');
	assert.equal(effectiveMode(null), 'hide');
});

test('nsfwInfo reports the setting key, with a default', () => {
	assert.equal(nsfwInfo(me('allowed', 'blur')).settingKey, 'media_nsfw_filter_mode');
	assert.equal(nsfwInfo({ nsfw: { ceiling: 'allowed', mode: 'blur', allowed_modes: ['blur'] } }).settingKey, 'media_nsfw_filter_mode');
	assert.deepEqual(nsfwInfo(me('blur', 'blur')).allowed, ['blur', 'hide']);
});

test('modeHint explains a disabled mode', () => {
	assert.equal(modeHint(me('allowed', 'blur'), 'show'), '');
	assert.match(modeHint(me('blur', 'blur'), 'show'), /blurred NSFW media at most/);
	assert.match(modeHint(me('blocked', 'hide'), 'blur'), /hides NSFW media/);
});

test('isRevealable: only a url-less NSFW image, in blur mode, under a ceiling that is not blocked', () => {
	assert.equal(isRevealable(BLURRED, me('allowed', 'blur')), true);
	assert.equal(isRevealable(BLURRED, me('blur', 'blur')), true);
	assert.equal(isRevealable(BLURRED, me('blocked', 'hide')), false);
	assert.equal(isRevealable(BLURRED, me('allowed', 'hide')), false);
	assert.equal(isRevealable(SHOWN_NSFW, me('allowed', 'blur')), false);
	assert.equal(isRevealable(SAFE, me('allowed', 'blur')), false);
	assert.equal(isRevealable(null, me('allowed', 'blur')), false);
});

test('mediaState: media when there is a url or a revealed one, blurred for NSFW, empty otherwise', () => {
	assert.equal(mediaState(SAFE), 'media');
	assert.equal(mediaState(SHOWN_NSFW), 'media');
	assert.equal(mediaState(BLURRED), 'blurred');
	assert.equal(mediaState(BLURRED, 'https://image.civitai.com/x/y/original=true/42.jpeg'), 'media');
	assert.equal(mediaState({ id: '1', url: null, nsfw: false }), 'empty');
	assert.equal(mediaState(null), 'empty');
});

function mediaAnswer(images) {
	return { ok: true, data: { images } };
}

test('createRevealCache asks once per model and maps urls by image id', async () => {
	const calls = [];
	const cache = createRevealCache(async (source, modelId) => {
		calls.push(`${source}/${modelId}`);
		return mediaAnswer([
			{ id: '42', url: 'https://cdn/42.jpeg', nsfw: true },
			{ id: 43, url: 'https://cdn/43.mp4', nsfw: true, kind: 'video' },
			{ id: null, url: 'https://cdn/anonymous.jpeg', nsfw: true },
			{ id: '44', url: null, nsfw: true }
		]);
	});
	assert.deepEqual(await cache.urlFor('civitai', '100', '42'), { url: 'https://cdn/42.jpeg' });
	assert.deepEqual(await cache.urlFor('civitai', '100', 43), { url: 'https://cdn/43.mp4' });
	assert.deepEqual(await cache.urlFor('civitai', '100', '44'), { url: null });
	assert.deepEqual(await cache.urlFor('civitai', '100', null), { url: null });
	assert.deepEqual(calls, ['civitai/100']);

	await cache.urlFor('huggingface', 'org/repo', 'abc');
	assert.deepEqual(calls, ['civitai/100', 'huggingface/org/repo']);
});

test('createRevealCache shares one request between concurrent clicks', async () => {
	let calls = 0;
	let release;
	const cache = createRevealCache(() => {
		calls += 1;
		return new Promise((resolve) => {
			release = () => resolve(mediaAnswer([{ id: '1', url: 'https://cdn/1.jpeg' }]));
		});
	});
	const first = cache.urlFor('civitai', '5', '1');
	const second = cache.urlFor('civitai', '5', '1');
	await Promise.resolve();
	await Promise.resolve();
	release();
	assert.deepEqual(await first, { url: 'https://cdn/1.jpeg' });
	assert.deepEqual(await second, { url: 'https://cdn/1.jpeg' });
	assert.equal(calls, 1);
});

test('createRevealCache forgets a failed request so the next click asks again', async () => {
	let calls = 0;
	const cache = createRevealCache(async () => {
		calls += 1;
		if (calls === 1) return { ok: false, error: { status: 404, code: 'not_found', message: 'Model not found' } };
		return mediaAnswer([{ id: '9', url: 'https://cdn/9.jpeg' }]);
	});
	assert.deepEqual(await cache.urlFor('civitai', '9', '9'), { error: { status: 404, code: 'not_found', message: 'Model not found' } });
	assert.deepEqual(await cache.urlFor('civitai', '9', '9'), { url: 'https://cdn/9.jpeg' });
	assert.equal(calls, 2);
});

test('createRevealCache turns a thrown error into an error result, and reset forgets answers', async () => {
	let calls = 0;
	const cache = createRevealCache(async () => {
		calls += 1;
		if (calls === 1) throw new Error('offline');
		return mediaAnswer([]);
	});
	assert.deepEqual(await cache.urlFor('civitai', '1', '1'), { error: { message: 'offline' } });
	assert.deepEqual(await cache.urlFor('civitai', '1', '1'), { url: null });
	assert.equal(calls, 2);
	await cache.urlFor('civitai', '1', '1');
	assert.equal(calls, 2);
	cache.reset();
	await cache.urlFor('civitai', '1', '1');
	assert.equal(calls, 3);
});
