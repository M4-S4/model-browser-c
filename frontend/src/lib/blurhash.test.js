import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
	BLURHASH_SIZE,
	BlurhashError,
	averageColor,
	decode83,
	decodeBlurhash,
	decodeCached,
	encode83,
	linearToSRGB,
	sRGBToLinear,
	validateBlurhash
} from './blurhash.js';

const SAMPLE = 'LEHV6nWB2yk8pyo0adR*.7kCMdnj';

function pixel(pixels, width, x, y) {
	const offset = 4 * (x + y * width);
	return [...pixels.slice(offset, offset + 4)];
}

function flatHash(r, g, b) {
	return `00${encode83((r << 16) + (g << 8) + b, 4)}`;
}

test('decode83 reads base 83 digits', () => {
	assert.equal(decode83('0'), 0);
	assert.equal(decode83('~'), 82);
	assert.equal(decode83('10'), 83);
	assert.equal(decode83('HV6n'), 9_934_485);
	assert.throws(() => decode83('a!'), BlurhashError);
});

test('encode83 is the inverse of decode83', () => {
	for (const value of [0, 1, 82, 83, 9_934_485, 83 ** 4 - 1]) {
		assert.equal(decode83(encode83(value, 4)), value);
	}
	assert.equal(encode83(9_934_485, 4), 'HV6n');
});

test('the sRGB conversions round-trip every channel value', () => {
	for (let value = 0; value <= 255; value += 1) {
		assert.equal(linearToSRGB(sRGBToLinear(value)), value);
	}
	assert.equal(linearToSRGB(-1), 0);
	assert.equal(linearToSRGB(2), 255);
});

test('validateBlurhash accepts the sample and reports its components', () => {
	assert.deepEqual(validateBlurhash(SAMPLE), { ok: true, numX: 4, numY: 3 });
});

test('validateBlurhash explains what is wrong', () => {
	assert.equal(validateBlurhash(null).ok, false);
	assert.match(validateBlurhash('LEHV6').error, /at least 6/);
	assert.match(validateBlurhash(SAMPLE.slice(0, -1)).error, /4x3 blurhash has 28 characters, not 27/);
	assert.match(validateBlurhash(`${SAMPLE}x`).error, /not 29/);
	assert.match(validateBlurhash('LEHV6nWB2yk8pyo0adR*.7kCMd"j').error, /not a base 83 digit/);
});

test('averageColor is the DC component', () => {
	assert.deepEqual(averageColor(SAMPLE), [151, 150, 149]);
	assert.throws(() => averageColor('nope'), BlurhashError);
});

test('decodeBlurhash fills every pixel of the sample, opaque, with a picture that is not flat', () => {
	const pixels = decodeBlurhash(SAMPLE, BLURHASH_SIZE, BLURHASH_SIZE);
	assert.equal(pixels.length, BLURHASH_SIZE * BLURHASH_SIZE * 4);
	assert.ok(pixels instanceof Uint8ClampedArray);

	let nonBlack = 0;
	const distinct = new Set();
	let sum = [0, 0, 0];
	for (let offset = 0; offset < pixels.length; offset += 4) {
		assert.equal(pixels[offset + 3], 255, 'alpha');
		if (pixels[offset] + pixels[offset + 1] + pixels[offset + 2] > 0) nonBlack += 1;
		distinct.add(`${pixels[offset]},${pixels[offset + 1]},${pixels[offset + 2]}`);
		sum = [sum[0] + pixels[offset], sum[1] + pixels[offset + 1], sum[2] + pixels[offset + 2]];
	}
	assert.equal(nonBlack, BLURHASH_SIZE * BLURHASH_SIZE);
	assert.ok(distinct.size > 50, `a varied picture (${distinct.size} colours)`);
	const mean = sum.map((total) => total / (BLURHASH_SIZE * BLURHASH_SIZE));
	for (const [index, channel] of [151, 150, 149].entries()) {
		assert.ok(Math.abs(mean[index] - channel) < 40, `channel ${index}: ${mean[index]} vs ${channel}`);
	}
});

test('decodeBlurhash is deterministic and honours the size asked for', () => {
	const a = decodeBlurhash(SAMPLE, 8, 6);
	const b = decodeBlurhash(SAMPLE, 8, 6);
	assert.equal(a.length, 8 * 6 * 4);
	assert.deepEqual(a, b);
});

test('a hash with only an average colour decodes to that exact colour everywhere', () => {
	const pixels = decodeBlurhash(flatHash(255, 0, 0), 4, 4);
	for (let y = 0; y < 4; y += 1) {
		for (let x = 0; x < 4; x += 1) assert.deepEqual(pixel(pixels, 4, x, y), [255, 0, 0, 255]);
	}
	const grey = decodeBlurhash(flatHash(12, 200, 99), 2, 2);
	assert.deepEqual(pixel(grey, 2, 1, 1), [12, 200, 99, 255]);
});

test('a horizontal AC component brightens the left edge and leaves the middle at the average colour', () => {
	const ac = 18 * 19 * 19 + 9 * 19 + 9;
	const hash = `1~${encode83((64 << 16) + (64 << 8) + 64, 4)}${encode83(ac, 2)}`;
	assert.deepEqual(validateBlurhash(hash), { ok: true, numX: 2, numY: 1 });
	const pixels = decodeBlurhash(hash, 2, 1);
	assert.deepEqual(pixel(pixels, 2, 1, 0), [64, 64, 64, 255]);
	const left = pixel(pixels, 2, 0, 0);
	assert.equal(left[0], linearToSRGB(sRGBToLinear(64) + 0.5));
	assert.equal(left[1], 64);
	assert.equal(left[2], 64);
});

test('punch scales the AC components', () => {
	const ac = 18 * 19 * 19 + 9 * 19 + 9;
	const hash = `1F${encode83((64 << 16) + (64 << 8) + 64, 4)}${encode83(ac, 2)}`;
	const soft = pixel(decodeBlurhash(hash, 2, 1, 1), 2, 0, 0)[0];
	const strong = pixel(decodeBlurhash(hash, 2, 1, 2), 2, 0, 0)[0];
	assert.ok(strong > soft, `${strong} > ${soft}`);
});

test('decodeBlurhash refuses malformed hashes and bad sizes', () => {
	assert.throws(() => decodeBlurhash('', 32, 32), BlurhashError);
	assert.throws(() => decodeBlurhash(SAMPLE.slice(0, 20), 32, 32), /28 characters/);
	assert.throws(() => decodeBlurhash(SAMPLE, 0, 32), /positive whole numbers/);
	assert.throws(() => decodeBlurhash(SAMPLE, 3.5, 32), /positive whole numbers/);
});

test('decodeCached remembers decoded hashes and gives null for bad ones', () => {
	const first = decodeCached(SAMPLE);
	assert.ok(first instanceof Uint8ClampedArray);
	assert.equal(decodeCached(SAMPLE), first);
	assert.equal(decodeCached('bad'), null);
	assert.equal(decodeCached(undefined), null);
});
