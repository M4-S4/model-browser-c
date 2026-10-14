const DIGITS = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz#$%*+,-.:;=?@[]^_{|}~';
const DIGIT_VALUES = new Map([...DIGITS].map((char, index) => [char, index]));

export const BLURHASH_SIZE = 32;

export class BlurhashError extends Error {
	constructor(message) {
		super(message);
		this.name = 'BlurhashError';
	}
}

export function decode83(text) {
	let value = 0;
	for (const char of text) {
		const digit = DIGIT_VALUES.get(char);
		if (digit === undefined) throw new BlurhashError(`'${char}' is not a base 83 digit`);
		value = value * 83 + digit;
	}
	return value;
}

export function encode83(value, length) {
	let result = '';
	for (let i = 1; i <= length; i += 1) {
		const digit = Math.floor(value / 83 ** (length - i)) % 83;
		result += DIGITS[digit];
	}
	return result;
}

export function sRGBToLinear(value) {
	const v = value / 255;
	return v <= 0.04045 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4;
}

export function linearToSRGB(value) {
	const v = Math.max(0, Math.min(1, value));
	return v <= 0.0031308 ? Math.trunc(v * 12.92 * 255 + 0.5) : Math.trunc((1.055 * v ** (1 / 2.4) - 0.055) * 255 + 0.5);
}

function signPow(value, exponent) {
	return Math.sign(value) * Math.abs(value) ** exponent;
}

export function validateBlurhash(hash) {
	if (typeof hash !== 'string') return { ok: false, error: 'A blurhash is a string' };
	if (hash.length < 6) return { ok: false, error: 'A blurhash has at least 6 characters' };
	for (const char of hash) {
		if (!DIGIT_VALUES.has(char)) return { ok: false, error: `'${char}' is not a base 83 digit` };
	}
	const sizeFlag = decode83(hash[0]);
	const numY = Math.floor(sizeFlag / 9) + 1;
	const numX = (sizeFlag % 9) + 1;
	const expected = 4 + 2 * numX * numY;
	if (hash.length !== expected) {
		return { ok: false, error: `A ${numX}x${numY} blurhash has ${expected} characters, not ${hash.length}` };
	}
	return { ok: true, numX, numY };
}

export function averageColor(hash) {
	const check = validateBlurhash(hash);
	if (!check.ok) throw new BlurhashError(check.error);
	const value = decode83(hash.slice(2, 6));
	return [value >> 16, (value >> 8) & 255, value & 255];
}

export function decodeBlurhash(hash, width, height, punch = 1) {
	const check = validateBlurhash(hash);
	if (!check.ok) throw new BlurhashError(check.error);
	if (!Number.isInteger(width) || !Number.isInteger(height) || width <= 0 || height <= 0) {
		throw new BlurhashError('The size must be positive whole numbers');
	}
	const { numX, numY } = check;
	const maximumValue = (decode83(hash[1]) + 1) / 166;
	const contrast = Number.isFinite(punch) && punch > 0 ? punch : 1;

	const colors = new Array(numX * numY);
	const dc = decode83(hash.slice(2, 6));
	colors[0] = [sRGBToLinear(dc >> 16), sRGBToLinear((dc >> 8) & 255), sRGBToLinear(dc & 255)];
	for (let i = 1; i < colors.length; i += 1) {
		const value = decode83(hash.slice(4 + i * 2, 6 + i * 2));
		const quantR = Math.floor(value / (19 * 19));
		const quantG = Math.floor(value / 19) % 19;
		const quantB = value % 19;
		colors[i] = [
			signPow((quantR - 9) / 9, 2) * maximumValue * contrast,
			signPow((quantG - 9) / 9, 2) * maximumValue * contrast,
			signPow((quantB - 9) / 9, 2) * maximumValue * contrast
		];
	}

	const cosX = new Float64Array(width * numX);
	for (let x = 0; x < width; x += 1) {
		for (let i = 0; i < numX; i += 1) cosX[x * numX + i] = Math.cos((Math.PI * x * i) / width);
	}
	const cosY = new Float64Array(height * numY);
	for (let y = 0; y < height; y += 1) {
		for (let j = 0; j < numY; j += 1) cosY[y * numY + j] = Math.cos((Math.PI * y * j) / height);
	}

	const pixels = new Uint8ClampedArray(width * height * 4);
	for (let y = 0; y < height; y += 1) {
		for (let x = 0; x < width; x += 1) {
			let r = 0;
			let g = 0;
			let b = 0;
			for (let j = 0; j < numY; j += 1) {
				const cy = cosY[y * numY + j];
				for (let i = 0; i < numX; i += 1) {
					const basis = cosX[x * numX + i] * cy;
					const color = colors[i + j * numX];
					r += color[0] * basis;
					g += color[1] * basis;
					b += color[2] * basis;
				}
			}
			const offset = 4 * (x + y * width);
			pixels[offset] = linearToSRGB(r);
			pixels[offset + 1] = linearToSRGB(g);
			pixels[offset + 2] = linearToSRGB(b);
			pixels[offset + 3] = 255;
		}
	}
	return pixels;
}

const cache = new Map();
const CACHE_LIMIT = 256;

export function decodeCached(hash) {
	if (typeof hash !== 'string') return null;
	if (cache.has(hash)) return cache.get(hash);
	let pixels = null;
	try {
		pixels = decodeBlurhash(hash, BLURHASH_SIZE, BLURHASH_SIZE);
	} catch {
		pixels = null;
	}
	if (cache.size >= CACHE_LIMIT) cache.delete(cache.keys().next().value);
	cache.set(hash, pixels);
	return pixels;
}
