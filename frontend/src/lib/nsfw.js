export const MODES = ['blur', 'show', 'hide'];

export const MODE_LABELS = { blur: 'Blur', show: 'Show', hide: 'Hide' };

const CEILING_RANK = { allowed: 0, blur: 1, blocked: 2 };

export function nsfwInfo(me) {
	const nsfw = me?.nsfw ?? {};
	const ceiling = nsfw.ceiling in CEILING_RANK ? nsfw.ceiling : 'blocked';
	const allowed = ceiling === 'blocked' ? ['hide'] : (nsfw.allowed_modes ?? []).filter((mode) => MODES.includes(mode));
	return {
		ceiling,
		mode: effectiveMode(me),
		allowed: allowed.length ? allowed : ['hide'],
		locked: ceiling === 'blocked',
		settingKey: typeof nsfw.setting_key === 'string' && nsfw.setting_key ? nsfw.setting_key : 'media_nsfw_filter_mode'
	};
}

export function effectiveMode(me) {
	const nsfw = me?.nsfw ?? {};
	const ceiling = nsfw.ceiling in CEILING_RANK ? nsfw.ceiling : 'blocked';
	if (ceiling === 'blocked') return 'hide';
	const mode = MODES.includes(nsfw.mode) ? nsfw.mode : 'hide';
	if (ceiling === 'blur' && mode === 'show') return 'blur';
	return mode;
}

export function effectiveModes(me) {
	const info = nsfwInfo(me);
	if (info.locked) return [];
	return MODES.map((id) => ({ id, label: MODE_LABELS[id], disabled: !info.allowed.includes(id) }));
}

export function isRevealable(image, me) {
	if (!image || image.nsfw !== true || image.url) return false;
	const info = nsfwInfo(me);
	return !info.locked && effectiveMode(me) === 'blur';
}

export function mediaState(image, revealedUrl) {
	if (!image) return 'empty';
	if (image.url || revealedUrl) return 'media';
	return image.nsfw ? 'blurred' : 'empty';
}

export function createRevealCache(fetchMedia) {
	const requests = new Map();

	function load(source, modelId) {
		const key = `${source}\u0000${modelId}`;
		let pending = requests.get(key);
		if (!pending) {
			pending = Promise.resolve()
				.then(() => fetchMedia(source, modelId))
				.then((result) => {
					if (!result?.ok) {
						requests.delete(key);
						return { ok: false, error: result?.error ?? { message: 'Could not reveal the media' } };
					}
					const urls = new Map();
					for (const image of result.data?.images ?? []) {
						if (image && image.id !== null && image.id !== undefined && typeof image.url === 'string' && image.url) {
							urls.set(String(image.id), image.url);
						}
					}
					return { ok: true, urls };
				})
				.catch((err) => {
					requests.delete(key);
					return { ok: false, error: { message: err instanceof Error ? err.message : 'Could not reveal the media' } };
				});
			requests.set(key, pending);
		}
		return pending;
	}

	return {
		urls: load,
		async urlFor(source, modelId, imageId) {
			const result = await load(source, modelId);
			if (!result.ok) return { error: result.error };
			const url = imageId === null || imageId === undefined ? null : result.urls.get(String(imageId)) ?? null;
			return { url };
		},
		reset() {
			requests.clear();
		}
	};
}
