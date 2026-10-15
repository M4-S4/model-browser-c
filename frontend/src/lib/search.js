export const DEFAULT_LIMIT = 24;

export const URL_KEYS = {
	source: 'source',
	q: 'q',
	type: 'type',
	base: 'base',
	sort: 'sort',
	period: 'period',
	model: 'model'
};

const SOURCE_ICONS = {
	civitai: 'palette',
	huggingface: 'smile',
	openmodeldb: 'database',
	civarchive: 'library'
};

export const DEFAULT_TYPES_LABEL = 'Types';
export const DEFAULT_BASE_MODELS_LABEL = 'Base model';
export const DEFAULT_BASE_MODEL_LABEL = 'Base model';

const SHA256 = /^[0-9a-f]{64}$/;

function isObject(value) {
	return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function text(value) {
	return typeof value === 'string' ? value : '';
}

export function emptyState(source = '') {
	return { source, q: '', types: [], baseModel: '', sort: '', period: '', model: '' };
}

export function enabledSources(sources) {
	if (!Array.isArray(sources)) return [];
	const options = (list) =>
		(Array.isArray(list) ? list : [])
			.filter((option) => isObject(option) && typeof option.value === 'string' && option.value)
			.map((option) => ({ value: option.value, label: text(option.label) || option.value }));
	return sources
		.filter((source) => isObject(source) && typeof source.id === 'string' && source.id)
		.map((source) => ({
			id: source.id,
			label: text(source.label) || source.id,
			types: options(source.types),
			base_models: options(source.base_models),
			sorts: options(source.sorts),
			periods: options(source.periods),
			search_help: text(source.search_help),
			notice: text(source.notice).trim(),
			types_label: text(source.types_label).trim() || DEFAULT_TYPES_LABEL,
			base_models_label: text(source.base_models_label).trim() || DEFAULT_BASE_MODELS_LABEL,
			base_model_label: text(source.base_model_label).trim() || DEFAULT_BASE_MODEL_LABEL
		}));
}

export function sourceSections(sources) {
	return sources.map((source) => ({ id: source.id, label: source.label, icon: SOURCE_ICONS[source.id] ?? 'globe' }));
}

export function sourceDefaults(info) {
	return { sort: info?.sorts?.[0]?.value ?? '', period: info?.periods?.[0]?.value ?? '' };
}

export function decodeState(search) {
	const params = search instanceof URLSearchParams ? search : new URLSearchParams(String(search ?? ''));
	return {
		source: params.get(URL_KEYS.source) ?? '',
		q: params.get(URL_KEYS.q) ?? '',
		types: [...new Set(params.getAll(URL_KEYS.type).filter(Boolean))],
		baseModel: params.get(URL_KEYS.base) ?? '',
		sort: params.get(URL_KEYS.sort) ?? '',
		period: params.get(URL_KEYS.period) ?? '',
		model: params.get(URL_KEYS.model) ?? ''
	};
}

function hasOption(list, value) {
	return Array.isArray(list) && list.some((option) => option.value === value);
}

export function normalizeState(state, sources) {
	const info = sources.find((source) => source.id === state.source) ?? sources[0];
	if (!info) return { ...emptyState(''), q: text(state.q) };
	const defaults = sourceDefaults(info);
	const sort = hasOption(info.sorts, state.sort) && state.sort !== defaults.sort ? state.sort : '';
	const period = hasOption(info.periods, state.period) && state.period !== defaults.period ? state.period : '';
	return {
		source: info.id,
		q: text(state.q),
		types: (state.types ?? []).filter((type, index, all) => hasOption(info.types, type) && all.indexOf(type) === index),
		baseModel: hasOption(info.base_models, state.baseModel) ? state.baseModel : '',
		sort,
		period,
		model: text(state.model)
	};
}

export function switchSource(state, sourceId) {
	return { ...emptyState(sourceId), q: state.q };
}

export const REQUESTS_LINK = { key: 'tab', value: 'requests' };

export function wantsRequestsTab(search) {
	const params = search instanceof URLSearchParams ? search : new URLSearchParams(String(search ?? ''));
	return params.getAll(REQUESTS_LINK.key).includes(REQUESTS_LINK.value);
}

export function withoutRequestsTab(search) {
	const params = search instanceof URLSearchParams ? search : new URLSearchParams(String(search ?? ''));
	const kept = new URLSearchParams();
	for (const [key, value] of params) {
		if (key !== REQUESTS_LINK.key || value !== REQUESTS_LINK.value) kept.append(key, value);
	}
	return kept.toString();
}

export function encodeState(state, currentSearch = '') {
	const params = new URLSearchParams(withoutRequestsTab(currentSearch));
	for (const key of Object.values(URL_KEYS)) params.delete(key);
	if (state.source) params.set(URL_KEYS.source, state.source);
	if (state.q) params.set(URL_KEYS.q, state.q);
	for (const type of state.types ?? []) params.append(URL_KEYS.type, type);
	if (state.baseModel) params.set(URL_KEYS.base, state.baseModel);
	if (state.sort) params.set(URL_KEYS.sort, state.sort);
	if (state.period) params.set(URL_KEYS.period, state.period);
	if (state.model) params.set(URL_KEYS.model, state.model);
	return params.toString();
}

export function searchRequest(state, info, { cursor = null } = {}) {
	const params = new URLSearchParams();
	const q = text(state.q).trim();
	if (q) params.set('q', q);
	for (const type of state.types ?? []) params.append('types', type);
	if (state.baseModel) params.append('base_models', state.baseModel);
	const defaults = sourceDefaults(info);
	const sort = state.sort || defaults.sort;
	if (sort) params.set('sort', sort);
	const period = info?.periods?.length ? state.period || defaults.period : '';
	if (period) params.set('period', period);
	if (cursor) params.set('cursor', cursor);
	params.set('limit', String(DEFAULT_LIMIT));
	return params.toString();
}

export function searchKey(state) {
	return JSON.stringify([
		state.source,
		text(state.q).trim(),
		[...(state.types ?? [])].sort(),
		state.baseModel ?? '',
		state.sort ?? '',
		state.period ?? ''
	]);
}

export function mergeItems(items, next) {
	const seen = new Set(items.map((item) => `${item.source}\u0000${item.id}`));
	const merged = [...items];
	for (const item of Array.isArray(next) ? next : []) {
		if (!isObject(item) || item.id === undefined || item.id === null) continue;
		const key = `${item.source}\u0000${item.id}`;
		if (seen.has(key)) continue;
		seen.add(key);
		merged.push(item);
	}
	return merged;
}

function labelOf(list, value) {
	return list?.find((option) => option.value === value)?.label ?? value;
}

export function filterChips(state, info) {
	const chips = (state.types ?? []).map((type) => ({ key: `type:${type}`, label: labelOf(info?.types, type) }));
	if (state.baseModel) chips.push({ key: 'base', label: labelOf(info?.base_models, state.baseModel) });
	if (state.period) chips.push({ key: 'period', label: labelOf(info?.periods, state.period) });
	return chips;
}

export function removeChip(state, key) {
	if (key.startsWith('type:')) {
		const type = key.slice('type:'.length);
		return { ...state, types: state.types.filter((value) => value !== type) };
	}
	if (key === 'base') return { ...state, baseModel: '' };
	if (key === 'period') return { ...state, period: '' };
	return state;
}

export function clearFilters(state) {
	return { ...state, types: [], baseModel: '', period: '' };
}

export function toggleType(state, value, on) {
	const rest = (state.types ?? []).filter((type) => type !== value);
	return { ...state, types: on ? [...rest, value] : rest };
}

export function matchOptions(options, query) {
	const needle = text(query).trim().toLowerCase();
	if (!needle) return [...(options ?? [])];
	return (options ?? []).filter(
		(option) => option.label.toLowerCase().includes(needle) || option.value.toLowerCase().includes(needle)
	);
}

export function unknownInstalledHashes(items) {
	const hashes = new Set();
	for (const item of items) {
		if (item?.installed === true || item?.installed === false) continue;
		const sha = typeof item?.primary_sha256 === 'string' ? item.primary_sha256.toLowerCase() : '';
		if (SHA256.test(sha)) hashes.add(sha);
	}
	return [...hashes];
}

export function applyInstalled(items, installed) {
	if (!isObject(installed)) return items;
	let changed = false;
	const next = items.map((item) => {
		const sha = typeof item?.primary_sha256 === 'string' ? item.primary_sha256.toLowerCase() : '';
		if (!sha || typeof installed[sha] !== 'boolean' || item.installed === installed[sha]) return item;
		changed = true;
		return { ...item, installed: installed[sha] };
	});
	return changed ? next : items;
}
