import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
	DEFAULT_LIMIT,
	applyInstalled,
	clearFilters,
	decodeState,
	emptyState,
	enabledSources,
	encodeState,
	filterChips,
	matchOptions,
	mergeItems,
	normalizeState,
	removeChip,
	searchKey,
	searchRequest,
	sourceDefaults,
	sourceSections,
	switchSource,
	toggleType,
	unknownInstalledHashes,
	wantsRequestsTab,
	withoutRequestsTab
} from './search.js';

const SOURCES = enabledSources([
	{
		id: 'civitai',
		label: 'CivitAI',
		types: [
			{ value: 'Checkpoint', label: 'Checkpoint' },
			{ value: 'LORA', label: 'LoRA' },
			{ value: 'TextualInversion', label: 'Embedding' }
		],
		base_models: [
			{ value: 'SD 1.5', label: 'SD 1.5' },
			{ value: 'SDXL 1.0', label: 'SDXL 1.0' },
			{ value: 'Flux.1 D', label: 'Flux.1 D' }
		],
		sorts: [
			{ value: 'highest_rated', label: 'Highest rated' },
			{ value: 'most_downloaded', label: 'Most downloaded' }
		],
		periods: [
			{ value: 'all_time', label: 'All time' },
			{ value: 'month', label: 'Past month' }
		],
		search_help: 'Text, #tag, @creator'
	},
	{
		id: 'huggingface',
		label: 'Hugging Face',
		types: [{ value: 'lora', label: 'LoRA' }],
		base_models: [],
		sorts: [{ value: 'trending', label: 'Trending' }, { value: 'downloads', label: 'Most downloaded' }],
		periods: [],
		search_help: 'org/repo, @author'
	}
]);
const CIVITAI = SOURCES[0];
const HF = SOURCES[1];

const OPENMODELDB = enabledSources([
	{
		id: 'openmodeldb',
		label: 'OpenModelDB',
		types: [
			{ value: 'esrgan', label: 'ESRGAN' },
			{ value: 'swinir', label: 'SwinIR' }
		],
		base_models: [
			{ value: '2', label: '2x' },
			{ value: '4', label: '4x' }
		],
		sorts: [
			{ value: 'newest', label: 'Newest' },
			{ value: 'name', label: 'Name' }
		],
		periods: [],
		search_help: 'Text, #tag',
		types_label: 'Architecture',
		base_models_label: 'Scale',
		base_model_label: 'Architecture'
	}
])[0];

test('enabledSources keeps each source with its option lists', () => {
	assert.deepEqual(
		SOURCES.map((source) => source.id),
		['civitai', 'huggingface']
	);
	assert.equal(CIVITAI.types.length, 3);
	assert.deepEqual(HF.periods, []);
	assert.deepEqual(enabledSources(null), []);
	assert.deepEqual(enabledSources([{ id: 'x' }])[0], {
		id: 'x',
		label: 'x',
		types: [],
		base_models: [],
		sorts: [],
		periods: [],
		search_help: '',
		notice: '',
		types_label: 'Types',
		base_models_label: 'Base model',
		base_model_label: 'Base model'
	});
});

test('enabledSources names the filter groups and the base model line, with defaults', () => {
	const names = (source) => [source.types_label, source.base_models_label, source.base_model_label];
	assert.deepEqual(names(CIVITAI), ['Types', 'Base model', 'Base model']);
	assert.deepEqual(names(HF), ['Types', 'Base model', 'Base model']);
	assert.deepEqual(names(OPENMODELDB), ['Architecture', 'Scale', 'Architecture']);
	const odd = enabledSources([
		{ id: 'a', types_label: '', base_models_label: '   ', base_model_label: '' },
		{ id: 'b', types_label: 7, base_models_label: null, base_model_label: false },
		{ id: 'c', types_label: ['Kind'], base_models_label: {}, base_model_label: ['Kind'] },
		{ id: 'd', types_label: '  Family ', base_models_label: ' Size ', base_model_label: ' Origin ' },
		{ id: 'e', types_label: 'Kind' },
		{ id: 'f', base_models_label: 'Scale' },
		{ id: 'g', base_model_label: 'Architecture' }
	]).map(names);
	assert.deepEqual(odd, [
		['Types', 'Base model', 'Base model'],
		['Types', 'Base model', 'Base model'],
		['Types', 'Base model', 'Base model'],
		['Family', 'Size', 'Origin'],
		['Kind', 'Base model', 'Base model'],
		['Types', 'Scale', 'Base model'],
		['Types', 'Base model', 'Architecture']
	]);
});

test('enabledSources keeps the notice of a source, trimmed, and gives "" when it is missing or not a string', () => {
	const notices = (list) => enabledSources(list).map((source) => source.notice);
	const CIVARCHIVE_NOTICE =
		'Unofficial source: civarchive.com is a community index of CivitAI, not an official API. Deleted models may be unavailable, or come from third-party mirrors.';
	assert.deepEqual(
		notices([
			{ id: 'civarchive', label: 'CivArchive', enabled: true, notice: CIVARCHIVE_NOTICE },
			{ id: 'civitai', label: 'CivitAI', enabled: true, notice: null },
			{ id: 'huggingface', label: 'Hugging Face', enabled: true }
		]),
		[CIVARCHIVE_NOTICE, '', '']
	);
	assert.deepEqual(SOURCES.map((source) => source.notice), ['', '']);
	assert.equal(OPENMODELDB.notice, '');
	assert.deepEqual(
		notices([
			{ id: 'a', notice: '  Careful, unofficial.  ' },
			{ id: 'b', notice: '' },
			{ id: 'c', notice: '   ' },
			{ id: 'd', notice: 7 },
			{ id: 'e', notice: true },
			{ id: 'f', notice: ['Careful'] },
			{ id: 'g', notice: { text: 'Careful' } },
			{ id: 'h', notice: undefined }
		]),
		['Careful, unofficial.', '', '', '', '', '', '', '']
	);
});

test('sourceSections gives each source a rail icon', () => {
	assert.deepEqual(sourceSections(SOURCES), [
		{ id: 'civitai', label: 'CivitAI', icon: 'palette' },
		{ id: 'huggingface', label: 'Hugging Face', icon: 'smile' }
	]);
	assert.equal(sourceSections([{ id: 'new', label: 'New' }])[0].icon, 'globe');
});

test('sourceDefaults are the first options', () => {
	assert.deepEqual(sourceDefaults(CIVITAI), { sort: 'highest_rated', period: 'all_time' });
	assert.deepEqual(sourceDefaults(HF), { sort: 'trending', period: '' });
	assert.deepEqual(sourceDefaults(undefined), { sort: '', period: '' });
});

test('decodeState reads every key, types repeated', () => {
	assert.deepEqual(decodeState('?source=civitai&q=detail&type=LORA&type=Checkpoint&type=LORA&base=SDXL+1.0&sort=most_downloaded&period=month&model=58390'), {
		source: 'civitai',
		q: 'detail',
		types: ['LORA', 'Checkpoint'],
		baseModel: 'SDXL 1.0',
		sort: 'most_downloaded',
		period: 'month',
		model: '58390'
	});
	assert.deepEqual(decodeState(''), emptyState(''));
});

test('wantsRequestsTab spots the link of the request notifications', () => {
	assert.equal(wantsRequestsTab('?tab=requests'), true);
	assert.equal(wantsRequestsTab('source=civitai&tab=requests'), true);
	assert.equal(wantsRequestsTab(new URLSearchParams('tab=other&tab=requests')), true);
	assert.equal(wantsRequestsTab('?tab=other'), false);
	assert.equal(wantsRequestsTab('?requests=tab'), false);
	assert.equal(wantsRequestsTab(''), false);
	assert.equal(wantsRequestsTab(undefined), false);
});

test('withoutRequestsTab drops tab=requests only, keeping the rest in order', () => {
	assert.equal(withoutRequestsTab('?tab=requests'), '');
	assert.equal(withoutRequestsTab('?source=civitai&tab=requests&q=detail'), 'source=civitai&q=detail');
	assert.equal(withoutRequestsTab('tab=other&tab=requests&foo=1'), 'tab=other&foo=1');
	assert.equal(withoutRequestsTab(new URLSearchParams('model=org%2Frepo&tab=requests')), 'model=org%2Frepo');
	assert.equal(withoutRequestsTab(''), '');
});

test('encodeState never writes tab=requests back', () => {
	assert.equal(encodeState(emptyState('civitai'), '?tab=requests&foo=1'), 'foo=1&source=civitai');
	assert.equal(encodeState(emptyState('civitai'), '?tab=other'), 'tab=other&source=civitai');
});

test('encodeState leaves defaults out and keeps foreign keys', () => {
	const state = { source: 'civitai', q: 'detail tweaker', types: ['LORA', 'Checkpoint'], baseModel: 'SDXL 1.0', sort: 'most_downloaded', period: '', model: '' };
	assert.equal(encodeState(state), 'source=civitai&q=detail+tweaker&type=LORA&type=Checkpoint&base=SDXL+1.0&sort=most_downloaded');
	assert.equal(encodeState(emptyState('huggingface'), '?foo=1&q=old'), 'foo=1&source=huggingface');
	assert.equal(encodeState({ ...emptyState('huggingface'), model: 'org/repo' }), 'source=huggingface&model=org%2Frepo');
});

test('a state survives the round trip through the URL', () => {
	const state = { source: 'civitai', q: '#anime @someone', types: ['TextualInversion'], baseModel: 'Flux.1 D', sort: 'most_downloaded', period: 'month', model: '12' };
	assert.deepEqual(normalizeState(decodeState(encodeState(state)), SOURCES), state);
});

test('normalizeState checks the state against the sources', () => {
	const state = normalizeState(
		{ source: 'nowhere', q: 'x', types: ['LORA', 'Bogus', 'LORA'], baseModel: 'SD 9', sort: 'random', period: 'decade', model: '' },
		SOURCES
	);
	assert.deepEqual(state, { source: 'civitai', q: 'x', types: ['LORA'], baseModel: '', sort: '', period: '', model: '' });

	assert.equal(normalizeState({ ...emptyState('civitai'), sort: 'highest_rated', period: 'all_time' }, SOURCES).sort, '');
	const hf = normalizeState({ ...emptyState('huggingface'), baseModel: 'SD 1.5', period: 'month', types: ['lora'] }, SOURCES);
	assert.deepEqual(hf, { ...emptyState('huggingface'), types: ['lora'] });
	assert.deepEqual(normalizeState({ ...emptyState('civitai'), q: 'kept' }, []), { ...emptyState(''), q: 'kept' });
});

test('switchSource keeps the text and drops the filters and the open model', () => {
	const state = { source: 'civitai', q: 'flux', types: ['LORA'], baseModel: 'Flux.1 D', sort: 'most_downloaded', period: 'month', model: '5' };
	assert.deepEqual(switchSource(state, 'huggingface'), { ...emptyState('huggingface'), q: 'flux' });
});

test('searchRequest repeats types and makes sort and period explicit', () => {
	const state = { source: 'civitai', q: '  detail  ', types: ['LORA', 'Checkpoint'], baseModel: 'SDXL 1.0', sort: '', period: 'month', model: '9' };
	assert.equal(
		searchRequest(state, CIVITAI),
		`q=detail&types=LORA&types=Checkpoint&base_models=SDXL+1.0&sort=highest_rated&period=month&limit=${DEFAULT_LIMIT}`
	);
	assert.equal(searchRequest(emptyState('civitai'), CIVITAI, { cursor: 'abc|1' }), `sort=highest_rated&period=all_time&cursor=abc%7C1&limit=${DEFAULT_LIMIT}`);
});

test('searchRequest leaves out what the source does not have', () => {
	assert.equal(searchRequest({ ...emptyState('huggingface'), period: 'month' }, HF), 'sort=trending&limit=24');
	assert.equal(searchRequest(emptyState('x'), undefined), 'limit=24');
});

test('searchKey ignores the open model, the order of types and surrounding spaces', () => {
	const a = { source: 'civitai', q: 'x ', types: ['LORA', 'Checkpoint'], baseModel: '', sort: '', period: '', model: '1' };
	const b = { ...a, q: 'x', types: ['Checkpoint', 'LORA'], model: '' };
	assert.equal(searchKey(a), searchKey(b));
	assert.notEqual(searchKey(a), searchKey({ ...a, sort: 'most_downloaded' }));
	assert.notEqual(searchKey(a), searchKey({ ...a, source: 'huggingface' }));
});

test('mergeItems appends new ids only', () => {
	const first = [
		{ source: 'civitai', id: '1', name: 'A' },
		{ source: 'civitai', id: '2', name: 'B' }
	];
	const merged = mergeItems(first, [
		{ source: 'civitai', id: '2', name: 'B again' },
		{ source: 'civitai', id: '3', name: 'C' },
		{ source: 'civitai', id: '3', name: 'C again' },
		null,
		{ source: 'civitai', name: 'no id' }
	]);
	assert.deepEqual(
		merged.map((item) => item.name),
		['A', 'B', 'C']
	);
	assert.notEqual(merged, first);
	assert.equal(first.length, 2);
	assert.equal(mergeItems(first, undefined).length, 2);
});

test('filterChips lists types, base model and a period other than the default', () => {
	const state = { ...emptyState('civitai'), types: ['LORA', 'TextualInversion'], baseModel: 'SDXL 1.0', period: 'month' };
	assert.deepEqual(filterChips(state, CIVITAI), [
		{ key: 'type:LORA', label: 'LoRA' },
		{ key: 'type:TextualInversion', label: 'Embedding' },
		{ key: 'base', label: 'SDXL 1.0' },
		{ key: 'period', label: 'Past month' }
	]);
	assert.deepEqual(filterChips(emptyState('civitai'), CIVITAI), []);
});

test('a source without periods and with its own group names filters like the others', () => {
	assert.deepEqual(OPENMODELDB.periods, []);
	assert.deepEqual(sourceDefaults(OPENMODELDB), { sort: 'newest', period: '' });
	const stale = { ...emptyState('openmodeldb'), period: 'month' };
	assert.equal(normalizeState(stale, [OPENMODELDB]).period, '');
	assert.equal(searchRequest(stale, OPENMODELDB), `sort=newest&limit=${DEFAULT_LIMIT}`);
	const state = { ...emptyState('openmodeldb'), types: ['esrgan'], baseModel: '4' };
	assert.deepEqual(normalizeState(state, [OPENMODELDB]), state);
	assert.equal(searchRequest(state, OPENMODELDB), `types=esrgan&base_models=4&sort=newest&limit=${DEFAULT_LIMIT}`);
	assert.deepEqual(filterChips(state, OPENMODELDB), [
		{ key: 'type:esrgan', label: 'ESRGAN' },
		{ key: 'base', label: '4x' }
	]);
});

test('removeChip, clearFilters and toggleType', () => {
	const state = { ...emptyState('civitai'), q: 'x', sort: 'most_downloaded', types: ['LORA', 'Checkpoint'], baseModel: 'SDXL 1.0', period: 'month' };
	assert.deepEqual(removeChip(state, 'type:LORA').types, ['Checkpoint']);
	assert.equal(removeChip(state, 'base').baseModel, '');
	assert.equal(removeChip(state, 'period').period, '');
	assert.equal(removeChip(state, 'unknown'), state);
	assert.deepEqual(clearFilters(state), { ...state, types: [], baseModel: '', period: '' });
	assert.deepEqual(toggleType(state, 'TextualInversion', true).types, ['LORA', 'Checkpoint', 'TextualInversion']);
	assert.deepEqual(toggleType(state, 'LORA', false).types, ['Checkpoint']);
	assert.deepEqual(toggleType(state, 'LORA', true).types, ['Checkpoint', 'LORA']);
});

test('matchOptions filters by label or value, any case', () => {
	assert.deepEqual(
		matchOptions(CIVITAI.base_models, 'sd').map((option) => option.value),
		['SD 1.5', 'SDXL 1.0']
	);
	assert.deepEqual(
		matchOptions(CIVITAI.base_models, 'FLUX').map((option) => option.value),
		['Flux.1 D']
	);
	assert.equal(matchOptions(CIVITAI.base_models, '  ').length, 3);
	assert.deepEqual(matchOptions(undefined, 'x'), []);
});

test('unknownInstalledHashes and applyInstalled fill the Installed badges', () => {
	const sha = 'a'.repeat(64);
	const other = 'B'.repeat(64);
	const items = [
		{ id: '1', installed: null, primary_sha256: sha },
		{ id: '2', primary_sha256: other },
		{ id: '3', installed: true, primary_sha256: 'c'.repeat(64) },
		{ id: '4', installed: null, primary_sha256: 'not a hash' },
		{ id: '5', installed: null, primary_sha256: sha }
	];
	assert.deepEqual(unknownInstalledHashes(items), [sha, 'b'.repeat(64)]);
	const next = applyInstalled(items, { [sha]: true, ['b'.repeat(64)]: false });
	assert.deepEqual(
		next.map((item) => item.installed),
		[true, false, true, null, true]
	);
	assert.equal(applyInstalled(items, {}), items);
	assert.equal(applyInstalled(items, null), items);
});
