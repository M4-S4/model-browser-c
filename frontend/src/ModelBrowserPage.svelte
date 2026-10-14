<script>
	import { onMount } from 'svelte';
	import HostUi from '../../../../sdk/HostUi.svelte';
	import DetailPanel from './components/DetailPanel.svelte';
	import DownloadDialog from './components/DownloadDialog.svelte';
	import DownloadTray from './components/DownloadTray.svelte';
	import FilterChips from './components/FilterChips.svelte';
	import FilterPopover from './components/FilterPopover.svelte';
	import Icon from './components/Icon.svelte';
	import NsfwModeControl from './components/NsfwModeControl.svelte';
	import RequestDialog from './components/RequestDialog.svelte';
	import RequestsButton from './components/RequestsButton.svelte';
	import ResultGrid from './components/ResultGrid.svelte';
	import { getInstalled, getMe, getNsfwMedia, getSources, searchModels, setNsfwMode } from './lib/client.js';
	import { can, normalizeMe } from './lib/capabilities.js';
	import { MODE_LABELS, createRevealCache, effectiveMode, isRevealable, nsfwInfo } from './lib/nsfw.js';
	import { showsRequests } from './lib/requests.js';
	import {
		applyInstalled,
		clearFilters,
		decodeState,
		emptyState,
		enabledSources,
		encodeState,
		filterChips,
		mergeItems,
		normalizeState,
		removeChip,
		searchKey,
		searchRequest,
		sourceDefaults,
		sourceSections,
		switchSource,
		unknownInstalledHashes,
		wantsRequestsTab,
		withoutRequestsTab
	} from './lib/search.js';
	import { scrollParent, toast } from './lib/dom.js';

	let { api, setHeaderHidden } = $props();

	const SEARCH_DELAY_MS = 400;
	const MAX_EMPTY_PAGES = 3;

	let phase = $state('loading');
	let bootError = $state.raw(null);
	let me = $state.raw(null);
	let sources = $state.raw([]);
	let query = $state.raw(emptyState());
	let results = $state.raw({ key: '', request: null, items: [], cursor: null, status: 'loading', error: null });
	let loadingMore = $state(false);
	let moreError = $state.raw(null);
	let detailItem = $state.raw(null);
	let downloadTarget = $state.raw(null);
	let requestTarget = $state.raw(null);
	let modeBusy = $state(false);
	let reloadToken = $state(0);
	let installedToken = $state(0);
	let tray = $state();
	let requestsButton = $state();
	let requestsAsked = $state(false);
	let content = $state();

	let searchSeq = 0;
	let searchTimer = null;
	let emptyPages = 0;
	let detailOpener = null;
	let alive = true;
	let pagePath = '';

	const revealCache = createRevealCache((source, id) => getNsfwMedia(api, source, id));

	const info = $derived(sources.find((source) => source.id === query.source));
	const sections = $derived(sourceSections(sources));
	const chips = $derived(filterChips(query, info));
	const showTray = $derived(can(me, 'download') || can(me, 'request'));
	const showRequests = $derived(showsRequests(me));
	const emptyDescription = $derived(
		nsfwInfo(me).locked
			? 'Try other words, another sort, or fewer filters. NSFW models are never shown to you here.'
			: effectiveMode(me) === 'hide'
				? 'Try other words, another sort, or fewer filters. NSFW models are hidden: your sensitive content setting is Hide.'
				: 'Try other words, another sort, or fewer filters.'
	);
	const pageProps = $derived({
		title: 'Model Browser',
		titleLabel: 'Model Browser',
		persistKey: 'model-browser:page',
		sections,
		section: query.source,
		onSelectSection: selectSource,
		count: results.status === 'ready' ? results.items.length : null,
		q: query.q,
		onQueryChange,
		searchPlaceholder: info ? `Search ${info.label}` : 'Search models',
		sortBy: query.sort || sourceDefaults(info).sort,
		sortOptions: info?.sorts ?? [],
		onSortChange
	});

	onMount(() => {
		setHeaderHidden?.(true);
		pagePath = location.pathname;
		takeRequestsLink();
		boot();
		const navigation = window.navigation;
		const onEntryChange = () => setTimeout(takeRequestsLink, 0);
		navigation?.addEventListener?.('currententrychange', onEntryChange);
		return () => {
			alive = false;
			searchSeq += 1;
			clearTimeout(searchTimer);
			navigation?.removeEventListener?.('currententrychange', onEntryChange);
		};
	});

	$effect(() => {
		if (!requestsAsked || phase !== 'ready') return;
		if (!showRequests) {
			requestsAsked = false;
			return;
		}
		if (!requestsButton) return;
		requestsAsked = false;
		requestsButton.openPanel?.();
	});

	async function boot() {
		phase = 'loading';
		bootError = null;
		const meResult = await getMe(api);
		if (!alive) return;
		if (!meResult.ok) {
			bootError = meResult.error;
			phase = 'error';
			return;
		}
		me = normalizeMe(meResult.data);
		if (!can(me, 'browse')) {
			phase = 'denied';
			return;
		}
		const sourcesResult = await getSources(api);
		if (!alive) return;
		if (!sourcesResult.ok) {
			bootError = sourcesResult.error;
			phase = 'error';
			return;
		}
		sources = enabledSources(sourcesResult.data);
		if (sources.length === 0) {
			phase = 'empty';
			return;
		}
		query = normalizeState(decodeState(location.search), sources);
		phase = 'ready';
		if (query.model) detailItem = { source: query.source, id: query.model };
		runSearch(true);
	}

	function takeRequestsLink() {
		if (!alive || location.pathname !== pagePath || !wantsRequestsTab(location.search)) return;
		const search = withoutRequestsTab(location.search);
		history.replaceState(history.state, '', `${location.pathname}${search ? `?${search}` : ''}${location.hash}`);
		requestsAsked = true;
	}

	function writeUrl() {
		const search = encodeState(query, location.search);
		const url = `${location.pathname}${search ? `?${search}` : ''}${location.hash}`;
		if (url !== `${location.pathname}${location.search}${location.hash}`) {
			history.replaceState(history.state, '', url);
		}
	}

	async function runSearch(force = false) {
		clearTimeout(searchTimer);
		searchTimer = null;
		if (phase !== 'ready' || !info) return;
		writeUrl();
		const key = searchKey(query);
		if (!force && key === results.key && results.status !== 'error') return;
		const mine = ++searchSeq;
		const request = { source: query.source, state: query, info };
		results = { key, request, items: [], cursor: null, status: 'loading', error: null };
		loadingMore = false;
		moreError = null;
		emptyPages = 0;
		scrollParent(content)?.scrollTo?.({ top: 0 });
		const result = await searchModels(api, request.source, searchRequest(request.state, request.info));
		if (mine !== searchSeq || !alive) return;
		if (!result.ok) {
			results = { ...results, status: 'error', error: result.error };
			return;
		}
		const items = mergeItems([], result.data?.items);
		results = { ...results, items, cursor: result.data?.next_cursor || null, status: 'ready' };
		fillInstalled(mine, items);
	}

	async function loadMore(force = false) {
		if (results.status !== 'ready' || !results.cursor || loadingMore) return;
		if (moreError && !force) return;
		const mine = searchSeq;
		const { request, cursor } = results;
		loadingMore = true;
		moreError = null;
		const result = await searchModels(api, request.source, searchRequest(request.state, request.info, { cursor }));
		if (mine !== searchSeq || !alive) return;
		loadingMore = false;
		if (!result.ok) {
			moreError = result.error;
			return;
		}
		const before = results.items.length;
		const items = mergeItems(results.items, result.data?.items);
		emptyPages = items.length === before ? emptyPages + 1 : 0;
		let next = result.data?.next_cursor || null;
		if (next === cursor || emptyPages >= MAX_EMPTY_PAGES) next = null;
		results = { ...results, items, cursor: next };
		fillInstalled(mine, items.slice(before));
	}

	async function fillInstalled(mine, items) {
		const hashes = unknownInstalledHashes(items);
		if (hashes.length === 0) return;
		const result = await getInstalled(api, hashes);
		if (mine !== searchSeq || !alive || !result.ok) return;
		results = { ...results, items: applyInstalled(results.items, result.data?.installed) };
	}

	function onQueryChange(value) {
		query = { ...query, q: value };
		clearTimeout(searchTimer);
		searchTimer = setTimeout(() => runSearch(), SEARCH_DELAY_MS);
	}

	function onPageKeydown(event) {
		const target = event.target;
		if (event.key === 'Enter' && target instanceof HTMLInputElement && target.type === 'search') {
			event.preventDefault();
			runSearch();
		}
	}

	function selectSource(id) {
		if (id === query.source || !sources.some((source) => source.id === id)) return;
		closeDetail(false);
		query = normalizeState(switchSource(query, id), sources);
		runSearch(true);
	}

	function onSortChange(value) {
		query = normalizeState({ ...query, sort: value }, sources);
		runSearch();
	}

	function applyFilters(next) {
		query = normalizeState({ ...next, q: query.q, model: query.model }, sources);
		runSearch();
	}

	function revealFor(item) {
		if (!isRevealable(item?.thumbnail, me)) return null;
		return () => revealCache.urlFor(item.source, item.id, item.thumbnail.id);
	}

	async function changeMode(mode) {
		if (modeBusy || !me || mode === me.nsfw.mode) return;
		modeBusy = true;
		const saved = await setNsfwMode(api, mode, { key: nsfwInfo(me).settingKey });
		if (!alive) return;
		if (!saved.ok) {
			modeBusy = false;
			toast('error', `Could not change the sensitive content setting: ${saved.error.message}`);
			return;
		}
		const meResult = await getMe(api);
		if (!alive) return;
		modeBusy = false;
		me = meResult.ok ? normalizeMe(meResult.data) : { ...me, nsfw: { ...me.nsfw, mode } };
		revealCache.reset();
		toast('success', `Sensitive content: ${MODE_LABELS[me.nsfw.mode] ?? me.nsfw.mode} (app-wide setting)`);
		reloadToken += 1;
		runSearch(true);
	}

	function openDetail(item, opener) {
		detailOpener = opener ?? (document.activeElement instanceof HTMLElement ? document.activeElement : null);
		detailItem = item;
		query = { ...query, model: String(item.id) };
		writeUrl();
	}

	function closeDetail(restoreFocus = true) {
		if (!detailItem) return;
		detailItem = null;
		downloadTarget = null;
		requestTarget = null;
		query = { ...query, model: '' };
		writeUrl();
		const opener = detailOpener;
		detailOpener = null;
		if (restoreFocus && opener?.isConnected) requestAnimationFrame(() => opener.focus({ preventScroll: true }));
	}

	function onDownloadStarted(data) {
		tray?.noteStarted?.(data?.pending);
		if (data?.already_installed) installedToken += 1;
	}

	function onRequestCreated() {
		requestsButton?.refreshCount?.();
	}

	function sourceLabel(id) {
		return sources.find((source) => source.id === id)?.label ?? id;
	}

	function baseModelLabel(id) {
		return sources.find((source) => source.id === id)?.base_model_label ?? '';
	}
</script>

{#if phase === 'loading'}
	<div class="flex h-full items-center justify-center p-6">
		<HostUi name="Spinner" props={{ size: 'lg' }} />
	</div>
{:else if phase === 'error'}
	<HostUi
		name="LoadErrorState"
		props={{ title: "Couldn't open the Model Browser", message: bootError?.message ?? 'Something went wrong', onRetry: boot, compact: false }}
	/>
{:else if phase === 'denied'}
	<div class="flex h-full items-center justify-center p-6">
		<div class="w-full max-w-lg">
			<HostUi
				name="EmptyState"
				props={{
					icon: 'shield',
					title: 'You do not have access to the Model Browser',
					description: 'Ask an administrator to give one of your user groups the Browse permission.'
				}}
			/>
		</div>
	</div>
{:else if phase === 'empty'}
	<div class="flex h-full items-center justify-center p-6">
		<div class="w-full max-w-lg">
			<HostUi
				name="EmptyState"
				props={{
					icon: 'globe',
					title: 'No source is enabled',
					description: 'There is no model source to browse at the moment. Ask an administrator.'
				}}
			/>
		</div>
	</div>
{:else}
	<div class="contents" role="presentation" onkeydown={onPageKeydown}>
		<HostUi name="LibraryPage" props={pageProps}>
			<div class="flex min-h-full flex-col" bind:this={content}>
				<div class="sticky top-0 z-20 flex flex-wrap items-center gap-x-3 gap-y-2 border-b border-line bg-canvas px-4 py-2 sm:px-6">
					<p class="min-w-[12rem] flex-1 truncate text-xs text-fg-subtle" title={info?.search_help || undefined}>
						{#if info?.search_help}
							<span class="sr-only">Search syntax:</span>
							{info.search_help}
						{/if}
					</p>
					<div class="ml-auto flex flex-wrap items-center gap-2">
						<FilterPopover {info} filters={query} count={chips.length} onChange={applyFilters} />
						<NsfwModeControl {me} busy={modeBusy} onChange={changeMode} />
						{#if showTray}
							<DownloadTray bind:this={tray} {api} />
						{/if}
						{#if showRequests}
							<RequestsButton bind:this={requestsButton} {api} {me} {sourceLabel} />
						{/if}
					</div>
				</div>

				{#if info?.notice}
					<div class="px-4 pt-3 sm:px-6">
						<p class="flex items-start gap-2 rounded-lg border border-warning/25 bg-warning/10 px-3 py-2 text-xs text-warning" role="note">
							<Icon name="warning" class="mt-0.5 h-3.5 w-3.5 flex-shrink-0" />
							<span class="min-w-0">{info.notice}</span>
						</p>
					</div>
				{/if}

				{#if chips.length > 0}
					<div class="px-4 pt-3 sm:px-6">
						<FilterChips
							{chips}
							shown={results.items.length}
							hasMore={!!results.cursor}
							onRemove={(key) => applyFilters(removeChip(query, key))}
							onClear={() => applyFilters(clearFilters(query))}
						/>
					</div>
				{/if}

				<div class="flex-1 px-4 py-4 sm:px-6">
					<ResultGrid
						items={results.items}
						status={results.status}
						error={results.error}
						hasMore={!!results.cursor}
						{loadingMore}
						{moreError}
						sourceLabel={info?.label ?? 'The source'}
						{emptyDescription}
						onLoadMore={loadMore}
						onRetry={() => runSearch(true)}
						onOpen={openDetail}
						{revealFor}
					/>
				</div>
			</div>
		</HostUi>
	</div>
{/if}

{#if detailItem}
	<DetailPanel
		{api}
		{me}
		source={detailItem.source}
		sourceLabel={sourceLabel(detailItem.source)}
		baseModelLabel={baseModelLabel(detailItem.source)}
		modelId={detailItem.id}
		item={detailItem}
		{revealCache}
		{reloadToken}
		{installedToken}
		onClose={() => closeDetail()}
		onDownload={(target) => (downloadTarget = target)}
		onRequest={(target) => (requestTarget = target)}
	/>
{/if}

{#if downloadTarget}
	{#key downloadTarget}
		<DownloadDialog {api} {me} target={downloadTarget} onClose={() => (downloadTarget = null)} onStarted={onDownloadStarted} />
	{/key}
{/if}

{#if requestTarget}
	{#key requestTarget}
		<RequestDialog {api} {me} target={requestTarget} onClose={() => (requestTarget = null)} onCreated={onRequestCreated} />
	{/key}
{/if}
