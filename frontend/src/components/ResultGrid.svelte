<script>
	import { onDestroy } from 'svelte';
	import HostUi from '../../../../../sdk/HostUi.svelte';
	import ModelCard from './ModelCard.svelte';
	import { scrollParent } from '../lib/dom.js';

	let {
		items = [],
		status = 'loading',
		error = null,
		hasMore = false,
		loadingMore = false,
		moreError = null,
		sourceLabel = 'The source',
		emptyDescription = 'Try other words, another sort, or fewer filters.',
		onLoadMore,
		onRetry,
		onOpen,
		revealFor = null
	} = $props();

	const SKELETONS = Array.from({ length: 12 }, (_, index) => index);

	let sentinel = $state();
	let observer = null;

	let deadline = $state(0);
	let now = $state(Date.now());
	let timer = null;
	const retryIn = $derived(Math.max(0, Math.ceil((deadline - now) / 1000)));
	const limited = $derived(error?.status === 429 || error?.code === 'rate_limited');

	$effect(() => {
		const seconds = limited ? error?.retryAfter : 0;
		clearInterval(timer);
		timer = null;
		if (!seconds) {
			deadline = 0;
			return;
		}
		const start = Date.now();
		const end = start + seconds * 1000;
		now = start;
		deadline = end;
		timer = setInterval(() => {
			const tick = Date.now();
			now = tick;
			if (tick >= end) {
				clearInterval(timer);
				timer = null;
			}
		}, 1000);
	});

	onDestroy(() => clearInterval(timer));

	const errorTitle = $derived(limited ? 'Rate limited' : error?.status === 404 ? 'Not found' : "Couldn't load the models");
	const errorMessage = $derived(
		limited
			? retryIn > 0
				? `${sourceLabel} asked to slow down: rate limited, retry in ${retryIn} s.`
				: `${sourceLabel} asked to slow down: you can retry now.`
			: (error?.message ?? 'Something went wrong')
	);

	function maybeLoad() {
		if (status === 'ready' && hasMore && !loadingMore && !moreError) onLoadMore?.();
	}

	$effect(() => {
		if (!sentinel || typeof IntersectionObserver === 'undefined') return;
		observer = new IntersectionObserver((entries) => {
			if (entries.some((entry) => entry.isIntersecting)) maybeLoad();
		}, { root: scrollParent(sentinel), rootMargin: '0px 0px 800px 0px' });
		observer.observe(sentinel);
		return () => {
			observer?.disconnect();
			observer = null;
		};
	});

	$effect(() => {
		const count = items.length;
		if (!observer || !sentinel || loadingMore || !hasMore || count === 0) return;
		observer.unobserve(sentinel);
		observer.observe(sentinel);
	});
</script>

{#if status === 'loading'}
	<div class="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5" aria-busy="true" aria-label="Loading models">
		{#each SKELETONS as index (index)}
			<div class="overflow-hidden rounded-lg border border-line-strong bg-surface-1">
				<div class="aspect-[3/4] w-full animate-pulse bg-surface-2"></div>
				<div class="h-8 border-t border-line"></div>
			</div>
		{/each}
	</div>
{:else if status === 'error'}
	<HostUi name="LoadErrorState" props={{ title: errorTitle, message: errorMessage, onRetry, compact: true }} />
{:else if items.length === 0}
	<HostUi
		name="EmptyState"
		props={{
			icon: 'search',
			title: 'No models found',
			description: emptyDescription,
			compact: true
		}}
	/>
{:else}
	<ul class="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5" aria-label="Models">
		{#each items as item (`${item.source}:${item.id}`)}
			<li class="min-w-0">
				<ModelCard {item} {onOpen} reveal={revealFor ? revealFor(item) : null} />
			</li>
		{/each}
	</ul>

	<div bind:this={sentinel} class="h-px w-full" aria-hidden="true"></div>

	<div class="flex flex-col items-center gap-2 py-6" aria-live="polite">
		{#if loadingMore}
			<HostUi name="Spinner" props={{ size: 'md' }} />
		{:else if moreError}
			<p class="max-w-md text-center text-sm text-danger">Couldn't load more: {moreError.message}</p>
			<HostUi name="Button" props={{ variant: 'secondary', size: 'sm', icon: 'refresh', onclick: () => onLoadMore?.(true) }}>Retry</HostUi>
		{:else if hasMore}
			<HostUi name="Button" props={{ variant: 'secondary', size: 'sm', onclick: () => onLoadMore?.() }}>Load more</HostUi>
		{:else}
			<p class="font-mono text-2xs uppercase tracking-[0.07em] text-fg-subtle">End of results</p>
		{/if}
	</div>
{/if}
