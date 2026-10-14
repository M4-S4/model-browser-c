<script>
	import Icon from './Icon.svelte';
	import NsfwMedia from './NsfwMedia.svelte';
	import Pill from './Pill.svelte';
	import { formatCount, modelTypeLabel } from '../lib/format.js';

	let { item, onOpen, reveal = null } = $props();

	let nameButton = $state();

	const typeLabel = $derived(item.mb_type ? modelTypeLabel(item.mb_type) : item.type || 'Model');
	const baseModel = $derived(item.base_model || item.base_models?.[0] || '');
	const downloads = $derived(formatCount(item.stats?.downloads));
	const likes = $derived(formatCount(item.stats?.likes));
	const flags = $derived(
		[
			item.nsfw && { key: 'nsfw', dot: 'bg-danger', label: 'NSFW', title: 'Marked NSFW by the source or by its images' },
			item.early_access && { key: 'ea', dot: 'bg-warning', label: 'Early access', title: 'Early access on the source: downloads need access' },
			item.gated && { key: 'gated', dot: 'bg-warning', label: 'Needs login', title: 'The source serves the files to signed-in accounts only' }
		].filter(Boolean)
	);

	function open(event) {
		event?.stopPropagation?.();
		onOpen?.(item, nameButton);
	}
</script>

<article class="group isolate flex min-w-0 flex-col overflow-hidden rounded-lg border border-line-strong bg-surface-1 transition-colors duration-100 hover:border-line-hover">
	<!-- svelte-ignore a11y_click_events_have_key_events, a11y_no_static_element_interactions -->
	<div class="media-zoom relative aspect-[3/4] cursor-pointer overflow-hidden bg-surface-2" onclick={open}>
		<NsfwMedia
			image={item.thumbnail}
			alt={item.name}
			width={450}
			seed={item.name}
			placeholderIcon="model"
			placeholderLabel={typeLabel}
			compact
			{reveal}
		/>

		{#if flags.length > 0}
			<div class="pointer-events-none absolute left-2 top-2 z-30 flex flex-wrap gap-1">
				{#each flags as flag (flag.key)}
					<span class="inline-flex items-center gap-1 rounded bg-black/60 px-1.5 py-0.5 text-2xs font-medium text-white backdrop-blur-sm" title={flag.title}>
						<span class="h-1.5 w-1.5 rounded-full {flag.dot}" aria-hidden="true"></span>
						{flag.label}
					</span>
				{/each}
			</div>
		{/if}

		{#if item.installed === true}
			<span
				class="pointer-events-none absolute right-2 top-2 z-30 inline-flex items-center gap-1 rounded bg-black/60 px-1.5 py-0.5 text-2xs font-medium text-white backdrop-blur-sm"
				title="Already in your models"
			>
				<Icon name="check" class="h-3 w-3 text-success" strokeWidth={3} />
				Installed
			</span>
		{/if}

		<div class="pointer-events-none absolute bottom-0 left-0 right-0 z-20 bg-gradient-to-t from-black/85 via-black/55 to-transparent px-2.5 pb-2 pt-10">
			<h3 class="min-w-0 text-sm font-semibold text-white">
				<button
					bind:this={nameButton}
					type="button"
					class="pointer-events-auto block max-w-full truncate rounded text-left focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-signal"
					title={item.name}
					onclick={open}
				>
					{item.name}
				</button>
			</h3>
			{#if item.creator}
				<p class="truncate font-mono text-2xs text-white/65">{item.creator}</p>
			{/if}
			<div class="mt-1 flex min-w-0 items-center gap-1 overflow-hidden">
				<span class="flex-shrink-0 rounded bg-black/35 px-1.5 py-0.5 text-2xs text-white/80 backdrop-blur-sm">{typeLabel}</span>
				{#if baseModel}
					<span class="max-w-[7rem] flex-shrink truncate rounded bg-black/35 px-1.5 py-0.5 text-2xs text-white/80 backdrop-blur-sm" title={baseModel}>
						{baseModel}
					</span>
				{/if}
			</div>
		</div>
	</div>

	<div class="flex min-h-8 items-center gap-3 border-t border-line px-2.5 py-1.5 font-mono text-2xs tabular-nums text-fg-subtle">
		{#if downloads}
			<span class="inline-flex items-center gap-1" title="Downloads">
				<Icon name="download" class="h-3 w-3" />
				<span class="sr-only">Downloads:</span>
				{downloads}
			</span>
		{/if}
		{#if likes}
			<span class="inline-flex items-center gap-1" title="Likes">
				<Icon name="heart" class="h-3 w-3" />
				<span class="sr-only">Likes:</span>
				{likes}
			</span>
		{/if}
		{#if !item.mb_type}
			<Pill class="ml-auto font-sans" title="PotionUI has no model type for this kind of file">Not downloadable</Pill>
		{/if}
	</div>
</article>
