<script>
	import Icon from './Icon.svelte';
	import NsfwMedia from './NsfwMedia.svelte';
	import { isRevealable } from '../lib/nsfw.js';
	import { plural } from '../lib/format.js';
	import { toast } from '../lib/dom.js';

	let { images = [], name = '', source, modelId, me, revealCache, onPreview } = $props();

	const LIMIT = 9;
	let showAll = $state(false);
	let revealing = $state(false);
	let revealedById = $state.raw(new Map());
	const NO_URLS = new Map();

	const revealed = $derived(revealedById.get(String(modelId)) ?? NO_URLS);
	const visible = $derived(showAll ? images : images.slice(0, LIMIT));
	const blurred = $derived(images.filter((image) => isRevealable(image, me) && !revealed.has(String(image.id))));

	function revealFor(image) {
		if (!isRevealable(image, me)) return null;
		return () => revealCache.urlFor(source, modelId, image.id);
	}

	async function revealAll() {
		if (revealing) return;
		revealing = true;
		const asked = String(modelId);
		const result = await revealCache.urls(source, modelId);
		revealing = false;
		if (!result.ok) {
			toast('error', `Could not reveal the previews: ${result.error?.message ?? 'unknown error'}`);
			return;
		}
		revealedById = new Map(revealedById).set(asked, result.urls);
	}
</script>

{#if images.length === 0}
	<p class="text-sm text-fg-subtle">No preview for this version.</p>
{:else}
	{#if blurred.length > 0}
		<div class="mb-3 flex flex-wrap items-center justify-between gap-2">
			<span class="text-xs text-fg-muted">{plural(blurred.length, 'NSFW preview')} blurred by your setting</span>
			<button
				type="button"
				class="inline-flex items-center gap-1.5 whitespace-nowrap rounded bg-surface-3 px-2.5 py-1 text-xs font-medium text-fg transition-colors duration-100 hover:bg-line-hover disabled:cursor-not-allowed disabled:opacity-50"
				disabled={revealing}
				onclick={revealAll}
			>
				<Icon name={revealing ? 'refresh' : 'eye'} class="h-3.5 w-3.5 {revealing ? 'animate-spin' : ''}" />
				Reveal all
			</button>
		</div>
	{/if}

	<ul class="grid grid-cols-2 gap-2 sm:grid-cols-3" aria-label="Previews">
		{#each visible as image, index (`${index}:${image.id ?? ''}`)}
			<li class="aspect-[3/4] overflow-hidden rounded border border-line bg-surface-2">
				<NsfwMedia
					{image}
					alt={`${name}, preview ${index + 1}`}
					width={450}
					seed={name}
					reveal={revealFor(image)}
					revealedUrl={image.id !== null && image.id !== undefined ? (revealed.get(String(image.id)) ?? null) : null}
					onOpen={onPreview}
				/>
			</li>
		{/each}
	</ul>

	{#if images.length > LIMIT}
		<button type="button" class="mt-3 text-xs font-medium text-fg-muted hover:text-fg" onclick={() => (showAll = !showAll)}>
			{showAll ? 'Show fewer' : `Show all ${images.length} previews`}
		</button>
	{/if}
{/if}
