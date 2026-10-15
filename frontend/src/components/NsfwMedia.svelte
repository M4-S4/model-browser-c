<script>
	import { onMount } from 'svelte';
	import Icon from './Icon.svelte';
	import { BLURHASH_SIZE, decodeCached } from '../lib/blurhash.js';
	import { placeholderStyle, thumbUrl } from '../lib/format.js';
	import { mediaState } from '../lib/nsfw.js';
	import { prefersReducedMotion } from '../lib/dom.js';

	let {
		image = null,
		alt = '',
		width = 0,
		reveal = null,
		revealedUrl = null,
		seed = '',
		placeholderIcon = 'image',
		placeholderLabel = '',
		compact = false,
		onOpen = null
	} = $props();

	let localUrl = $state(null);
	let revealing = $state(false);
	let revealError = $state('');
	let resizedFailed = $state(false);
	let broken = $state(false);
	let reduced = $state(false);
	let canvas = $state();
	let video = $state();

	const imageKey = $derived(`${image?.id ?? ''}|${image?.url ?? ''}|${image?.blurhash ?? ''}`);
	let seenKey = '';
	$effect.pre(() => {
		if (imageKey === seenKey) return;
		seenKey = imageKey;
		localUrl = null;
		revealError = '';
		resizedFailed = false;
		broken = false;
	});

	const url = $derived(image?.url || revealedUrl || localUrl || null);
	const state = $derived(broken ? 'empty' : mediaState(image, revealedUrl || localUrl));
	const kind = $derived(image?.kind === 'video' ? 'video' : 'image');
	const src = $derived(url && width > 0 && !resizedFailed ? thumbUrl(url, width, kind) : url);
	const pixels = $derived(state === 'blurred' ? decodeCached(image?.blurhash) : null);

	onMount(() => {
		reduced = prefersReducedMotion();
	});

	$effect(() => {
		if (!canvas || !pixels) return;
		const context = canvas.getContext('2d');
		if (!context) return;
		const data = context.createImageData(BLURHASH_SIZE, BLURHASH_SIZE);
		data.data.set(pixels);
		context.putImageData(data, 0, 0);
	});

	function onImageError() {
		if (src !== url) resizedFailed = true;
		else broken = true;
	}

	function play() {
		if (reduced || !video) return;
		video.play?.()?.catch?.(() => {});
	}

	function pause() {
		video?.pause?.();
	}

	async function doReveal(event) {
		event.stopPropagation();
		if (!reveal || revealing) return;
		revealing = true;
		revealError = '';
		let result;
		try {
			result = await reveal();
		} catch {
			result = { error: { message: 'Could not reveal this media' } };
		}
		revealing = false;
		if (result?.url) localUrl = result.url;
		else revealError = result?.error?.message || 'This media is not available';
	}
</script>

<div class="relative h-full w-full overflow-hidden" style={placeholderStyle(seed || alt)}>
	{#if state === 'media'}
		{#if kind === 'video'}
			<!-- svelte-ignore a11y_media_has_caption -->
			<video
				bind:this={video}
				src={src}
				class="h-full w-full object-cover"
				muted
				loop
				playsinline
				preload="metadata"
				aria-label={alt}
				onmouseenter={play}
				onmouseleave={pause}
				onerror={() => (broken = true)}
			></video>
			<span class="pointer-events-none absolute bottom-2 right-2 rounded bg-black/60 p-1 text-white backdrop-blur-sm" aria-hidden="true">
				<Icon name="play" class="h-3.5 w-3.5" />
			</span>
		{:else}
			<img {src} {alt} class="h-full w-full object-cover" loading="lazy" decoding="async" onerror={onImageError} />
		{/if}
		{#if onOpen}
			<button
				type="button"
				class="absolute inset-0 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-signal"
				aria-label={`View ${alt || 'media'}`}
				onclick={(event) => {
					event.stopPropagation();
					onOpen({ url, kind, alt });
				}}
				onmouseenter={play}
				onmouseleave={pause}
			></button>
		{/if}
	{:else if state === 'blurred'}
		{#if pixels}
			<canvas bind:this={canvas} width={BLURHASH_SIZE} height={BLURHASH_SIZE} class="absolute inset-0 h-full w-full" aria-hidden="true"></canvas>
		{/if}
		{#if reveal}
			<button
				type="button"
				class="absolute inset-0 flex flex-col items-center justify-center gap-1 bg-black/35 px-2 text-center text-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-signal"
				aria-label={`NSFW: reveal ${alt || 'this media'}`}
				aria-busy={revealing}
				onclick={doReveal}
			>
				<Icon name={revealing ? 'refresh' : 'eye-off'} class="{compact ? 'h-4 w-4' : 'h-5 w-5'} {revealing ? 'animate-spin' : ''}" />
				<span class="{compact ? 'text-2xs' : 'text-xs'} font-semibold">NSFW · click to reveal</span>
				{#if revealError}
					<span class="text-2xs text-white/80">{revealError}</span>
				{/if}
			</button>
		{:else}
			<div class="absolute inset-0 flex flex-col items-center justify-center gap-1 bg-black/35 px-2 text-center text-white">
				<Icon name="eye-off" class={compact ? 'h-4 w-4' : 'h-5 w-5'} />
				<span class="{compact ? 'text-2xs' : 'text-xs'} font-semibold">NSFW</span>
			</div>
		{/if}
	{:else}
		<div class="absolute inset-0 flex flex-col items-center justify-center gap-2 px-3 text-center">
			<Icon name={broken ? 'warning' : placeholderIcon} class="h-8 w-8 text-fg-subtle" strokeWidth={1.5} />
			{#if broken}
				<span class="text-2xs text-fg-subtle">Preview unavailable</span>
			{:else if placeholderLabel}
				<span class="text-xs font-medium text-fg-subtle">{placeholderLabel}</span>
			{/if}
		</div>
	{/if}
</div>
