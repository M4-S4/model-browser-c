<script>
	import { onMount } from 'svelte';
	import Icon from './Icon.svelte';
	import Pill from './Pill.svelte';
	import { listDownloads } from '../lib/client.js';
	import {
		LIST_LIMIT,
		POLL_INTERVAL_MS,
		activeCount,
		detectTransitions,
		noteStarted as noteRowStarted,
		normalizeRows,
		rowKey,
		rowStatus,
		rowSubtitle,
		rowTitle,
		shouldPoll,
		transitionToast
	} from '../lib/downloads.js';
	import { escapeCloses, hostLayer, portal, toast } from '../lib/dom.js';

	let { api } = $props();

	const uid = $props.id();
	let rows = $state.raw([]);
	let loaded = $state(false);
	let error = $state.raw(null);
	let open = $state(false);
	let refreshing = $state(false);
	let button = $state();

	let seen = new Map();
	let timer = null;
	let inFlight = false;
	let again = false;
	let destroyed = false;

	const active = $derived(activeCount(rows));
	const failed = $derived(rows.filter((row) => rowStatus(row).tone === 'danger').length);

	onMount(() => {
		refresh();
		return () => {
			destroyed = true;
			clearTimeout(timer);
		};
	});

	export async function refresh() {
		clearTimeout(timer);
		timer = null;
		if (inFlight) {
			again = true;
			return;
		}
		inFlight = true;
		refreshing = true;
		const result = await listDownloads(api, { all: false, limit: LIST_LIMIT });
		inFlight = false;
		refreshing = false;
		if (destroyed) return;
		loaded = true;
		if (result.ok) {
			const next = normalizeRows(result.data);
			const change = detectTransitions(seen, next);
			seen = change.seen;
			rows = next;
			error = null;
			for (const event of change.events) {
				const message = transitionToast(event);
				toast(message.level, message.message);
			}
		} else {
			error = result.error;
		}
		if (again) {
			again = false;
			timer = setTimeout(refresh, 0);
		} else if (shouldPoll(rows)) {
			timer = setTimeout(refresh, result.ok ? POLL_INTERVAL_MS : POLL_INTERVAL_MS * 2);
		}
	}

	export function noteStarted(pending) {
		seen = noteRowStarted(seen, pending);
		refresh();
	}

	function toggle() {
		open = !open;
		if (open) refresh();
	}

	function close() {
		open = false;
		button?.focus();
	}
</script>

<div class="relative flex-shrink-0">
	<button
		bind:this={button}
		type="button"
		class="input flex h-8 items-center gap-1.5 whitespace-nowrap text-xs font-medium"
		aria-expanded={open}
		aria-controls="{uid}-panel"
		onclick={toggle}
	>
		<Icon name="download" class="h-3.5 w-3.5 text-fg-subtle" />
		<span>Downloads</span>
		{#if active > 0}
			<span class="rounded-full bg-signal/10 px-1.5 font-mono text-2xs tabular-nums text-signal" title="{active} in progress">{active}</span>
		{:else if failed > 0}
			<span class="h-1.5 w-1.5 rounded-full bg-danger" title="A download failed"></span>
		{/if}
	</button>
</div>

{#if open}
	<section
		id="{uid}-panel"
		use:portal
		use:hostLayer
		use:escapeCloses={close}
		class="fixed bottom-4 right-4 flex max-h-[60vh] w-[min(24rem,calc(100vw-2rem))] flex-col overflow-hidden rounded-xl border border-line-strong bg-surface-2 shadow-floating"
		aria-label="Downloads"
	>
		<header class="flex items-center gap-2 border-b border-line px-3 py-2">
			<h2 class="text-sm font-semibold text-fg">Downloads</h2>
			{#if active > 0}
				<span class="font-mono text-2xs tabular-nums text-fg-subtle">{active} in progress</span>
			{/if}
			<button
				type="button"
				class="ml-auto inline-flex min-h-6 min-w-6 items-center justify-center rounded p-1 text-fg-muted transition-colors hover:bg-surface-3/50 hover:text-fg disabled:opacity-50"
				aria-label="Refresh the downloads"
				title="Refresh"
				disabled={refreshing}
				onclick={() => refresh()}
			>
				<Icon name="refresh" class="h-3.5 w-3.5 {refreshing ? 'animate-spin' : ''}" />
			</button>
			<button
				type="button"
				class="inline-flex min-h-6 min-w-6 items-center justify-center rounded p-1 text-fg-muted transition-colors hover:bg-surface-3/50 hover:text-fg"
				aria-label="Close the downloads"
				title="Close"
				onclick={close}
			>
				<Icon name="close" class="h-3.5 w-3.5" />
			</button>
		</header>

		<div class="min-h-0 flex-1 overflow-y-auto" aria-live="polite">
			{#if !loaded}
				<p class="px-3 py-6 text-center text-xs text-fg-subtle">Loading…</p>
			{:else if rows.length === 0 && error}
				<p class="px-3 py-6 text-center text-xs text-danger">Couldn't load the downloads: {error.message}</p>
			{:else if rows.length === 0}
				<p class="px-3 py-6 text-center text-xs text-fg-subtle">No download yet. The files you download from the Model Browser show up here.</p>
			{:else}
				{#if error}
					<p class="px-3 pt-2 text-xs text-danger">Couldn't refresh: {error.message}</p>
				{/if}
				<ul class="divide-y divide-line">
					{#each rows as row (rowKey(row))}
						{@const status = rowStatus(row)}
						<li class="space-y-1.5 px-3 py-2.5">
							<div class="flex items-start gap-2">
								<div class="min-w-0 flex-1">
									<p class="truncate text-sm font-medium text-fg" title={rowTitle(row)}>{rowTitle(row)}</p>
									<p class="truncate font-mono text-2xs text-fg-subtle" title={rowSubtitle(row)}>{rowSubtitle(row)}</p>
								</div>
								<Pill tone={status.tone}>{status.label}</Pill>
							</div>
							{#if status.progress !== null}
								<div
									class="h-1.5 w-full overflow-hidden rounded-full bg-surface-3"
									role="progressbar"
									aria-label="Progress of {rowTitle(row)}"
									aria-valuemin="0"
									aria-valuemax="100"
									aria-valuenow={status.progress}
								>
									<div class="h-full rounded-full bg-signal transition-all" style="width: {status.progress}%"></div>
								</div>
							{/if}
							{#if status.sizeText || status.speedText}
								<p class="font-mono text-2xs tabular-nums text-fg-subtle">
									{[status.progress !== null && status.label === 'Downloading' ? `${status.progress}%` : '', status.sizeText, status.speedText]
										.filter(Boolean)
										.join(' · ')}
								</p>
							{/if}
							{#if status.error}
								<p class="break-words text-xs text-danger">{status.error}</p>
							{/if}
						</li>
					{/each}
				</ul>
			{/if}
		</div>
	</section>
{/if}
