<script>
	import Icon from './Icon.svelte';
	import Pill from './Pill.svelte';
	import { formatBytes } from '../lib/format.js';

	let {
		files = [],
		installed = new Map(),
		gated = false,
		actionFor = () => ({ action: 'none', reason: '' }),
		onDownload,
		onRequest
	} = $props();

	function meta(file) {
		return [formatBytes(file.size_bytes), file.format, file.precision, file.file_type].filter(Boolean);
	}

	function isInstalled(file) {
		return typeof file.sha256 === 'string' && installed.get(file.sha256.toLowerCase()) === true;
	}
</script>

{#if files.length === 0}
	<p class="px-4 py-4 text-sm text-fg-subtle sm:px-5">This version has no file to download.</p>
{:else}
	<ul class="divide-y divide-line" aria-label="Files">
		{#each files as file (file.id)}
			{@const info = actionFor(file)}
			<li class="flex flex-wrap items-center gap-x-3 gap-y-2 px-4 py-3 sm:px-5">
				<div class="min-w-0 flex-1">
					<p class="flex min-w-0 items-center gap-1.5">
						{#if file.primary}
							<span class="flex-shrink-0 text-warning" title="Primary file">
								<Icon name="star" class="h-3.5 w-3.5" />
								<span class="sr-only">Primary file:</span>
							</span>
						{/if}
						<span class="truncate font-mono text-xs text-fg" title={file.name}>{file.name}</span>
					</p>
					<div class="mt-1 flex flex-wrap items-center gap-1.5">
						{#if meta(file).length > 0}
							<span class="font-mono text-2xs tabular-nums text-fg-subtle">{meta(file).join(' · ')}</span>
						{/if}
						{#if isInstalled(file)}
							<Pill tone="success" title="This file is already in your models">
								<Icon name="check" class="h-3 w-3" strokeWidth={3} />
								Installed
							</Pill>
						{/if}
						{#if gated && file.auth_provider_id}
							<Pill tone="warning" title="The source serves this file to signed-in accounts only">Needs login</Pill>
						{/if}
						{#if !file.sha256}
							<Pill title="The source gives no SHA256 for this file: it cannot be checked or recognised as already installed">No checksum</Pill>
						{/if}
					</div>
				</div>

				{#if info.action === 'download'}
					<button
						type="button"
						class="inline-flex flex-shrink-0 items-center justify-center gap-1.5 whitespace-nowrap rounded bg-accent px-2.5 py-1 text-xs font-medium text-accent-contrast transition-colors duration-100 hover:bg-accent-hover active:bg-accent-active"
						aria-label={`Download ${file.name}`}
						onclick={(event) => onDownload?.(file, event.currentTarget)}
					>
						<Icon name="download" class="h-3.5 w-3.5" />
						Download
					</button>
				{:else if info.action === 'source-blocked'}
					<div class="flex flex-shrink-0 flex-col items-end gap-0.5">
						<button
							type="button"
							class="inline-flex cursor-not-allowed items-center gap-1.5 whitespace-nowrap rounded bg-surface-3 px-2.5 py-1 text-xs font-medium text-fg opacity-50"
							disabled
						>
							<Icon name="download" class="h-3.5 w-3.5" />
							Download
						</button>
						<span class="text-2xs text-fg-subtle">{info.reason}</span>
					</div>
				{:else if info.action === 'request' && !isInstalled(file)}
					<button
						type="button"
						class="inline-flex flex-shrink-0 items-center justify-center gap-1.5 whitespace-nowrap rounded bg-accent px-2.5 py-1 text-xs font-medium text-accent-contrast transition-colors duration-100 hover:bg-accent-hover active:bg-accent-active"
						aria-label={`Request ${file.name}`}
						title="Ask an admin to download this file for you"
						onclick={(event) => onRequest?.(file, event.currentTarget)}
					>
						<Icon name="hand" class="h-3.5 w-3.5" strokeWidth={1.5} />
						Request
					</button>
				{:else if info.action === 'request-blocked'}
					<div class="flex flex-shrink-0 flex-col items-end gap-0.5">
						<button
							type="button"
							class="inline-flex cursor-not-allowed items-center gap-1.5 whitespace-nowrap rounded bg-surface-3 px-2.5 py-1 text-xs font-medium text-fg opacity-50"
							disabled
						>
							<Icon name="hand" class="h-3.5 w-3.5" strokeWidth={1.5} />
							Request
						</button>
						<span class="text-2xs text-fg-subtle">{info.reason}</span>
					</div>
				{/if}
			</li>
		{/each}
	</ul>
{/if}
