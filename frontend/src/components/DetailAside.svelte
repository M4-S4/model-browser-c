<script>
	import HostUi from '../../../../../sdk/HostUi.svelte';
	import Pill from './Pill.svelte';
	import Icon from './Icon.svelte';
	import { formatCount, formatDate, modelTypeLabel } from '../lib/format.js';
	import { licenseFlags } from '../lib/downloadForm.js';
	import { copyText, toast } from '../lib/dom.js';
	import { DEFAULT_BASE_MODEL_LABEL } from '../lib/search.js';
	import { isPartial } from '../lib/versions.js';

	let { detail, version, sourceUrl = '', sourceHost = '', baseModelLabel = '' } = $props();

	const triggers = $derived((version?.trigger_words ?? []).filter((word) => typeof word === 'string' && word.trim()));
	const flags = $derived(licenseFlags(detail.license));
	const tags = $derived((detail.tags ?? []).slice(0, 40));
	const stats = $derived(
		[
			['Downloads', formatCount(detail.stats?.downloads)],
			['Likes', formatCount(detail.stats?.likes)],
			['Rating', typeof detail.stats?.rating === 'number' ? detail.stats.rating.toFixed(1) : '']
		].filter(([, value]) => value)
	);

	async function copyWord(word) {
		if (await copyText(word)) toast('success', `Copied "${word}"`);
		else toast('error', 'Could not copy');
	}
</script>

{#if version}
	<HostUi name="DetailSection" props={{ label: 'Version', icon: 'layers' }}>
		<HostUi name="KVGrid">
			<HostUi name="KVItem" props={{ label: 'Name', full: true }}>
				<span class="break-words">{version.name}</span>
			</HostUi>
			{#if !isPartial(version)}
				<HostUi name="KVItem" props={{ label: baseModelLabel || DEFAULT_BASE_MODEL_LABEL }}>{version.base_model || '—'}</HostUi>
			{/if}
			<HostUi name="KVItem" props={{ label: 'Published' }}>{formatDate(version.created_at) || '—'}</HostUi>
			{#if version.early_access || version.nsfw}
				<HostUi name="KVItem" props={{ label: 'Flags', full: true }}>
					<span class="flex flex-wrap gap-1">
						{#if version.early_access}<Pill tone="warning">Early access</Pill>{/if}
						{#if version.nsfw}<Pill tone="danger">NSFW</Pill>{/if}
					</span>
				</HostUi>
			{/if}
			{#if !isPartial(version)}
				<HostUi name="KVItem" props={{ label: 'Trigger words', full: true }}>
					{#if triggers.length === 0}
						<span class="text-fg-subtle">None</span>
					{:else}
						<span class="flex flex-wrap items-center gap-1.5">
							{#each triggers as word, index (`${index}:${word}`)}
								<button
									type="button"
									class="inline-flex max-w-full items-center gap-1 rounded border border-line-strong bg-surface-2 px-1.5 py-0.5 text-left font-mono text-xs text-fg transition-colors hover:bg-surface-3"
									title="Copy this trigger word"
									onclick={() => copyWord(word)}
								>
									<span class="break-words">{word}</span>
									<Icon name="copy" class="h-3 w-3 flex-shrink-0 text-fg-subtle" />
								</button>
							{/each}
							{#if triggers.length > 1}
								<HostUi name="CopyButton" props={{ text: triggers.join(', '), label: 'Copy all', size: 'xs', variant: 'secondary' }} />
							{/if}
						</span>
					{/if}
				</HostUi>
			{/if}
		</HostUi>
	</HostUi>
{/if}

<HostUi name="DetailSection" props={{ label: 'Model', icon: 'model' }}>
	<HostUi name="KVGrid">
		<HostUi name="KVItem" props={{ label: 'Source type' }}>{detail.type || '—'}</HostUi>
		<HostUi name="KVItem" props={{ label: 'Files as' }}>
			{#if detail.mb_type}
				{modelTypeLabel(detail.mb_type)}
			{:else}
				<span class="text-fg-subtle">No PotionUI type</span>
			{/if}
		</HostUi>
		{#if detail.creator}
			<HostUi name="KVItem" props={{ label: 'Creator', full: true }}><span class="break-words">{detail.creator}</span></HostUi>
		{/if}
		{#each stats as [label, value] (label)}
			<HostUi name="KVItem" props={{ label, mono: true }}>{value}</HostUi>
		{/each}
		{#if sourceUrl}
			<HostUi name="KVItem" props={{ label: 'Source page', full: true }}>
				<a class="inline-flex items-center gap-1 break-all text-signal underline underline-offset-2" href={sourceUrl} target="_blank" rel="noopener noreferrer">
					{sourceHost || sourceUrl}
					<Icon name="external-link" class="h-3 w-3 flex-shrink-0" />
				</a>
			</HostUi>
		{/if}
	</HostUi>
</HostUi>

<HostUi name="DetailSection" props={{ label: 'Licence', icon: 'shield' }}>
	{#if detail.license}
		<p class="text-sm text-fg">{detail.license.summary}</p>
		{#if flags.length > 0}
			<ul class="mt-2 space-y-1 text-xs text-fg-muted">
				{#each flags as flag (flag)}
					<li>{flag}</li>
				{/each}
			</ul>
		{/if}
		{#if detail.license.url}
			<a class="mt-2 inline-flex items-center gap-1 text-xs text-signal underline underline-offset-2" href={detail.license.url} target="_blank" rel="noopener noreferrer">
				Read the licence
				<Icon name="external-link" class="h-3 w-3" />
			</a>
		{/if}
	{:else}
		<p class="text-sm text-fg-subtle">The source states no licence.</p>
	{/if}
</HostUi>

{#if tags.length > 0}
	<HostUi name="DetailSection" props={{ label: 'Tags', icon: 'tag' }}>
		<ul class="flex flex-wrap gap-1.5">
			{#each tags as tag, index (`${index}:${tag}`)}
				<li><Pill>{tag}</Pill></li>
			{/each}
		</ul>
	</HostUi>
{/if}
