<script>
	import Icon from './Icon.svelte';
	import { licenseFlags } from '../lib/downloadForm.js';

	let { license = null } = $props();

	const flags = $derived(licenseFlags(license));
</script>

<section class="rounded border border-line bg-surface-2 px-3 py-2.5" aria-label="Licence">
	<p class="font-mono text-2xs uppercase tracking-[0.07em] text-fg-subtle">Licence</p>
	{#if license}
		<p class="mt-1 text-sm text-fg">{license.summary}</p>
		{#if flags.length > 0}
			<p class="mt-1 text-xs text-fg-muted">{flags.join(' · ')}</p>
		{/if}
		{#if license.url}
			<a class="mt-1 inline-flex items-center gap-1 text-xs text-signal underline underline-offset-2" href={license.url} target="_blank" rel="noopener noreferrer">
				Read the licence
				<Icon name="external-link" class="h-3 w-3" />
			</a>
		{/if}
	{:else}
		<p class="mt-1 text-sm text-fg-subtle">The source states no licence: check the model page before using it.</p>
	{/if}
</section>
