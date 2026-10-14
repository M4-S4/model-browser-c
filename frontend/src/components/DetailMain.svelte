<script>
	import HostUi from '../../../../../sdk/HostUi.svelte';
	import FileList from './FileList.svelte';
	import MediaGallery from './MediaGallery.svelte';
	import { fileActionInfo } from '../lib/capabilities.js';
	import { plural } from '../lib/format.js';
	import { isPartial, mediaModelId } from '../lib/versions.js';

	let {
		detail,
		version,
		me,
		installed,
		sourceLabel,
		revealCache,
		byVersion = false,
		versionError = null,
		onRetryVersion,
		onDownload,
		onRequest,
		onPreview
	} = $props();

	const actionFor = $derived((file) =>
		fileActionInfo(me, detail.source, { type: detail.mb_type ?? null, sizeBytes: file.size_bytes ?? null }, sourceLabel)
	);
	const files = $derived(version?.files ?? []);
	const images = $derived(version?.images ?? []);
	const shell = $derived(isPartial(version));
	const shellTitle = $derived(
		versionError?.status === 404 ? 'Version not found' : versionError?.status === 429 ? 'Rate limited' : "Couldn't load this version"
	);
</script>

{#if shell}
	{#if versionError}
		<HostUi name="LoadErrorState" props={{ title: shellTitle, message: versionError.message, onRetry: onRetryVersion }} />
	{:else}
		<div class="flex items-center justify-center gap-3 py-16 text-sm text-fg-muted">
			<HostUi name="Spinner" props={{ size: 'md' }} />
			<span>Loading version…</span>
		</div>
	{/if}
{:else}
	<HostUi name="DetailSection" props={{ label: images.length ? `Previews (${images.length})` : 'Previews', icon: 'image' }}>
		<MediaGallery {images} name={detail.name} source={detail.source} modelId={mediaModelId(detail, version, byVersion)} {me} {revealCache} {onPreview} />
	</HostUi>

	<HostUi name="DetailSection" props={{ label: files.length ? plural(files.length, 'File') : 'Files', icon: 'document', padded: false }}>
		<FileList
			{files}
			{installed}
			gated={detail.gated}
			{actionFor}
			onDownload={(file, opener) => onDownload?.({ detail, version, file, opener })}
			onRequest={(file, opener) => onRequest?.({ detail, version, file, opener })}
		/>
	</HostUi>
{/if}

{#if detail.description_html}
	<HostUi name="DetailSection" props={{ label: 'Description', icon: 'document' }}>
		<div class="description-html break-words text-sm leading-relaxed text-fg">
			{@html detail.description_html}
		</div>
	</HostUi>
{/if}

<style>
	.description-html :global(p),
	.description-html :global(ul),
	.description-html :global(ol),
	.description-html :global(blockquote),
	.description-html :global(pre) {
		margin: 0.5rem 0;
	}
	.description-html :global(ul) {
		list-style: disc;
		padding-left: 1.25rem;
	}
	.description-html :global(ol) {
		list-style: decimal;
		padding-left: 1.25rem;
	}
	.description-html :global(h3),
	.description-html :global(h4) {
		margin: 0.75rem 0 0.25rem;
		font-weight: 600;
	}
	.description-html :global(a) {
		color: rgb(var(--signal));
		text-decoration: underline;
		text-underline-offset: 2px;
	}
	.description-html :global(blockquote) {
		border-left: 2px solid rgb(var(--line-strong));
		padding-left: 0.75rem;
		color: rgb(var(--fg-muted));
	}
	.description-html :global(pre) {
		white-space: pre-wrap;
		border-radius: 4px;
		background: rgb(var(--surface-2));
		padding: 0.5rem;
	}
	.description-html :global(hr) {
		margin: 0.75rem 0;
		border-color: rgb(var(--line));
	}
</style>
