<script>
	import { onMount } from 'svelte';
	import HostUi from '../../../../../sdk/HostUi.svelte';
	import Icon from './Icon.svelte';
	import LicenceBox from './LicenceBox.svelte';
	import {
		MAX_NOTE_LENGTH,
		buildRequestBody,
		createRequest,
		createToast,
		initialRequestForm,
		requestFormProblems,
		requestWarnings
	} from '../lib/requests.js';
	import { allowedTypes } from '../lib/capabilities.js';
	import { baseName, canSubmit } from '../lib/downloadForm.js';
	import { formatBytes, modelTypeLabel } from '../lib/format.js';
	import { escapeCloses, hostLayer, portal, toast, trapFocus } from '../lib/dom.js';

	let { api, me, target, onClose, onCreated } = $props();

	const uid = $props.id();
	let mounted = $state(false);
	const startingForm = () => initialRequestForm(me, target.detail);
	let form = $state(startingForm());
	let busy = $state(false);
	let serverError = $state.raw(null);

	const detail = $derived(target.detail);
	const version = $derived(target.version);
	const file = $derived(target.file);
	const choice = $derived(allowedTypes(me, detail.mb_type, { purpose: 'request' }));
	const problems = $derived(requestFormProblems(form, { me, detail, file }));
	const ready = $derived(canSubmit(problems) && !busy);
	const warnings = $derived(requestWarnings({ detail, version, file }));
	const sizeText = $derived(formatBytes(file.size_bytes));
	const fileMeta = $derived([file.format, file.precision].filter(Boolean).join(' · '));
	const typeError = $derived(form.modelType && problems.modelType ? problems.modelType : null);
	const typeHelp = $derived(
		detail.mb_type
			? choice.suggestionAllowed
				? `Suggested by the source: ${modelTypeLabel(detail.mb_type)}. The admin may change it.`
				: `The source suggests ${modelTypeLabel(detail.mb_type)}, which your permissions do not let you ask for: choose another type`
			: `The source's type (${detail.type || 'unknown'}) has no PotionUI equivalent: choose the closest one, the admin may change it`
	);
	const noteHelp = $derived(`${form.note.length}/${MAX_NOTE_LENGTH} · Why you need it, or anything the admin should know`);

	onMount(() => {
		mounted = true;
	});

	async function submit(event) {
		event?.preventDefault?.();
		if (!ready) return;
		busy = true;
		serverError = null;
		const result = await createRequest(api, buildRequestBody({ detail, version, file, form }));
		busy = false;
		const message = createToast(result, file);
		toast(message.level, message.message);
		if (result.ok) onCreated?.(result.data);
		if (message.close) {
			onClose?.();
			return;
		}
		serverError = result.error;
	}

	function onBackdrop(event) {
		if (event.target === event.currentTarget && !busy) onClose?.();
	}
</script>

{#if mounted}
<div
	use:portal
	use:hostLayer
	use:escapeCloses={() => {
		if (!busy) onClose?.();
	}}
	class="fixed inset-0 flex items-center justify-center bg-black/60 p-4 backdrop-blur-sm"
	role="presentation"
	onclick={onBackdrop}
>
	<div
		class="flex max-h-[90vh] w-full max-w-lg flex-col rounded-xl bg-surface-1 shadow-overlay outline-none"
		role="dialog"
		aria-modal="true"
		aria-labelledby="{uid}-title"
		aria-describedby="{uid}-intro"
		tabindex="-1"
		use:trapFocus={target.opener}
		onkeydown={(event) => event.stopPropagation()}
	>
	<form class="flex min-h-0 flex-1 flex-col" novalidate onsubmit={submit}>
		<header class="flex items-start gap-3 border-b border-line px-5 py-4">
			<div class="min-w-0 flex-1">
				<h2 id="{uid}-title" class="truncate text-lg font-semibold text-fg" title={file.name}>Request {baseName(file.name)}</h2>
				<p class="mt-0.5 truncate text-xs text-fg-muted">
					{[detail.name, version.name, sizeText].filter(Boolean).join(' · ')}
				</p>
			</div>
			<button
				type="button"
				class="inline-flex min-h-8 min-w-8 flex-shrink-0 items-center justify-center rounded p-1.5 text-fg-muted transition-colors duration-100 hover:bg-surface-3/50 hover:text-fg"
				aria-label="Close"
				disabled={busy}
				onclick={() => onClose?.()}
			>
				<Icon name="close" class="h-4 w-4" />
			</button>
		</header>

		<div class="flex min-h-0 flex-1 flex-col gap-4 overflow-y-auto px-5 py-4">
			<p id="{uid}-intro" class="text-sm text-fg-muted">
				An admin reviews this request. Once it is approved, the server downloads the file and the model shows up in your
				models.
			</p>

			<dl class="grid grid-cols-1 gap-x-6 gap-y-3 sm:grid-cols-2">
				<div class="min-w-0">
					<dt class="mb-1 font-mono text-2xs uppercase tracking-[0.07em] text-fg-subtle">Model</dt>
					<dd class="break-words text-sm text-fg">{detail.name}</dd>
				</div>
				<div class="min-w-0">
					<dt class="mb-1 font-mono text-2xs uppercase tracking-[0.07em] text-fg-subtle">Version</dt>
					<dd class="break-words text-sm text-fg">{version.name || '—'}</dd>
				</div>
				<div class="min-w-0 sm:col-span-2">
					<dt class="mb-1 font-mono text-2xs uppercase tracking-[0.07em] text-fg-subtle">File</dt>
					<dd class="break-all font-mono text-xs text-fg">{file.name}</dd>
				</div>
				<div class="min-w-0">
					<dt class="mb-1 font-mono text-2xs uppercase tracking-[0.07em] text-fg-subtle">Size</dt>
					<dd class="font-mono text-sm tabular-nums text-fg">{sizeText || 'Unknown'}</dd>
				</div>
				{#if fileMeta}
					<div class="min-w-0">
						<dt class="mb-1 font-mono text-2xs uppercase tracking-[0.07em] text-fg-subtle">Format</dt>
						<dd class="text-sm text-fg">{fileMeta}</dd>
					</div>
				{/if}
			</dl>

			{#each warnings as warning (warning.title)}
				<HostUi name="Alert" props={{ variant: warning.tone, density: 'compact', icon: true, title: warning.title }}>
					{warning.text}
				</HostUi>
			{/each}

			{#if problems.size}
				<HostUi name="Alert" props={{ variant: 'danger', density: 'compact', icon: true, title: 'Too large for your permissions' }}>
					{problems.size}
				</HostUi>
			{/if}

			<HostUi name="DetailField" props={{ label: 'File as', id: `${uid}-type`, wide: true, error: typeError, help: typeHelp }}>
				<select id="{uid}-type" class="input w-full text-sm" bind:value={form.modelType} disabled={busy} data-autofocus>
					{#if !form.modelType}
						<option value="" disabled>Choose a model type…</option>
					{/if}
					{#each choice.types as type (type)}
						<option value={type}>{modelTypeLabel(type)}</option>
					{/each}
				</select>
			</HostUi>

			<HostUi name="DetailField" props={{ label: 'Note for the admin (optional)', id: `${uid}-note`, wide: true, error: problems.note || null, help: noteHelp }}>
				<textarea
					id="{uid}-note"
					class="input w-full text-sm"
					rows="3"
					maxlength={MAX_NOTE_LENGTH}
					bind:value={form.note}
					disabled={busy}
					aria-invalid={problems.note ? 'true' : undefined}
				></textarea>
			</HostUi>

			<LicenceBox license={detail.license} />

			{#if serverError}
				<HostUi name="Alert" props={{ variant: 'danger', density: 'compact', icon: true, title: 'The server refused this request' }}>
					{serverError.message}
				</HostUi>
			{/if}
		</div>

		<footer class="flex items-center justify-end gap-2 border-t border-line px-5 py-3">
			<HostUi name="Button" props={{ variant: 'ghost', size: 'sm', disabled: busy, onclick: () => onClose?.() }}>Cancel</HostUi>
			<HostUi name="Button" props={{ variant: 'primary', size: 'sm', type: 'submit', icon: 'mail', loading: busy, disabled: !ready }}>Send request</HostUi>
		</footer>
	</form>
	</div>
</div>
{/if}
