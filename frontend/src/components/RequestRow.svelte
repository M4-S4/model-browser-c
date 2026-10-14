<script>
	import { tick } from 'svelte';
	import ActionButton from './ActionButton.svelte';
	import Icon from './Icon.svelte';
	import Pill from './Pill.svelte';
	import {
		MAX_NOTE_LENGTH,
		approvalOptions,
		approvalProblems,
		approvalTypeChoice,
		canCancel,
		canDecide,
		canDecideOn,
		decisionLine,
		isOpen,
		requestByline,
		requestMeta,
		requestTitle,
		requestedType,
		sourceHost,
		statusInfo
	} from '../lib/requests.js';
	import { modelTypeLabel, thumbUrl } from '../lib/format.js';
	import { onEscape } from '../lib/dom.js';

	let {
		request,
		me,
		showUser = true,
		canAct = false,
		busy = false,
		locked = false,
		sourceLabel = (id) => id,
		onCancel,
		onApprove,
		onReject
	} = $props();

	const uid = $props.id();
	let mode = $state('');
	let approval = $state({ modelType: '', note: '' });
	let rejectNote = $state('');
	let formError = $state('');
	let approveButton = $state();
	let rejectButton = $state();
	let typeSelect = $state();
	let rejectField = $state();

	let thumbFailed = $state(false);

	const thumb = $derived(thumbUrl(request.snapshot.thumbnail_url, 80));
	const title = $derived(requestTitle(request));
	const meta = $derived(requestMeta(request, sourceLabel).join(' · '));
	const status = $derived(statusInfo(request.status));
	const host = $derived(sourceHost(request));
	const byline = $derived(requestByline(request, { showUser }));
	const decision = $derived(decisionLine(request));
	const who = $derived(request.username || 'the requester');
	const deciding = $derived(canAct && canDecideOn(me, request));
	const ownOpen = $derived(canAct && isOpen(request) && canDecide(me) && !canDecideOn(me, request));
	const cancellable = $derived(canCancel(me, request));
	const typeChoice = $derived(approvalTypeChoice(request));
	const problems = $derived(approvalProblems(request, approval));
	const approvalReady = $derived(!problems.modelType && !problems.note);
	const requested = $derived(requestedType(request));
	const typeHelp = $derived(
		!requested
			? 'The request names no type: choose one'
			: approval.modelType && approval.modelType !== requested
				? `Instead of ${modelTypeLabel(requested)}, the type asked for`
				: 'The type asked for'
	);

	$effect(() => {
		if (!deciding && mode) mode = '';
	});

	$effect(() => {
		if (!mode) return;
		return onEscape(() => closeForm());
	});

	async function openForm(kind) {
		formError = '';
		if (kind === 'approve') approval = { modelType: typeChoice.selected, note: '' };
		else rejectNote = '';
		mode = kind;
		await tick();
		(kind === 'approve' ? typeSelect : rejectField)?.focus();
	}

	async function closeForm() {
		const kind = mode;
		mode = '';
		formError = '';
		await tick();
		(kind === 'approve' ? approveButton : rejectButton)?.focus();
	}

	async function approve(event) {
		event.preventDefault();
		if (!approvalReady || busy || locked) return;
		formError = '';
		const outcome = await onApprove?.(request, approvalOptions(request, approval));
		if (outcome?.ok) mode = '';
		else formError = outcome?.message ?? '';
	}

	async function reject(event) {
		event.preventDefault();
		if (rejectNote.trim().length > MAX_NOTE_LENGTH || busy || locked) return;
		formError = '';
		const outcome = await onReject?.(request, { note: rejectNote });
		if (outcome?.ok) mode = '';
		else formError = outcome?.message ?? '';
	}
</script>

{#snippet noteBox(label, text)}
	<div class="rounded border border-line bg-surface-2 px-2.5 py-1.5">
		<p class="font-mono text-2xs uppercase tracking-[0.07em] text-fg-subtle">{label}</p>
		<p class="mt-0.5 whitespace-pre-wrap break-words text-xs text-fg">{text}</p>
	</div>
{/snippet}

<li class="space-y-2 px-4 py-3 sm:px-5">
	<div class="flex items-start gap-3">
		{#if thumb && !thumbFailed}
			<img
				src={thumb}
				alt=""
				class="h-10 w-10 flex-shrink-0 rounded bg-surface-3 object-cover"
				loading="lazy"
				decoding="async"
				onerror={() => (thumbFailed = true)}
			/>
		{:else}
			<span class="flex h-10 w-10 flex-shrink-0 items-center justify-center rounded bg-surface-3 text-fg-subtle" aria-hidden="true">
				<Icon name="model" class="h-4 w-4" />
			</span>
		{/if}
		<div class="min-w-0 flex-1">
			<p class="flex min-w-0 items-center gap-1">
				<span class="truncate text-sm font-medium text-fg" title={title}>{title}</span>
				{#if request.snapshot.source_url}
					<a
						class="inline-flex flex-shrink-0 items-center rounded p-0.5 text-fg-subtle transition-colors hover:text-fg"
						href={request.snapshot.source_url}
						target="_blank"
						rel="noopener noreferrer"
						aria-label={`Open ${title} on ${host || 'its source'} (new tab)`}
						title={`Open on ${host || 'the source'}`}
					>
						<Icon name="external-link" class="h-3.5 w-3.5" />
					</a>
				{/if}
			</p>
			{#if meta}
				<p class="truncate font-mono text-2xs text-fg-subtle" title={meta}>{meta}</p>
			{/if}
		</div>
		<Pill tone={status.tone} title={status.hint || undefined}>{status.label}</Pill>
	</div>

	{#if request.snapshot.nsfw || request.snapshot.poi || request.snapshot.already_installed}
		<div class="flex flex-wrap items-center gap-1.5">
			{#if request.snapshot.nsfw}
				<Pill tone="danger" title="The model is marked NSFW">NSFW</Pill>
			{/if}
			{#if request.snapshot.poi}
				<Pill tone="warning" title="The model depicts a real person">Real person</Pill>
			{/if}
			{#if request.snapshot.already_installed}
				<Pill tone="success" title="The file is already on the server: approving the request only assigns it">Installed</Pill>
			{/if}
		</div>
	{/if}

	{#if byline.text}
		<p class="text-xs text-fg-muted" title={byline.title || undefined}>{byline.text}</p>
	{/if}

	{#if request.note}
		{@render noteBox(showUser ? `Note from ${who}` : 'Your note', request.note)}
	{/if}

	{#if decision}
		<p class="text-xs text-fg-muted">{decision}</p>
		{#if request.decision_note}
			{@render noteBox(request.decided_by_username ? `Answer from ${request.decided_by_username}` : 'Answer', request.decision_note)}
		{/if}
	{/if}

	{#if ownOpen && !mode}
		<p class="text-xs text-fg-subtle">Your own request: another approver decides on it.</p>
	{/if}

	{#if mode === 'approve'}
		<form class="space-y-3 rounded border border-line-strong bg-surface-2 p-3" novalidate onsubmit={approve} aria-label={`Approve the request of ${who}`}>
			<div>
				<label class="mb-1 block text-xs font-medium text-fg-muted" for="{uid}-type">File as</label>
				<select id="{uid}-type" class="input h-8 w-full py-0 text-xs" bind:this={typeSelect} bind:value={approval.modelType} disabled={busy}>
					{#if !approval.modelType}
						<option value="" disabled>Choose a model type…</option>
					{/if}
					{#each typeChoice.types as type (type)}
						<option value={type}>{modelTypeLabel(type)}</option>
					{/each}
				</select>
				<p class="mt-1 text-2xs text-fg-subtle">{typeHelp}</p>
			</div>
			<div>
				<label class="mb-1 block text-xs font-medium text-fg-muted" for="{uid}-approve-note">Note to {who} (optional)</label>
				<textarea
					id="{uid}-approve-note"
					class="input w-full text-xs"
					rows="2"
					maxlength={MAX_NOTE_LENGTH}
					bind:value={approval.note}
					disabled={busy}
				></textarea>
			</div>
			{#if formError}
				<p class="text-xs text-danger" role="alert">{formError}</p>
			{/if}
			<div class="flex flex-wrap items-center justify-end gap-2">
				<ActionButton variant="ghost" disabled={busy} onclick={closeForm}>Back</ActionButton>
				<ActionButton variant="primary" type="submit" icon="check" loading={busy} disabled={!approvalReady || locked}>Approve</ActionButton>
			</div>
		</form>
	{:else if mode === 'reject'}
		<form class="space-y-3 rounded border border-line-strong bg-surface-2 p-3" novalidate onsubmit={reject} aria-label={`Reject the request of ${who}`}>
			<div>
				<label class="mb-1 block text-xs font-medium text-fg-muted" for="{uid}-reject-note">Reason, sent to {who} (optional)</label>
				<textarea
					id="{uid}-reject-note"
					class="input w-full text-xs"
					rows="2"
					maxlength={MAX_NOTE_LENGTH}
					bind:this={rejectField}
					bind:value={rejectNote}
					disabled={busy}
				></textarea>
			</div>
			{#if formError}
				<p class="text-xs text-danger" role="alert">{formError}</p>
			{/if}
			<div class="flex flex-wrap items-center justify-end gap-2">
				<ActionButton variant="ghost" disabled={busy} onclick={closeForm}>Back</ActionButton>
				<ActionButton variant="danger" type="submit" icon="close" loading={busy} disabled={locked}>Reject</ActionButton>
			</div>
		</form>
	{:else if deciding || cancellable}
		<div class="flex flex-wrap items-center gap-2">
			{#if deciding}
				<ActionButton bind:element={approveButton} variant="primary" icon="check" disabled={locked} onclick={() => openForm('approve')}>
					Approve…
				</ActionButton>
				<ActionButton bind:element={rejectButton} variant="secondary" icon="close" disabled={locked} onclick={() => openForm('reject')}>
					Reject…
				</ActionButton>
			{/if}
			{#if cancellable}
				<ActionButton
					variant="ghost"
					loading={busy}
					disabled={locked}
					ariaLabel={`Cancel the request for ${title}`}
					onclick={() => onCancel?.(request)}
				>
					Cancel request
				</ActionButton>
			{/if}
		</div>
	{/if}
</li>
