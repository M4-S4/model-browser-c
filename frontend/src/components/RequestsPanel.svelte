<script>
	import { onMount, tick } from 'svelte';
	import HostUi from '../../../../../sdk/HostUi.svelte';
	import Icon from './Icon.svelte';
	import RequestRow from './RequestRow.svelte';
	import {
		POLL_INTERVAL_MS,
		STATUSES,
		approveRequest,
		approveToast,
		canDecide,
		cancelRequest,
		cancelToast,
		defaultTab,
		listRequests,
		normalizeListing,
		panelTabs,
		rejectRequest,
		rejectToast,
		statusLabel,
		tabQuery,
		tabRequests
	} from '../lib/requests.js';
	import { escapeCloses, hostLayer, portal, toast, trapFocus } from '../lib/dom.js';

	let { api, me, count = null, sourceLabel = (id) => id, opener = null, onClose, onChanged } = $props();

	const uid = $props.id();
	let mounted = $state(false);
	const startingTab = () => defaultTab(me);
	let tab = $state(startingTab());
	let status = $state('');
	let view = $state.raw({ key: '', requests: [], phase: 'loading', error: null });
	let refreshing = $state(false);
	let busyId = $state('');
	let dialog = $state();

	let seq = 0;
	let timer = null;
	let destroyed = false;

	const decide = $derived(canDecide(me));
	const tabs = $derived(panelTabs(me, count));
	const subtitle = $derived(
		decide ? 'What users asked for: approving downloads the file for them.' : 'The models you asked an admin for.'
	);
	const empty = $derived(
		tab === 'review'
			? { title: 'Nothing to review', description: 'New requests show up here, and you get a notification for each one.' }
			: tab === 'mine'
				? { title: 'No request yet', description: 'Open a model and choose Request on one of its files: an admin reviews it.' }
				: { title: status ? `No ${statusLabel(status).toLowerCase()} request` : 'No request yet', description: 'The requests of every user show up here.' }
	);

	onMount(() => {
		mounted = true;
		load();
		return () => {
			destroyed = true;
			clearTimeout(timer);
		};
	});

	async function load() {
		clearTimeout(timer);
		timer = null;
		const mine = ++seq;
		const key = `${tab}:${status}`;
		if (view.key !== key) view = { key, requests: [], phase: 'loading', error: null };
		refreshing = true;
		const result = await listRequests(api, tabQuery(tab, status));
		if (mine !== seq || destroyed) return;
		refreshing = false;
		if (result.ok) {
			view = { key, requests: tabRequests(tab, normalizeListing(result.data).requests, me, status), phase: 'ready', error: null };
		} else if (view.phase === 'ready') {
			view = { ...view, error: result.error };
		} else {
			view = { key, requests: [], phase: 'error', error: result.error };
		}
		onChanged?.();
		timer = setTimeout(load, POLL_INTERVAL_MS);
		await tick();
		if (dialog?.isConnected && (!document.activeElement || document.activeElement === document.body)) {
			dialog.focus({ preventScroll: true });
		}
	}

	function selectTab(id) {
		if (id === tab || busyId) return;
		tab = id;
		load();
	}

	async function act(request, kind, options) {
		if (busyId) return { ok: false, message: '' };
		busyId = request.id;
		let result;
		if (kind === 'approve') result = await approveRequest(api, request.id, options);
		else if (kind === 'reject') result = await rejectRequest(api, request.id, options);
		else result = await cancelRequest(api, request.id);
		busyId = '';
		const message = (kind === 'approve' ? approveToast : kind === 'reject' ? rejectToast : cancelToast)(result, request);
		toast(message.level, message.message);
		if (destroyed) {
			if (message.refresh) onChanged?.();
			return { ok: result.ok, message: '' };
		}
		if (message.refresh) load();
		return { ok: result.ok, message: result.ok ? '' : message.message };
	}

	function onBackdrop(event) {
		if (event.target === event.currentTarget) onClose?.();
	}
</script>

{#if mounted}
<div
	use:portal
	use:hostLayer
	use:escapeCloses={() => onClose?.()}
	class="fixed inset-0 flex justify-end bg-black/60 backdrop-blur-sm"
	role="presentation"
	onclick={onBackdrop}
>
	<div
		bind:this={dialog}
		class="flex h-full w-full max-w-2xl flex-col bg-surface-1 shadow-overlay outline-none"
		role="dialog"
		aria-modal="true"
		aria-labelledby="{uid}-title"
		tabindex="-1"
		use:trapFocus={opener}
		onkeydown={(event) => event.stopPropagation()}
	>
		<header class="flex flex-shrink-0 items-start gap-2 border-b border-line px-4 py-3 sm:px-5">
			<div class="min-w-0 flex-1">
				<h2 id="{uid}-title" class="text-lg font-semibold text-fg">Requests</h2>
				<p class="mt-0.5 text-xs text-fg-muted">{subtitle}</p>
			</div>
			<button
				type="button"
				class="inline-flex min-h-8 min-w-8 items-center justify-center rounded p-1.5 text-fg-muted transition-colors duration-100 hover:bg-surface-3/50 hover:text-fg disabled:opacity-50"
				aria-label="Refresh the requests"
				title="Refresh"
				disabled={refreshing}
				onclick={() => load()}
			>
				<Icon name="refresh" class="h-4 w-4 {refreshing ? 'animate-spin' : ''}" />
			</button>
			<button
				type="button"
				class="inline-flex min-h-8 min-w-8 items-center justify-center rounded p-1.5 text-fg-muted transition-colors duration-100 hover:bg-surface-3/50 hover:text-fg"
				aria-label="Close the requests"
				title="Close (Esc)"
				data-autofocus
				onclick={() => onClose?.()}
			>
				<Icon name="close" class="h-4 w-4" />
			</button>
		</header>

		{#if tabs.length > 1}
			<HostUi name="DetailTabs" props={{ tabs, active: tab, onSelect: selectTab, ariaLabel: 'Requests' }} />
		{/if}

		{#if tab === 'all'}
			<div class="flex flex-shrink-0 items-center gap-2 border-b border-line px-4 py-1.5 sm:px-5">
				<label class="text-xs font-medium text-fg-muted" for="{uid}-status">Status</label>
				<select id="{uid}-status" class="input h-7 w-auto max-w-full py-0 text-xs" bind:value={status} disabled={!!busyId} onchange={() => load()}>
					<option value="">Any status</option>
					{#each STATUSES as value (value)}
						<option {value}>{statusLabel(value)}</option>
					{/each}
				</select>
			</div>
		{/if}

		<div class="min-h-0 flex-1 overflow-y-auto">
			{#if view.phase === 'loading'}
				<div class="flex items-center justify-center py-16">
					<HostUi name="Spinner" props={{ size: 'lg' }} />
				</div>
			{:else if view.phase === 'error'}
				<HostUi
					name="LoadErrorState"
					props={{ title: "Couldn't load the requests", message: view.error?.message ?? 'Something went wrong', onRetry: () => load() }}
				/>
			{:else if view.requests.length === 0}
				<div class="p-4 sm:p-5">
					<HostUi name="EmptyState" props={{ icon: 'inbox', title: empty.title, description: empty.description, compact: true }} />
				</div>
			{:else}
				{#if view.error}
					<p class="px-4 pt-3 text-xs text-danger sm:px-5" role="status">Couldn't refresh: {view.error.message}</p>
				{/if}
				<ul class="divide-y divide-line" aria-label="Requests">
					{#each view.requests as request (request.id)}
						<RequestRow
							{request}
							{me}
							{sourceLabel}
							showUser={tab !== 'mine'}
							canAct={decide && tab !== 'mine'}
							busy={busyId === request.id}
							locked={busyId !== ''}
							onCancel={(target) => act(target, 'cancel')}
							onApprove={(target, options) => act(target, 'approve', options)}
							onReject={(target, options) => act(target, 'reject', options)}
						/>
					{/each}
				</ul>
			{/if}
		</div>
	</div>
</div>
{/if}
