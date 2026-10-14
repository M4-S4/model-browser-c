<script>
	import { onMount } from 'svelte';
	import HostUi from '../../../../sdk/HostUi.svelte';
	import {
		cloneState,
		fetchPermissions,
		isDirty,
		optionsFor,
		parseMaxFileGb,
		savePermissions,
		toFormState,
		toggleItem,
		toPutBody,
		validateState
	} from './lib/permissions.js';

	let { pluginId = 'model-browser-c' } = $props();

	const CAPABILITY_LABELS = {
		browse: 'Browse',
		request: 'Request',
		download: 'Download',
		approve: 'Approve'
	};
	const SOURCE_LABELS = {
		civitai: 'CivitAI',
		huggingface: 'Hugging Face',
		openmodeldb: 'OpenModelDB',
		civarchive: 'CivArchive (experimental)'
	};
	const ASSIGN_TO_ITEMS = [
		{ id: 'user', label: 'The user' },
		{ id: 'group', label: "One of the user's groups" }
	];

	let phase = $state('loading');
	let loadError = $state('');
	let initial = $state.raw(null);
	let form = $state(null);
	let saving = $state(false);
	let saveError = $state('');

	let latestLoad = 0;

	const groups = $derived(form?.groups ?? []);
	const problems = $derived(form ? validateState(form) : []);
	const dirty = $derived(isDirty(initial, form));
	const canSave = $derived(dirty && problems.length === 0 && !saving);

	onMount(() => {
		load();
		return () => {
			latestLoad += 1;
		};
	});

	async function load() {
		const mine = ++latestLoad;
		phase = 'loading';
		saveError = '';
		const result = await fetchPermissions({ pluginId });
		if (mine !== latestLoad) return;
		if (!result.ok) {
			loadError = result.error;
			phase = 'error';
			return;
		}
		accept(result.data);
		phase = 'ready';
	}

	function accept(data) {
		initial = toFormState(data);
		form = toFormState(data);
	}

	async function save() {
		if (!canSave) return;
		const body = toPutBody(form);
		saving = true;
		saveError = '';
		const result = await savePermissions(body, { pluginId });
		saving = false;
		if (!result.ok) {
			saveError = result.error;
			notify('error', `Permissions not saved: ${result.error}`);
			return;
		}
		accept(result.data);
		notify('success', 'Permissions saved');
	}

	function reset() {
		form = cloneState(initial);
		saveError = '';
	}

	function notify(level, message) {
		window.__potionui?.notifications?.toast?.(level, message);
	}

	function setCapability(group, capability, on) {
		group.capabilities = toggleItem(group.capabilities, capability, on);
		saveError = '';
	}

	function setListParam(group, key, id, on) {
		group.params[key] = toggleItem(group.params[key], id, on);
		saveError = '';
	}

	function setParam(group, key, value) {
		group.params[key] = value;
		saveError = '';
	}

	const fieldId = (group, name) => `mb-permissions-${group.group_id}-${name}`;
</script>

<div class="space-y-4">
	<div class="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
		<p class="min-w-0 max-w-3xl text-sm text-fg-muted">
			What each user group can do in the Model Browser. <span class="text-fg">Browse</span>: see the page and search.
			<span class="text-fg">Request</span>: ask for a model, an admin decides.
			<span class="text-fg">Download</span>: download directly, without approval.
			<span class="text-fg">Approve</span>: handle other users' requests.
		</p>

		{#if phase === 'ready' && groups.length > 0}
			<div class="flex flex-shrink-0 items-center gap-2">
				{#if dirty}
					<span class="text-sm text-fg-muted">Unsaved changes</span>
				{/if}
				<HostUi name="Button" props={{ variant: 'ghost', size: 'sm', disabled: !dirty || saving, onclick: reset }}>Reset</HostUi>
				<HostUi name="Button" props={{ variant: 'primary', size: 'sm', disabled: !canSave, loading: saving, onclick: save }}>Save</HostUi>
			</div>
		{/if}
	</div>

	{#if phase === 'loading'}
		<div class="flex items-center justify-center py-12">
			<HostUi name="Spinner" props={{ size: 'lg' }} />
		</div>
	{:else if phase === 'error'}
		<HostUi
			name="LoadErrorState"
			props={{ title: "Couldn't load the permissions", message: loadError, onRetry: load }}
		/>
	{:else if groups.length === 0}
		<HostUi
			name="EmptyState"
			props={{ icon: 'group', title: 'No user groups', description: 'There is no user group to set permissions for.', compact: true }}
		/>
	{:else}
		{#if problems.length > 0}
			<div class="rounded-lg border border-warning/25 bg-warning/10 px-3 py-2 text-sm text-warning" role="status">
				<p class="font-medium">Fix this before saving</p>
				<ul class="mt-1 list-disc space-y-0.5 pl-5">
					{#each problems as problem}
						<li>{problem}</li>
					{/each}
				</ul>
			</div>
		{/if}

		{#if saveError}
			<div class="rounded-lg border border-danger/25 bg-danger/10 px-3 py-2 text-sm text-danger" role="alert">
				<p class="font-medium">Could not save the permissions</p>
				<p>{saveError}</p>
			</div>
		{/if}

		{#each groups as group (group.group_id)}
			{@const usesParams = group.capabilities.includes('download') || group.capabilities.includes('request')}
			{@const sizeInvalid = !parseMaxFileGb(group.params.max_file_gb).ok}
			<section class="rounded-lg border border-line bg-surface-1 p-4 shadow-raised" aria-label="Permissions of {group.name}">
				<div class="flex flex-wrap items-center gap-2">
					<h3 class="truncate text-base font-semibold text-fg">{group.name}</h3>
					{#if group.is_system}
						<HostUi name="Badge" props={{ size: 'sm', variant: 'neutral' }}>System</HostUi>
					{/if}
					<span class="font-mono text-xs text-fg-subtle">{group.group_id}</span>
				</div>

				<div class="mt-3 flex flex-wrap gap-x-6 gap-y-2">
					{#each form.allowed.capabilities as capability (capability)}
						{@const id = fieldId(group, capability)}
						{@const name = CAPABILITY_LABELS[capability] ?? capability}
						<div class="flex items-center gap-2">
							<HostUi
								name="Switch"
								props={{
									id,
									label: `${group.name}: ${name}`,
									checked: group.capabilities.includes(capability),
									disabled: saving,
									onchange: (on) => setCapability(group, capability, on)
								}}
							/>
							<label class="text-sm text-fg" for={id}>{name}</label>
						</div>
					{/each}
				</div>

				{#if usesParams}
					<div class="mt-4 space-y-3 border-t border-line pt-4">
						<p class="font-mono text-2xs font-medium uppercase tracking-[0.07em] text-fg-subtle">Download and request parameters</p>

						<div class="flex flex-col gap-1 sm:flex-row sm:gap-4">
							<span class="w-32 flex-shrink-0 text-sm text-fg-muted">Sources</span>
							<div class="flex flex-wrap gap-x-4 gap-y-1.5" role="group" aria-label="Sources of {group.name}">
								{#each optionsFor(form.allowed.sources, group.params.sources) as source (source)}
									<label class="inline-flex items-center gap-1.5 text-sm text-fg">
										<input
											type="checkbox"
											class="accent-accent"
											checked={group.params.sources.includes(source)}
											disabled={saving}
											onchange={(e) => setListParam(group, 'sources', source, e.currentTarget.checked)}
										/>
										{SOURCE_LABELS[source] ?? source}
									</label>
								{/each}
							</div>
						</div>

						<div class="flex flex-col gap-1 sm:flex-row sm:gap-4">
							<span class="w-32 flex-shrink-0 text-sm text-fg-muted">Model types</span>
							<div class="min-w-0 flex-1 space-y-1">
								<div class="flex flex-wrap gap-x-4 gap-y-1.5" role="group" aria-label="Model types of {group.name}">
									{#each optionsFor(form.allowed.model_types, group.params.model_types) as type (type)}
										<label class="inline-flex items-center gap-1.5 font-mono text-xs text-fg">
											<input
												type="checkbox"
												class="accent-accent"
												checked={group.params.model_types.includes(type)}
												disabled={saving}
												onchange={(e) => setListParam(group, 'model_types', type, e.currentTarget.checked)}
											/>
											{type}
										</label>
									{/each}
								</div>
								<p class="text-xs text-fg-subtle">
									{group.params.model_types.length === 0
										? 'None checked: every model type is allowed.'
										: 'Only the checked model types are allowed.'}
								</p>
							</div>
						</div>

						<div class="flex flex-col gap-1 sm:flex-row sm:items-center sm:gap-4">
							<label class="w-32 flex-shrink-0 text-sm text-fg-muted" for={fieldId(group, 'max_file_gb')}>Max file size</label>
							<div class="flex items-center gap-2">
								<HostUi
									name="Input"
									props={{
										id: fieldId(group, 'max_file_gb'),
										type: 'text',
										inputmode: 'decimal',
										autocomplete: 'off',
										placeholder: 'No limit',
										class: 'w-28 font-mono text-sm tabular-nums',
										value: group.params.max_file_gb,
										invalid: sizeInvalid,
										'aria-invalid': sizeInvalid ? 'true' : undefined,
										disabled: saving,
										oninput: (e) => setParam(group, 'max_file_gb', e.currentTarget.value)
									}}
								/>
								<span class="font-mono text-xs text-fg-subtle">GB</span>
								<span class="text-xs {sizeInvalid ? 'text-danger' : 'text-fg-subtle'}">
									{sizeInvalid ? 'A number above 0, decimals with a dot.' : 'Empty: no limit.'}
								</span>
							</div>
						</div>

						<div class="flex flex-col gap-1 sm:flex-row sm:items-center sm:gap-4">
							<span class="w-32 flex-shrink-0 text-sm text-fg-muted">Assign to</span>
							<div class="flex flex-wrap items-center gap-2">
								<HostUi
									name="SegmentedControl"
									props={{
										variant: 'toggle',
										ariaLabel: `Assign the models downloaded by ${group.name} to`,
										items: ASSIGN_TO_ITEMS.map((item) => ({ ...item, disabled: saving })),
										selected: group.params.assign_to,
										onSelect: (id) => setParam(group, 'assign_to', id)
									}}
								/>
								<span class="text-xs text-fg-subtle">
									{group.params.assign_to === 'group'
										? 'The user picks one of their groups when downloading.'
										: 'The model goes to the user who downloads it.'}
								</span>
							</div>
						</div>

						<div class="flex flex-col gap-1 sm:flex-row sm:items-center sm:gap-4">
							<label class="w-32 flex-shrink-0 text-sm text-fg-muted" for={fieldId(group, 'allow_remote')}>Remote backends</label>
							<div class="flex items-center gap-2">
								<HostUi
									name="Switch"
									props={{
										id: fieldId(group, 'allow_remote'),
										label: `${group.name}: allow remote backends`,
										checked: group.params.allow_remote,
										disabled: saving,
										onchange: (on) => setParam(group, 'allow_remote', on)
									}}
								/>
								<span class="text-sm text-fg">Allow choosing a remote backend as the destination</span>
							</div>
						</div>
					</div>
				{:else}
					<p class="mt-3 text-xs text-fg-subtle">Turn on Download or Request to choose what this group may download or ask for.</p>
				{/if}
			</section>
		{/each}

		<section class="rounded-lg border border-line bg-surface-1 p-4 shadow-raised">
			<h3 class="font-mono text-2xs font-medium uppercase tracking-[0.07em] text-fg-subtle">How the rights of a user's groups combine</h3>
			<ul class="mt-2 list-disc space-y-1 pl-4 text-xs text-fg-muted">
				<li>A user gets every right that at least one of their groups grants.</li>
				<li>
					A group's parameters bound both its downloads and its requests. For a user in several groups, the download parameters
					are merged from the groups that grant Download, the request parameters from the groups that grant Request: sources and
					types united, the largest size wins, no limit beats any limit, 'group' assignment as soon as one group says so.
				</li>
				<li>Admins have every right and no limit, whatever is set here.</li>
				<li>
					Until edited, All Users can browse and request, All Admins can do everything, and any other group can do nothing.
				</li>
			</ul>
		</section>
	{/if}
</div>
