<script>
	import HostUi from '../../../../../sdk/HostUi.svelte';
	import Icon from './Icon.svelte';
	import {
		DEFAULT_BASE_MODELS_LABEL,
		DEFAULT_TYPES_LABEL,
		clearFilters,
		matchOptions,
		sourceDefaults,
		toggleType
	} from '../lib/search.js';
	import { escapeCloses } from '../lib/dom.js';

	let { info, filters, count = 0, onChange } = $props();

	const uid = $props.id();
	let open = $state(false);
	let alignRight = $state(true);
	let maxHeight = $state(480);
	let root = $state();
	let toggle = $state();
	let baseQuery = $state('');

	const types = $derived(info?.types ?? []);
	const baseModels = $derived(info?.base_models ?? []);
	const periods = $derived(info?.periods ?? []);
	const typesLabel = $derived(info?.types_label || DEFAULT_TYPES_LABEL);
	const baseLabel = $derived(info?.base_models_label || DEFAULT_BASE_MODELS_LABEL);
	const baseNoun = $derived(baseLabel.toLowerCase());
	const offered = $derived(types.length > 0 || baseModels.length > 0 || periods.length > 0);
	const baseMatches = $derived(matchOptions(baseModels, baseQuery));
	const period = $derived(filters.period || sourceDefaults(info).period);

	function show() {
		const rect = toggle?.getBoundingClientRect();
		const width = Math.min(480, window.innerWidth * 0.9);
		alignRight = !rect || rect.right >= width + 8;
		maxHeight = rect ? Math.max(200, Math.floor(window.innerHeight - rect.bottom - 16)) : 480;
		open = true;
	}

	function close(returnFocus = true) {
		open = false;
		baseQuery = '';
		if (returnFocus) toggle?.focus();
	}

	function onWindowClick(event) {
		if (open && root && !root.contains(event.target)) close(false);
	}

	function setBaseModel(value) {
		onChange?.({ ...filters, baseModel: value });
	}
</script>

<svelte:window onclick={onWindowClick} />

{#if offered}
	<div class="relative flex-shrink-0" bind:this={root}>
		<button
			bind:this={toggle}
			type="button"
			class="input flex h-8 items-center gap-1.5 whitespace-nowrap text-xs font-medium"
			aria-haspopup="dialog"
			aria-expanded={open}
			onclick={() => (open ? close(false) : show())}
		>
			<Icon name="filter" class="h-3.5 w-3.5 text-fg-subtle" />
			<span>Filters</span>
			{#if count > 0}
				<span class="font-mono tabular-nums text-signal">{count}</span>
			{/if}
			<Icon name="chevron-down" class="h-3 w-3 text-fg-subtle" />
		</button>

		{#if open}
			<div
				class="absolute top-[calc(100%+6px)] z-40 overflow-y-auto rounded-xl"
				style="{alignRight ? 'right: 0' : 'left: 0'}; max-height: {maxHeight}px"
				use:escapeCloses={() => close()}
			>
				<HostUi
					name="FilterPopoverFrame"
					props={{
						label: `${info.label} filters`,
						width: 'w-[30rem]',
						onClearAll: () => onChange?.(clearFilters(filters)),
						onClose: () => close()
					}}
				>
					{#if types.length > 0}
						<div class="col-span-2 flex flex-col gap-1.5">
							<span class="text-xs font-medium text-fg-muted" id="{uid}-types">{typesLabel}</span>
							<div
								class="flex max-h-28 flex-wrap gap-1.5 overflow-y-auto rounded border border-line-strong bg-surface-2 p-1.5"
								role="group"
								aria-labelledby="{uid}-types"
							>
								{#each types as option (option.value)}
									{@const on = filters.types.includes(option.value)}
									<button
										type="button"
										class="inline-flex h-6 items-center rounded border px-1.5 text-xs font-medium {on
											? 'border-signal/25 bg-signal/10 text-signal'
											: 'border-line-strong text-fg-muted hover:text-fg'}"
										aria-pressed={on}
										onclick={() => onChange?.(toggleType(filters, option.value, !on))}
									>
										{option.label}
									</button>
								{/each}
							</div>
						</div>
					{/if}

					{#if baseModels.length > 0}
						<div class="col-span-2 flex flex-col gap-1.5">
							<label class="text-xs font-medium text-fg-muted" for="{uid}-base">{baseLabel}</label>
							<input
								id="{uid}-base"
								type="text"
								class="input h-8 w-full text-xs"
								placeholder={`Search ${baseModels.length} ${baseNoun}${baseModels.length === 1 ? '' : 's'}`}
								autocomplete="off"
								bind:value={baseQuery}
							/>
							<div class="max-h-32 overflow-y-auto rounded border border-line-strong bg-surface-2 p-1" role="group" aria-label={baseLabel}>
								<button
									type="button"
									class="block w-full rounded px-2 py-1 text-left text-xs {filters.baseModel === ''
										? 'bg-signal/10 text-signal'
										: 'text-fg-muted hover:bg-surface-3/50 hover:text-fg'}"
									aria-pressed={filters.baseModel === ''}
									onclick={() => setBaseModel('')}
								>
									Any {baseNoun}
								</button>
								{#each baseMatches as option (option.value)}
									<button
										type="button"
										class="block w-full truncate rounded px-2 py-1 text-left text-xs {filters.baseModel === option.value
											? 'bg-signal/10 text-signal'
											: 'text-fg-muted hover:bg-surface-3/50 hover:text-fg'}"
										aria-pressed={filters.baseModel === option.value}
										onclick={() => setBaseModel(option.value)}
									>
										{option.label}
									</button>
								{/each}
								{#if baseMatches.length === 0}
									<p class="px-2 py-1 text-xs text-fg-subtle">No {baseNoun} matches.</p>
								{/if}
							</div>
						</div>
					{/if}

					{#if periods.length > 0}
						<div class="col-span-2">
							<HostUi
								name="SelectFilterField"
								props={{ label: 'Period', value: period, options: periods, onChange: (value) => onChange?.({ ...filters, period: value }) }}
							/>
						</div>
					{/if}
				</HostUi>
			</div>
		{/if}
	</div>
{/if}
