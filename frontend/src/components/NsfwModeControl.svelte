<script>
	import HostUi from '../../../../../sdk/HostUi.svelte';
	import Icon from './Icon.svelte';
	import { effectiveModes, nsfwInfo } from '../lib/nsfw.js';

	let { me, busy = false, onChange } = $props();

	const uid = $props.id();
	const info = $derived(nsfwInfo(me));
	const items = $derived(effectiveModes(me).map((item) => ({ ...item, disabled: item.disabled || busy })));
	const hint = $derived(
		`App-wide setting (${info.settingKey}): it also applies to History and the other pages.` +
			(info.ceiling === 'blur' ? ' Your content policy allows blurred NSFW media at most.' : '')
	);
</script>

<div class="flex flex-shrink-0 items-center gap-2">
	<span class="hidden text-xs text-fg-muted sm:inline" id="{uid}-label">Sensitive content</span>
	{#if info.locked}
		<HostUi name="Tooltip" props={{ text: 'Your content policy hides NSFW media: it is never sent to this page.', position: 'bottom' }}>
			<span class="inline-flex h-8 items-center gap-1.5 rounded border border-line bg-surface-2 px-2.5 text-xs font-medium text-fg-muted">
				<Icon name="eye-off" class="h-3.5 w-3.5" />
				Hidden by policy
			</span>
		</HostUi>
	{:else}
		<HostUi name="Tooltip" props={{ text: hint, position: 'bottom' }}>
			<span class="inline-flex">
				<HostUi
					name="SegmentedControl"
					props={{
						variant: 'toggle',
						ariaLabel: 'Sensitive content (app-wide setting)',
						items,
						selected: info.mode,
						onSelect: (id) => {
							if (id !== info.mode && !busy) onChange?.(id);
						}
					}}
				/>
			</span>
		</HostUi>
		<span class="sr-only">{hint}</span>
	{/if}
</div>
