<script>
	import Icon from './Icon.svelte';

	let {
		element = $bindable(),
		variant = 'secondary',
		type = 'button',
		icon = '',
		loading = false,
		disabled = false,
		title = undefined,
		ariaLabel = undefined,
		onclick = undefined,
		children
	} = $props();

	const VARIANTS = {
		primary: 'bg-accent text-accent-contrast hover:bg-accent-hover active:bg-accent-active',
		secondary: 'bg-surface-3 text-fg hover:bg-line-hover',
		ghost: 'text-fg-muted hover:text-fg hover:bg-surface-3/50',
		danger: 'bg-danger-solid text-white hover:bg-danger-solid/90'
	};
</script>

<button
	bind:this={element}
	{type}
	class="inline-flex touch-manipulation items-center justify-center gap-1.5 whitespace-nowrap rounded px-2.5 py-1 text-xs font-medium transition-colors duration-100 disabled:cursor-not-allowed disabled:opacity-50 {VARIANTS[variant] ?? VARIANTS.secondary}"
	disabled={disabled || loading}
	{title}
	aria-label={ariaLabel}
	aria-busy={loading ? 'true' : undefined}
	{onclick}
>
	{#if loading}
		<span class="h-3.5 w-3.5 animate-spin rounded-full border-2 border-line-strong border-t-current" aria-hidden="true"></span>
	{:else if icon}
		<Icon name={icon} class="h-3.5 w-3.5" />
	{/if}
	{@render children?.()}
</button>
