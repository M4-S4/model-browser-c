<script>
	let { tone = 'neutral', title = undefined, class: className = '', children } = $props();

	const TONES = {
		neutral: 'bg-surface-2 text-fg-muted border border-line-strong',
		success: 'bg-success/10 text-success border border-success/25',
		warning: 'bg-warning/10 text-warning border border-warning/25',
		danger: 'bg-danger/10 text-danger border border-danger/25',
		info: 'bg-info/10 text-info border border-info/25',
		signal: 'bg-signal/10 text-signal border border-signal/25'
	};
</script>

<span class="inline-flex items-center gap-1 whitespace-nowrap rounded px-1.5 py-0 text-2xs font-medium {TONES[tone] ?? TONES.neutral} {className}" {title}>
	{@render children?.()}
</span>
