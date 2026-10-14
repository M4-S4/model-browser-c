export function toast(level, message) {
	window.__potionui?.notifications?.toast?.(level, message);
}

export function prefersReducedMotion() {
	try {
		return window.matchMedia?.('(prefers-reduced-motion: reduce)')?.matches === true;
	} catch {
		return false;
	}
}

export function portal(node) {
	document.body.appendChild(node);
	return {
		destroy() {
			node.remove();
		}
	};
}

export function hostLayer(node, tier = 'overlay') {
	const action = window.__potionui?.layers?.overlayLayer;
	if (typeof action === 'function') return action(node, tier);
	node.style.zIndex = tier === 'overlay' ? '1001' : '10001';
	return { destroy() {} };
}

const FOCUSABLE =
	'a[href], button:not([disabled]), input:not([disabled]):not([type="hidden"]), select:not([disabled]), textarea:not([disabled]), video[controls], [tabindex]:not([tabindex="-1"])';

function focusables(node) {
	return [...node.querySelectorAll(FOCUSABLE)].filter(
		(element) => !element.closest('[hidden]') && element.getClientRects().length > 0
	);
}

function visible(element) {
	return !!element && !element.closest('[hidden]') && element.getClientRects().length > 0;
}

export function trapFocus(node, returnTo) {
	let previous = returnTo ?? (document.activeElement instanceof HTMLElement ? document.activeElement : null);
	let attempts = 0;
	let timer = null;

	function focusInitial() {
		timer = null;
		if (!node.isConnected) return;
		const active = document.activeElement;
		if (active !== node && node.contains(active)) return;
		const wanted = node.querySelector('[data-autofocus]');
		const target = visible(wanted) ? wanted : wanted ? null : focusables(node)[0];
		if (target) {
			target.focus({ preventScroll: true });
			if (document.activeElement === target) return;
		} else if (active !== node) {
			node.focus?.({ preventScroll: true });
		}
		attempts += 1;
		if (attempts < 40) timer = setTimeout(focusInitial, 25);
	}
	queueMicrotask(focusInitial);

	function onKeydown(event) {
		if (event.key !== 'Tab') return;
		const items = focusables(node);
		if (items.length === 0) {
			event.preventDefault();
			node.focus?.();
			return;
		}
		const first = items[0];
		const last = items[items.length - 1];
		const active = document.activeElement;
		if (event.shiftKey && (active === first || !node.contains(active))) {
			event.preventDefault();
			last.focus();
		} else if (!event.shiftKey && (active === last || !node.contains(active))) {
			event.preventDefault();
			first.focus();
		}
	}

	node.addEventListener('keydown', onKeydown);
	return {
		update(next) {
			if (next) previous = next;
		},
		destroy() {
			clearTimeout(timer);
			node.removeEventListener('keydown', onKeydown);
			const active = document.activeElement;
			if (previous?.isConnected && (!active || active === document.body || node.contains(active))) {
				previous.focus({ preventScroll: true });
			}
		}
	};
}

const escapeStack = [];
let listening = false;

function onWindowKeydown(event) {
	if (event.key !== 'Escape' || escapeStack.length === 0 || event.defaultPrevented) return;
	const top = escapeStack[escapeStack.length - 1];
	event.preventDefault();
	event.stopPropagation();
	top();
}

export function onEscape(close) {
	escapeStack.push(close);
	if (!listening) {
		window.addEventListener('keydown', onWindowKeydown, true);
		listening = true;
	}
	return () => {
		const index = escapeStack.lastIndexOf(close);
		if (index !== -1) escapeStack.splice(index, 1);
		if (escapeStack.length === 0 && listening) {
			window.removeEventListener('keydown', onWindowKeydown, true);
			listening = false;
		}
	};
}

export function escapeCloses(node, close) {
	let current = close;
	const remove = onEscape(() => current?.());
	return {
		update(next) {
			current = next;
		},
		destroy: remove
	};
}

const homes = new WeakMap();

export function slotOf(getNode) {
	return {
		slot: (el) => {
			const node = getNode();
			if (!node) return undefined;
			if (!homes.has(node)) homes.set(node, node.parentNode);
			const active = document.activeElement;
			const hadFocus = active instanceof HTMLElement && node.contains(active);
			el.append(node);
			if (hadFocus && document.activeElement !== active) active.focus({ preventScroll: true });
			return () => {
				if (node.parentNode !== el) return;
				const home = homes.get(node);
				if (home) home.append(node);
				else node.remove();
			};
		}
	};
}

export function scrollParent(element) {
	let node = element?.parentElement ?? null;
	while (node && node !== document.body) {
		const { overflowY } = getComputedStyle(node);
		if (overflowY === 'auto' || overflowY === 'scroll' || overflowY === 'overlay') return node;
		node = node.parentElement;
	}
	return null;
}

export async function copyText(text) {
	try {
		await navigator.clipboard.writeText(text);
		return true;
	} catch {
		try {
			const area = document.createElement('textarea');
			area.value = text;
			area.setAttribute('readonly', '');
			area.style.position = 'fixed';
			area.style.opacity = '0';
			document.body.appendChild(area);
			area.select();
			const done = document.execCommand('copy');
			area.remove();
			return done;
		} catch {
			return false;
		}
	}
}
