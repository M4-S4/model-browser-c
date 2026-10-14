import { mount, unmount } from 'svelte';
import ModelBrowserPage from './ModelBrowserPage.svelte';

export { ModelBrowserPage as default };

export function mountPlugin(target, props) {
	return mount(ModelBrowserPage, { target, props });
}

export function unmountPlugin(instance) {
	unmount(instance);
}
