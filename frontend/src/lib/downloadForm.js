import { allowedTypes, exceedsSizeLimit, sizeLimitBytes } from './capabilities.js';
import { formatBytes } from './format.js';

export const MAX_SUBDIR_LENGTH = 255;
export const MAX_FILENAME_LENGTH = 255;

// eslint-disable-next-line no-control-regex
const CONTROL = /[\u0000-\u001f\u007f]/;

export function baseName(name) {
	const text = typeof name === 'string' ? name : '';
	const parts = text.split(/[\\/]/).filter(Boolean);
	return parts.length ? parts[parts.length - 1] : '';
}

export function extensionOf(name) {
	const base = baseName(name);
	const dot = base.lastIndexOf('.');
	return dot > 0 ? base.slice(dot).toLowerCase() : '';
}

export function validateSubdir(text) {
	const value = typeof text === 'string' ? text.trim() : '';
	if (!value) return '';
	if (CONTROL.test(value) || value.includes('\\') || value.includes(':')) {
		return 'Separate folders with /; no backslash, colon or control character';
	}
	if (value.startsWith('/')) return 'The folder must be relative: remove the leading /';
	const segments = value.replace(/\/+$/, '').split('/');
	if (segments.some((segment) => ['', '.', '..'].includes(segment.trim()))) {
		return "No empty, '.' or '..' folder";
	}
	if (value.length > MAX_SUBDIR_LENGTH) return `At most ${MAX_SUBDIR_LENGTH} characters`;
	return '';
}

export function cleanSubdir(text) {
	return (typeof text === 'string' ? text.trim() : '').replace(/\/+$/, '');
}

export function validateFilename(text, original) {
	const value = typeof text === 'string' ? text.trim() : '';
	if (!value || value === '.' || value === '..') return 'Enter a file name';
	if (value.includes('/') || value.includes('\\') || value.includes(':') || CONTROL.test(value)) {
		return 'A plain file name: no folder, colon or control character';
	}
	const wanted = extensionOf(original);
	if (wanted && extensionOf(value) !== wanted) return `Keep the ${wanted} extension`;
	if (value.length > MAX_FILENAME_LENGTH) return `At most ${MAX_FILENAME_LENGTH} characters`;
	return '';
}

export function filenameEdited(text, original) {
	return (typeof text === 'string' ? text.trim() : '') !== baseName(original);
}

export function sizeProblem(me, sizeBytes, { purpose = 'download' } = {}) {
	if (!exceedsSizeLimit(me, sizeBytes, { purpose })) return '';
	const limitGb = sizeLimitBytes(me, { purpose }) / 1024 ** 3;
	return purpose === 'request'
		? `This file is ${formatBytes(sizeBytes)}: your permissions let you ask for files of at most ${limitGb} GB`
		: `This file is ${formatBytes(sizeBytes)}: your permissions allow files of at most ${limitGb} GB`;
}

export function initialForm(me, detail, file) {
	return {
		modelType: allowedTypes(me, detail?.mb_type).selected,
		subdir: '',
		filename: baseName(file?.name),
		destination: '',
		group: ''
	};
}

export function formProblems(form, { me, detail, file }) {
	const { types } = allowedTypes(me, detail?.mb_type);
	let modelType = '';
	if (!form.modelType) modelType = 'Choose what to file this model as';
	else if (!types.includes(form.modelType)) modelType = 'Your permissions do not allow this model type';
	return {
		modelType,
		subdir: validateSubdir(form.subdir),
		filename: filenameEdited(form.filename, file?.name) ? validateFilename(form.filename, file?.name) : '',
		size: sizeProblem(me, file?.size_bytes)
	};
}

export function canSubmit(problems) {
	return Object.values(problems).every((problem) => !problem);
}

export function buildDownloadBody({ detail, version, file, form }) {
	const body = {
		source: detail.source,
		model_id: String(detail.id),
		version_id: String(version.id),
		file_id: String(file.id)
	};
	if (form.modelType) body.model_type = form.modelType;
	const subdir = cleanSubdir(form.subdir);
	if (subdir) body.subdir = subdir;
	if (filenameEdited(form.filename, file.name)) body.filename = form.filename.trim();
	if (form.destination) body.destination_backend_id = form.destination;
	if (form.group) body.assign_group_id = form.group;
	return body;
}

export function downloadWarnings({ detail, version, file }) {
	const warnings = [];
	if (version?.early_access) {
		warnings.push({
			tone: 'warning',
			title: 'Early access',
			text: 'This version is in early access on the source: the download fails unless the source account used by the server has access.'
		});
	}
	if (detail?.gated && file?.auth_provider_id) {
		warnings.push({
			tone: 'warning',
			title: 'Needs a login',
			text: 'The source only serves this file to signed-in accounts that accepted its terms: the server uses the provider plugin’s credentials.'
		});
	}
	const license = detail?.license;
	if (license && Array.isArray(license.commercial_use) && license.commercial_use.length === 0) {
		warnings.push({ tone: 'info', title: 'No commercial use', text: 'The licence does not allow commercial use of this model.' });
	}
	return warnings;
}

export function licenseFlags(license) {
	if (!license || typeof license !== 'object') return [];
	const flags = [];
	if (Array.isArray(license.commercial_use)) {
		flags.push(license.commercial_use.length ? `Commercial use: ${license.commercial_use.join(', ')}` : 'No commercial use');
	}
	if (license.no_credit === true) flags.push('No credit required');
	if (license.no_credit === false) flags.push('Credit required');
	if (license.derivatives === true) flags.push('Derivatives allowed');
	if (license.derivatives === false) flags.push('No derivatives');
	if (license.different_license === true) flags.push('Derivatives may use another licence');
	if (license.different_license === false) flags.push('Derivatives keep this licence');
	return flags;
}
