import { formatBytes, formatSpeed, percent } from './format.js';

export const POLL_INTERVAL_MS = 5000;

export const LIST_LIMIT = 30;

const PENDING_UNFINISHED = new Set(['queued', 'indexed', 'assigned']);
const PENDING_FINAL = new Set(['enriched', 'failed']);
const DOWNLOAD_UNFINISHED = new Set(['pending', 'downloading', 'paused']);

function isObject(value) {
	return value !== null && typeof value === 'object' && !Array.isArray(value);
}

export function normalizeRows(data) {
	return (Array.isArray(data) ? data : []).filter((row) => isObject(row) && isObject(row.pending) && rowKey(row));
}

export function rowKey(row) {
	const key = row?.pending?.sha256;
	return typeof key === 'string' ? key : '';
}

export function isUnfinished(row) {
	const pending = row?.pending?.status;
	if (PENDING_FINAL.has(pending)) return false;
	return PENDING_UNFINISHED.has(pending) || DOWNLOAD_UNFINISHED.has(row?.download?.status);
}

export function shouldPoll(rows) {
	return rows.some(isUnfinished);
}

export function activeCount(rows) {
	return rows.filter(isUnfinished).length;
}

export function rowTitle(row) {
	const name = row?.pending?.info?.model?.name;
	if (typeof name === 'string' && name) return name;
	return row?.pending?.file_name || row?.download?.filename || 'Download';
}

export function rowSubtitle(row) {
	const file = row?.pending?.file_name || row?.download?.filename || '';
	const versions = row?.pending?.info?.model?.versions;
	const version = Array.isArray(versions) && versions.length ? versions[0]?.name : '';
	return [file, version].filter(Boolean).join(' · ');
}

export function rowStatus(row) {
	const pending = row?.pending ?? {};
	const download = isObject(row?.download) ? row.download : null;
	const base = { progress: null, sizeText: '', speedText: '', error: '' };
	const downloaded = download?.downloaded_bytes;
	const total = download?.total_bytes;
	const sizeText =
		typeof total === 'number' && total > 0
			? `${formatBytes(typeof downloaded === 'number' ? downloaded : 0)} / ${formatBytes(total)}`
			: formatBytes(downloaded);

	switch (pending.status) {
		case 'enriched':
			return { ...base, label: 'Ready', tone: 'success', sizeText: typeof total === 'number' ? formatBytes(total) : '' };
		case 'failed':
			return { ...base, label: 'Failed', tone: 'danger', error: pending.error || download?.error_message || '' };
		case 'assigned':
			return { ...base, label: 'Filling in details', tone: 'info' };
		case 'indexed':
			return { ...base, label: 'Assigning', tone: 'info' };
		case 'queued':
			break;
		default:
			return { ...base, label: pending.status ? String(pending.status) : 'Unknown', tone: 'neutral' };
	}

	switch (download?.status) {
		case 'downloading':
			return {
				...base,
				label: 'Downloading',
				tone: 'signal',
				progress: percent(download.progress),
				sizeText,
				speedText: formatSpeed(download.speed_bytes_per_sec)
			};
		case 'paused':
			return { ...base, label: 'Paused', tone: 'warning', progress: percent(download.progress), sizeText };
		case 'completed':
			return { ...base, label: 'Indexing', tone: 'info', progress: 100, sizeText };
		case 'failed':
			return { ...base, label: 'Failed', tone: 'danger', error: download.error_message || pending.error || '' };
		case 'cancelled':
			return { ...base, label: 'Cancelled', tone: 'neutral', error: download.error_message || '' };
		default:
			return { ...base, label: 'Waiting', tone: 'neutral' };
	}
}

export function noteStarted(seen, pending) {
	const next = new Map(seen);
	if (isObject(pending) && typeof pending.sha256 === 'string' && pending.sha256) {
		next.set(pending.sha256, typeof pending.status === 'string' ? pending.status : 'queued');
	}
	return next;
}

export function detectTransitions(seen, rows) {
	const next = new Map(seen);
	const events = [];
	for (const row of rows) {
		const key = rowKey(row);
		if (!key) continue;
		const status = row.pending.status;
		const previous = seen.get(key);
		if (PENDING_FINAL.has(status) && previous !== undefined && !PENDING_FINAL.has(previous)) {
			events.push({ kind: status === 'enriched' ? 'done' : 'failed', key, row });
		}
		next.set(key, status);
	}
	return { events, seen: next };
}

export function transitionToast(event) {
	const name = rowTitle(event.row);
	if (event.kind === 'done') return { level: 'success', message: `${name} is ready in your models` };
	const error = rowStatus(event.row).error;
	return { level: 'error', message: error ? `${name}: download failed (${error})` : `${name}: download failed` };
}

export function startToast(data, groups = []) {
	const pending = isObject(data?.pending) ? data.pending : {};
	const file = pending.file_name || 'The file';
	if (data?.already_installed) {
		const target = typeof pending.assign_to === 'string' ? pending.assign_to : '';
		if (target.startsWith('group:')) {
			const id = target.slice('group:'.length);
			const name = groups.find((group) => group.id === id)?.name ?? id;
			return { level: 'success', message: `${file} is already installed, now assigned to the group ${name}` };
		}
		if (target.startsWith('user:')) return { level: 'success', message: `${file} is already installed, now assigned to you` };
		return { level: 'info', message: `${file} is already installed` };
	}
	if (data?.already_queued) return { level: 'info', message: `${file} is already being downloaded` };
	return { level: 'success', message: `Download started: ${file}` };
}
