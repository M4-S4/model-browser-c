import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
	activeCount,
	detectTransitions,
	isUnfinished,
	normalizeRows,
	noteStarted,
	rowKey,
	rowStatus,
	rowSubtitle,
	rowTitle,
	shouldPoll,
	startToast,
	transitionToast
} from './downloads.js';

function row(status, download = null, extra = {}) {
	return {
		pending: {
			sha256: extra.sha256 ?? 'a'.repeat(64),
			source: 'civitai',
			file_name: 'detailTweaker.safetensors',
			status,
			error: extra.error ?? null,
			assign_to: 'user:u1',
			info: { model: { name: 'Detail Tweaker', versions: [{ name: 'v1.0' }] } }
		},
		download
	};
}

function core(status, extra = {}) {
	return {
		id: 'd1',
		status,
		progress: 0,
		total_bytes: 239_600_000,
		downloaded_bytes: 0,
		speed_bytes_per_sec: 0,
		error_message: null,
		filename: 'detailTweaker.safetensors',
		...extra
	};
}

test('normalizeRows keeps rows with a pending row and a key', () => {
	assert.equal(normalizeRows([row('queued'), null, { pending: {} }, { download: core('pending') }]).length, 1);
	assert.deepEqual(normalizeRows(undefined), []);
	assert.equal(rowKey(row('queued')), 'a'.repeat(64));
	assert.equal(rowKey({}), '');
});

test('isUnfinished follows the pending status, then the core download', () => {
	assert.equal(isUnfinished(row('queued', core('pending'))), true);
	assert.equal(isUnfinished(row('queued', core('downloading'))), true);
	assert.equal(isUnfinished(row('indexed')), true);
	assert.equal(isUnfinished(row('assigned')), true);
	assert.equal(isUnfinished(row('enriched', core('completed'))), false);
	assert.equal(isUnfinished(row('failed', core('paused'))), false);
	assert.equal(isUnfinished({ pending: { status: 'other' }, download: core('paused') }), true);
	assert.equal(isUnfinished({ pending: { status: 'other' }, download: core('completed') }), false);
});

test('shouldPoll and activeCount', () => {
	const rows = [row('enriched'), row('queued', core('downloading')), row('failed'), row('indexed')];
	assert.equal(shouldPoll(rows), true);
	assert.equal(activeCount(rows), 2);
	assert.equal(shouldPoll([row('enriched'), row('failed')]), false);
	assert.equal(shouldPoll([]), false);
});

test('rowTitle and rowSubtitle', () => {
	assert.equal(rowTitle(row('queued')), 'Detail Tweaker');
	assert.equal(rowSubtitle(row('queued')), 'detailTweaker.safetensors · v1.0');
	const bare = { pending: { sha256: 'x', file_name: 'x.gguf', status: 'queued', info: {} }, download: null };
	assert.equal(rowTitle(bare), 'x.gguf');
	assert.equal(rowSubtitle(bare), 'x.gguf');
});

test('rowStatus of a running download carries progress, sizes and speed', () => {
	const status = rowStatus(
		row('queued', core('downloading', { progress: 0.4567, downloaded_bytes: 109_400_000, speed_bytes_per_sec: 4_404_019 }))
	);
	assert.deepEqual(status, {
		label: 'Downloading',
		tone: 'signal',
		progress: 45,
		sizeText: '104.3 MB / 228.5 MB',
		speedText: '4.2 MB/s',
		error: ''
	});
});

test('rowStatus names every stage', () => {
	assert.equal(rowStatus(row('queued', null)).label, 'Waiting');
	assert.equal(rowStatus(row('queued', core('pending'))).label, 'Waiting');
	assert.equal(rowStatus(row('queued', core('paused', { progress: 0.5 }))).label, 'Paused');
	assert.equal(rowStatus(row('queued', core('paused', { progress: 0.5 }))).progress, 50);
	assert.equal(rowStatus(row('queued', core('completed'))).label, 'Indexing');
	assert.equal(rowStatus(row('indexed')).label, 'Assigning');
	assert.equal(rowStatus(row('assigned')).label, 'Filling in details');
	assert.deepEqual(rowStatus(row('enriched', core('completed'))), {
		label: 'Ready',
		tone: 'success',
		progress: null,
		sizeText: '228.5 MB',
		speedText: '',
		error: ''
	});
	assert.equal(rowStatus({ pending: { status: 'mystery' } }).label, 'mystery');
});

test('rowStatus reports errors from the pending row or the core download', () => {
	assert.deepEqual(rowStatus(row('failed', core('failed', { error_message: 'HTTP 401' }), { error: 'Download failed: HTTP 401' })), {
		label: 'Failed',
		tone: 'danger',
		progress: null,
		sizeText: '',
		speedText: '',
		error: 'Download failed: HTTP 401'
	});
	assert.equal(rowStatus(row('queued', core('failed', { error_message: 'checksum mismatch' }))).error, 'checksum mismatch');
	assert.equal(rowStatus(row('queued', core('cancelled'))).label, 'Cancelled');
});

test('detectTransitions reports rows that finish during the session only', () => {
	const a = 'a'.repeat(64);
	const b = 'b'.repeat(64);
	const c = 'c'.repeat(64);
	let state = detectTransitions(new Map(), [row('queued', core('downloading'), { sha256: a }), row('enriched', null, { sha256: b })]);
	assert.deepEqual(state.events, []);
	state = detectTransitions(state.seen, [
		row('enriched', core('completed'), { sha256: a }),
		row('enriched', null, { sha256: b }),
		row('failed', null, { sha256: c })
	]);
	assert.deepEqual(
		state.events.map((event) => [event.kind, event.key]),
		[['done', a]]
	);
	state = detectTransitions(state.seen, [row('enriched', core('completed'), { sha256: a })]);
	assert.deepEqual(state.events, []);
	assert.equal(state.seen.get(c), 'failed');
});

test('noteStarted makes a fast download report its end', () => {
	const key = 'd'.repeat(64);
	const seen = noteStarted(new Map(), { sha256: key, status: 'indexed' });
	assert.equal(seen.get(key), 'indexed');
	const { events } = detectTransitions(seen, [row('failed', null, { sha256: key, error: 'Model not indexed' })]);
	assert.deepEqual(
		events.map((event) => event.kind),
		['failed']
	);
	assert.equal(noteStarted(seen, null).size, 1);
});

test('transitionToast says what happened', () => {
	assert.deepEqual(transitionToast({ kind: 'done', row: row('enriched') }), {
		level: 'success',
		message: 'Detail Tweaker is ready in your models'
	});
	assert.deepEqual(transitionToast({ kind: 'failed', row: row('failed', null, { error: 'HTTP 401' }) }), {
		level: 'error',
		message: 'Detail Tweaker: download failed (HTTP 401)'
	});
	assert.deepEqual(transitionToast({ kind: 'failed', row: row('failed') }), {
		level: 'error',
		message: 'Detail Tweaker: download failed'
	});
});

test('startToast covers started, already installed and already queued', () => {
	const pending = { file_name: 'x.safetensors', assign_to: 'user:u1' };
	assert.deepEqual(startToast({ pending }), { level: 'success', message: 'Download started: x.safetensors' });
	assert.deepEqual(startToast({ pending, already_installed: true }), {
		level: 'success',
		message: 'x.safetensors is already installed, now assigned to you'
	});
	assert.deepEqual(startToast({ pending: { ...pending, assign_to: 'group:g1' }, already_installed: true }, [{ id: 'g1', name: 'Artists' }]), {
		level: 'success',
		message: 'x.safetensors is already installed, now assigned to the group Artists'
	});
	assert.deepEqual(startToast({ pending: { ...pending, assign_to: 'none' }, already_installed: true }), {
		level: 'info',
		message: 'x.safetensors is already installed'
	});
	assert.deepEqual(startToast({ pending, already_queued: true }), { level: 'info', message: 'x.safetensors is already being downloaded' });
	assert.deepEqual(startToast(null), { level: 'success', message: 'Download started: The file' });
});
