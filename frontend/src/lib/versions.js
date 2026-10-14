function isObject(value) {
	return value !== null && typeof value === 'object' && !Array.isArray(value);
}

export function isPartial(version) {
	return isObject(version) && version.partial === true;
}

export function hasPartialVersions(detail) {
	return Array.isArray(detail?.versions) && detail.versions.some(isPartial);
}

export function versionRef(modelId, versionId) {
	return `${modelId}:${versionId}`;
}

export function mediaModelId(detail, version, byVersion) {
	return byVersion && version ? versionRef(detail.id, version.id) : detail.id;
}

function refused(message) {
	return { ok: false, error: { status: 0, code: 'invalid_response', message, retryAfter: null } };
}

export function mergeVersion(detail, versionId, answer) {
	const id = String(versionId ?? '');
	const versions = Array.isArray(detail?.versions) ? detail.versions : [];
	const index = versions.findIndex((entry) => String(entry?.id) === id);
	if (!id || index < 0) return refused('This version is not in the list any more');
	if (!isObject(answer) || !Array.isArray(answer.versions)) return refused('Unexpected answer from the server');
	if (answer.id != null && detail.id != null && String(answer.id) !== String(detail.id)) {
		return refused('The server answered for another model');
	}
	const loaded = answer.versions.find((entry) => isObject(entry) && String(entry.id) === id);
	if (!loaded) return refused('The server did not send this version');
	if (isPartial(loaded)) return refused('The server sent this version without its files');
	return {
		ok: true,
		detail: { ...detail, versions: versions.map((entry, position) => (position === index ? { ...loaded, partial: false } : entry)) }
	};
}
