export type PreviewErrorCode =
  | 'MISSING_CACHE_FILE'
  | 'STALE_SESSION'
  | 'MISSING_SOURCE_FILE'

export function previewLoadMessage(code?: string, fallback?: string): string {
  if (code === 'STALE_SESSION') {
    return 'This is a historical session and cached stems are no longer available. Run a new separation to preview this track.'
  }
  if (code === 'MISSING_CACHE_FILE') {
    return 'Cached preview file is missing. Run a new separation to regenerate this stem.'
  }
  if (code === 'MISSING_SOURCE_FILE') {
    return 'Source audio file is missing or moved. Verify file paths and try again.'
  }
  return fallback || 'Unable to load preview audio.'
}

export function exportFailureMessage(
  code?: string,
  error?: string,
  hint?: string,
): string {
  const base = previewLoadMessage(code, error || 'Export failed')
  if (!hint) return base
  return `${base} ${hint}`
}

