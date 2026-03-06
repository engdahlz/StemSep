import { describe, it, expect } from 'vitest'
import { previewLoadMessage, exportFailureMessage } from '../lib/previewErrors'

describe('preview error mapping', () => {
  it('maps stale sessions to actionable message', () => {
    const message = previewLoadMessage('STALE_SESSION')
    expect(message).toContain('historical session')
    expect(message).toContain('Run a new separation')
  })

  it('maps missing cache files deterministically', () => {
    const message = previewLoadMessage('MISSING_CACHE_FILE')
    expect(message).toContain('Cached preview file is missing')
  })

  it('keeps fallback text for unknown code', () => {
    expect(previewLoadMessage('SOMETHING_ELSE', 'fallback')).toBe('fallback')
  })

  it('includes hint in export failure message', () => {
    const message = exportFailureMessage('MISSING_SOURCE_FILE', 'Export failed', 'Try again')
    expect(message).toContain('Source audio file is missing')
    expect(message).toContain('Try again')
  })
})

