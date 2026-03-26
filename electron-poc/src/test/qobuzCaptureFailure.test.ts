import { describe, expect, it } from 'vitest'

import {
  buildQobuzCaptureFailurePlan,
  classifyQobuzCaptureFailure,
  shouldRetryQobuzCaptureFailure,
} from '../lib/qobuzCaptureFailure'

describe('qobuz capture failure planning', () => {
  it('classifies verification failures explicitly', () => {
    const failure = classifyQobuzCaptureFailure(
      'Hidden Qobuz playback could not be verified against the selected track.',
      null,
    )

    expect(failure.kind).toBe('verification_failed')
    expect(failure.code).toBe('QOBUZ_VERIFICATION_FAILED')
    expect(failure.retryable).toBe(true)
  })

  it('classifies upstream segment failures before generic playback stop', () => {
    const failure = classifyQobuzCaptureFailure(
      'Qobuz playback ended early after 12.5s.',
      {
        playbackHealth: {
          bodySnippet: 'Akamai segment 504 while playing the current track.',
        },
      },
    )

    expect(failure.kind).toBe('segment_fetch_failed')
    expect(failure.code).toBe('QOBUZ_SEGMENT_FETCH_FAILED')
    expect(failure.retryable).toBe(true)
  })

  it('bounds retries for retryable failures', () => {
    const failure = classifyQobuzCaptureFailure(
      'Qobuz playback ended early after 9.0s of expected 31.0s.',
      null,
    )

    expect(shouldRetryQobuzCaptureFailure(failure, 1, 2)).toBe(true)
    expect(shouldRetryQobuzCaptureFailure(failure, 2, 2)).toBe(false)
  })

  it('cleans partial captures even when the user cancels', () => {
    const plan = buildQobuzCaptureFailurePlan({
      message: 'Capture cancelled',
      cancelled: true,
      attempt: 1,
      maxAttempts: 2,
    })

    expect(plan.status).toBe('cancelled')
    expect(plan.cleanupPartialCapture).toBe(true)
    expect(plan.willRetry).toBe(false)
  })
})
