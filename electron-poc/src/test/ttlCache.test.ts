import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { TtlCache } from '../lib/systemRuntime/ttlCache'

describe('TtlCache', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    vi.setSystemTime(new Date('2026-03-05T12:00:00Z'))
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it('returns hit before expiry and miss after expiry', () => {
    const cache = new TtlCache<string>(1000)
    cache.set('value')

    expect(cache.get()).toBe('value')

    vi.advanceTimersByTime(999)
    expect(cache.get()).toBe('value')

    vi.advanceTimersByTime(2)
    expect(cache.get()).toBeNull()
  })

  it('clear removes cached value immediately', () => {
    const cache = new TtlCache<number>(5000)
    cache.set(42)
    cache.clear()
    expect(cache.get()).toBeNull()
  })
})
