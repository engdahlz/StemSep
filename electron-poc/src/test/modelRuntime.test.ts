import { describe, expect, it } from 'vitest'
import { modelRequiresFnoRuntime } from '../lib/systemRuntime/modelRuntime'

describe('modelRequiresFnoRuntime', () => {
  it('detects explicit runtime variant fno', () => {
    expect(
      modelRequiresFnoRuntime({ id: 'm1', runtime: { variant: 'fno' } }),
    ).toBe(true)
  })

  it('detects FNO markers from id/name/architecture', () => {
    expect(modelRequiresFnoRuntime({ id: 'fno-bsr-model' })).toBe(true)
    expect(modelRequiresFnoRuntime({ name: 'FNO Super Model' })).toBe(true)
    expect(modelRequiresFnoRuntime({ architecture: 'BS-Roformer-FNO' })).toBe(true)
  })

  it('does not block non-fno models', () => {
    expect(
      modelRequiresFnoRuntime({ id: 'bs-roformer-viperx-1297', architecture: 'BS-Roformer' }),
    ).toBe(false)
  })
})

