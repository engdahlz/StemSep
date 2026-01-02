import { describe, it, expect, beforeEach, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'

import { SimplePresetPicker } from '../components/simple/SimplePresetPicker'
import { ConfigurePage } from '../components/ConfigurePage'
import type { Preset } from '../presets'

function mockElectronApi() {
  ;(window as any).electronAPI = {
    getGpuDevices: vi.fn().mockResolvedValue({ has_cuda: false, gpus: [] }),
  }
}

describe('Simple Mode UX (multi-step + details)', () => {
  beforeEach(() => {
    mockElectronApi()
  })

  it('SimplePresetPicker can hide/show multi-step presets', async () => {
    const user = userEvent.setup()

    const normal: Preset = {
      id: 'p_normal',
      name: 'Normal preset',
      description: 'Normal',
      stems: ['vocals', 'instrumental'],
      qualityLevel: 'balanced',
      category: 'instrumental',
      tags: [],
    } as any

    const recipe: Preset = {
      id: 'p_recipe',
      name: 'Recipe preset',
      description: 'Multi-step',
      stems: ['vocals', 'instrumental'],
      qualityLevel: 'balanced',
      category: 'utility',
      tags: [],
      isRecipe: true,
      recipe: {
        id: 'r1',
        type: 'pipeline',
        requiredModels: ['m1', 'm2'],
        steps: [{ step_name: 'Step A', model_id: 'm1' }, { step_name: 'Step B', model_id: 'm2' }],
      },
    } as any

    render(
      <SimplePresetPicker
        presets={[normal, recipe]}
        selectedPresetId={'p_normal'}
        onSelectPreset={() => {}}
        availability={{ m1: { available: true }, m2: { available: true } }}
      />
    )

    // More options is open by default when there are no recommended presets.
    expect(screen.getByText('Recipe preset')).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: /multi-step\s+on/i }))

    expect(screen.queryByText('Recipe preset')).not.toBeInTheDocument()
  })

  it('Preset cards use an info icon tooltip (no Details button)', async () => {
    const user = userEvent.setup()

    const recipe: Preset = {
      id: 'p_recipe',
      name: 'Recipe preset',
      description: 'Multi-step',
      stems: ['vocals', 'instrumental'],
      qualityLevel: 'balanced',
      category: 'utility',
      tags: [],
      isRecipe: true,
      recipe: {
        id: 'r1',
        type: 'pipeline',
        requiredModels: ['m1', 'm2'],
        steps: [{ step_name: 'Step A', model_id: 'm1' }, { step_name: 'Step B', model_id: 'm2' }],
      },
    } as any

    const models = [
      { id: 'm1', name: 'M1', installed: true },
      { id: 'm2', name: 'M2', installed: true },
    ]

    render(
      <ConfigurePage
        fileName="test.wav"
        filePath="C:/tmp/test.wav"
        onBack={() => {}}
        onConfirm={() => {}}
        initialPresetId="p_recipe"
        presets={[recipe]}
        models={models as any}
        availability={{ m1: { available: true, model_name: 'M1' }, m2: { available: true, model_name: 'M2' } } as any}
      />
    )

    expect(screen.queryByRole('button', { name: /details/i })).not.toBeInTheDocument()

    const infoBtn = screen.getByRole('button', { name: /preset info: recipe preset/i })
    await user.hover(infoBtn)

    // Tooltip renders a single-line summary; ConfigurePage also has its own Required models section.
    expect(screen.getByText('Required models: M1, M2')).toBeInTheDocument()
  })
})
