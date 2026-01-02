import { describe, it, expect, beforeEach, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'

import { ConfigurePage } from '../components/ConfigurePage'
import SeparationConfigDialog from '../components/SeparationConfigDialog'
import type { Preset } from '../presets'

type ConfirmedConfig = {
  volumeCompensation?: {
    enabled: boolean
    stage?: 'export' | 'blend' | 'both'
    dbPerExtraModel?: number
  }
}

function mockElectronApi() {
  ;(window as any).electronAPI = {
    getGpuDevices: vi.fn().mockResolvedValue({ has_cuda: false, gpus: [] }),
  }
}

describe('Volume Compensation UI config wiring', () => {
  beforeEach(() => {
    mockElectronApi()
  })

  it('ConfigurePage omits volumeCompensation when VC is OFF', async () => {
    const user = userEvent.setup()

    const presets: Preset[] = [
      {
        id: 'p1',
        name: 'Preset 1',
        description: 'Preset',
        stems: ['vocals', 'instrumental'],
        modelId: 'model1',
      } as any,
    ]

    const models = [{ id: 'model1', name: 'Model 1', installed: true }]

    let confirmed: ConfirmedConfig | null = null

    render(
      <ConfigurePage
        fileName="test.wav"
        filePath="C:/tmp/test.wav"
        onBack={() => {}}
        onConfirm={(cfg: any) => {
          confirmed = cfg as ConfirmedConfig
        }}
        initialPresetId="p1"
        presets={presets}
        models={models as any}
      />
    )

    await user.click(screen.getByRole('button', { name: /start separation/i }))

    expect(confirmed).not.toBeNull()
    expect((confirmed as any)?.volumeCompensation).toBeUndefined()
  })

  it('ConfigurePage sends best defaults when VC is ON', async () => {
    const user = userEvent.setup()

    const presets: Preset[] = [
      {
        id: 'p1',
        name: 'Preset 1',
        description: 'Preset',
        stems: ['vocals', 'instrumental'],
        modelId: 'model1',
      } as any,
    ]

    const models = [{ id: 'model1', name: 'Model 1', installed: true }]

    let confirmed: ConfirmedConfig | null = null

    render(
      <ConfigurePage
        fileName="test.wav"
        filePath="C:/tmp/test.wav"
        onBack={() => {}}
        onConfirm={(cfg: any) => {
          confirmed = cfg as ConfirmedConfig
        }}
        initialPresetId="p1"
        presets={presets}
        models={models as any}
      />
    )

    // VC is now under Enhancements (Optional)
    await user.click(screen.getByRole('button', { name: /enhancements/i }))

    // Toggle VC ON
    await user.click(screen.getByText(/enable vc/i))

    await user.click(screen.getByRole('button', { name: /start separation/i }))

    expect(confirmed).not.toBeNull()
    expect((confirmed as any)?.volumeCompensation).toEqual({
      enabled: true,
      stage: 'both',
      dbPerExtraModel: 3,
    })
  })

  it('SeparationConfigDialog omits volumeCompensation when VC is OFF', async () => {
    const user = userEvent.setup()

    const presets: Preset[] = [
      {
        id: 'p1',
        name: 'Preset 1',
        description: 'Preset',
        stems: ['vocals', 'instrumental'],
        modelId: 'model1',
      } as any,
    ]

    let confirmed: ConfirmedConfig | null = null

    render(
      <SeparationConfigDialog
        open={true}
        onOpenChange={() => {}}
        onConfirm={(cfg: any) => {
          confirmed = cfg as ConfirmedConfig
        }}
        initialPresetId="p1"
        presets={presets}
        modelMap={{ p1: 'model1' }}
        models={[{ id: 'model1', name: 'Model 1', installed: true }] as any}
      />
    )

    await user.click(screen.getByRole('button', { name: /start separation/i }))

    expect(confirmed).not.toBeNull()
    expect((confirmed as any)?.volumeCompensation).toBeUndefined()
  })

  it('SeparationConfigDialog sends best defaults when VC is ON', async () => {
    const user = userEvent.setup()

    const presets: Preset[] = [
      {
        id: 'p1',
        name: 'Preset 1',
        description: 'Preset',
        stems: ['vocals', 'instrumental'],
        modelId: 'model1',
      } as any,
    ]

    let confirmed: ConfirmedConfig | null = null

    render(
      <SeparationConfigDialog
        open={true}
        onOpenChange={() => {}}
        onConfirm={(cfg: any) => {
          confirmed = cfg as ConfirmedConfig
        }}
        initialPresetId="p1"
        presets={presets}
        modelMap={{ p1: 'model1' }}
        models={[{ id: 'model1', name: 'Model 1', installed: true }] as any}
      />
    )

    const vc = screen.getByLabelText(/volume compensation/i)
    await user.click(vc)

    await user.click(screen.getByRole('button', { name: /start separation/i }))

    expect(confirmed).not.toBeNull()
    expect((confirmed as any)?.volumeCompensation).toEqual({
      enabled: true,
      stage: 'both',
      dbPerExtraModel: 3,
    })
  })
})
