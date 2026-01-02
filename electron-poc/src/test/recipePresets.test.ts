import { describe, expect, it } from 'vitest'
import type { Recipe } from '@/types/recipes'
import { recipeToPreset } from '@/lib/recipePresets'
import { getRequiredModels } from '@/presets'

describe('recipe presets', () => {
  it('executes recipes via modelId=recipe.id but requires step model IDs', () => {
    const recipe: Recipe = {
      id: 'pipeline_demo',
      name: 'Pipeline Demo',
      description: 'Demo',
      type: 'pipeline',
      target: 'vocals',
      steps: [
        { step_name: 'separate', model_id: 'bs-roformer-viperx-1297' },
        { action: 'phase_fix', source_model: 'becruily-vocal' },
      ],
    }

    const preset = recipeToPreset(recipe)
    expect(preset.isRecipe).toBe(true)
    expect(preset.modelId).toBe('pipeline_demo')
    expect(preset.stems).toEqual(['vocals', 'instrumental'])

    const required = getRequiredModels(preset)
    expect(required).toContain('bs-roformer-viperx-1297')
    expect(required).toContain('becruily-vocal')
    expect(required).not.toContain('pipeline_demo')
  })

  it('uses explicit step outputs for display stems', () => {
    const recipe: Recipe = {
      id: 'chain_drums',
      name: 'Chain Drums',
      type: 'chained',
      target: 'drums',
      steps: [
        { step: 1, name: 'extract', model_id: 'htdemucs', output: 'drums' },
        { step: 2, name: 'split', model_id: 'mdx23c-drumsep-jarredou', output: ['kick', 'snare'] },
      ],
    }

    const preset = recipeToPreset(recipe)
    expect(preset.stems).toEqual(['drums', 'kick', 'snare'])
  })
})
