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
      selection_envelope: {
        catalogTier: 'verified',
        sourceKind: 'guide',
        installPolicy: 'direct',
      },
      required_model_statuses: [
        {
          id: 'bs-roformer-viperx-1297',
          catalog_tier: 'verified',
          source_kind: 'guide',
          install_policy: 'direct',
          selection_envelope: {
            catalogTier: 'verified',
            sourceKind: 'guide',
            installPolicy: 'direct',
          },
        },
      ],
      steps: [
        { step_name: 'separate', model_id: 'bs-roformer-viperx-1297' },
        { action: 'phase_fix', source_model: 'becruily-vocal' },
      ],
    }

    const preset = recipeToPreset(recipe)
    expect(preset.isRecipe).toBe(true)
    expect(preset.modelId).toBe('pipeline_demo')
    expect(preset.stems).toEqual(['vocals', 'instrumental'])
    expect(preset.selectionEnvelope?.catalogTier).toBe('verified')
    expect(preset.recipe?.selection_envelope?.sourceKind).toBe('guide')
    expect(preset.recipe?.required_model_statuses?.[0]?.catalog_tier).toBe('verified')

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

  it('hides blocked recipes from simple recommendations even if simple_surface was requested', () => {
    const recipe: Recipe = {
      id: 'workflow_phantom_center_dual_beta2',
      name: 'Phantom Center Dual Beta2',
      type: 'pipeline',
      target: 'instrumental',
      simple_surface: true,
      surface_policy: 'verified_only',
      surface_blockers: ['candidate model is not verified for simple mode'],
      promotion_status: 'supported_advanced',
      qa_status: 'pending',
      steps: [{ step_name: 'split', model_id: 'gilliaan-monostereo-dual-beta2' }],
    }

    const preset = recipeToPreset(recipe)
    expect(preset.recommended).toBe(false)
    expect(preset.simpleVisible).toBe(false)
  })
})
