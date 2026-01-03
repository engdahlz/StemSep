import type { Recipe, RecipeStep } from '@/types/recipes'
import type { Preset } from '@/presets'

function extractModelIdsFromStep(step: RecipeStep): string[] {
  const ids: string[] = []

  const maybeModelId = (step as any)?.model_id
  if (typeof maybeModelId === 'string' && maybeModelId.trim()) {
    ids.push(maybeModelId)
  }

  const maybeSourceModel = (step as any)?.source_model
  if (typeof maybeSourceModel === 'string' && maybeSourceModel.trim()) {
    ids.push(maybeSourceModel)
  }

  return ids
}

export function getRecipeRequiredModels(recipe: Recipe): string[] {
  const models = new Set<string>()
  for (const step of recipe.steps || []) {
    for (const id of extractModelIdsFromStep(step)) {
      models.add(id)
    }
  }
  return Array.from(models)
}

function recipeDisplayStems(recipe: Recipe): string[] {
  // Prefer explicit declared outputs (common for chained workflows)
  const outputs: string[] = []
  for (const step of recipe.steps || []) {
    const out = (step as any)?.output
    if (typeof out === 'string' && out.trim()) outputs.push(out)
    if (Array.isArray(out)) {
      for (const s of out) {
        if (typeof s === 'string' && s.trim()) outputs.push(s)
      }
    }
  }
  if (outputs.length > 0) {
    return Array.from(new Set(outputs))
  }

  // Otherwise infer from target for UI display
  const target = (recipe.target || '').toLowerCase()
  if (target === 'vocals') return ['vocals', 'instrumental']
  if (target === 'instrumental') return ['instrumental', 'vocals']
  if (target === 'all') return ['vocals', 'instrumental']
  if (target) return [target]
  return []
}

function recipeQuality(recipe: Recipe): Preset['qualityLevel'] {
  const id = recipe.id.toLowerCase()
  if (id.startsWith('mode_quick')) return 'fast'
  if (id.startsWith('mode_quality')) return 'quality'
  if (id.startsWith('golden_') || recipe.name.includes('🏆')) return 'ultra'
  if (recipe.type === 'pipeline' || recipe.type === 'chained') return 'quality'
  return 'balanced'
}

function recipeEstimatedVram(recipe: Recipe): number {
  // UI hint only; real VRAM detection is per-model.
  if (recipe.vram_category === 'low') return 4
  if (recipe.vram_category === 'medium') return 8
  if (recipe.vram_category === 'high') return 12
  // Conservative default.
  return recipe.type === 'ensemble' ? 10 : 8
}

export function recipeToPreset(recipe: Recipe): Preset {
  const requiredModels = getRecipeRequiredModels(recipe)
  const stems = recipeDisplayStems(recipe)

  const defaults = recipe.defaults
  const advancedDefaults = defaults
    ? {
        overlap: defaults.overlap,
        segmentSize: defaults.segment_size ?? defaults.chunk_size,
        shifts: defaults.shifts,
        tta: defaults.tta,
      }
    : undefined

  return {
    id: recipe.id,
    name: recipe.name,
    description: recipe.description || '',
    stems,
    recommended: false,
    category: 'smart',
    qualityLevel: recipeQuality(recipe),
    estimatedVram: recipeEstimatedVram(recipe),
    tags: ['recipe', recipe.type],
    // Important: execute as a recipe by passing the recipe id as modelId.
    modelId: recipe.id,
    isRecipe: true,
    recipe: {
      type: recipe.type,
      target: recipe.target,
      warning: recipe.warning,
      source: recipe.source,
      defaults: recipe.defaults,
      requiredModels,
      steps: recipe.steps || [],
    },
    advancedDefaults,
  }
}

export function recipesToPresets(recipes: Recipe[]): Preset[] {
  return recipes.map(recipeToPreset)
}
