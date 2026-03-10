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
  if (recipe.expected_runtime_tier === 'fast') return 'fast'
  if (recipe.difficulty === 'expert') return 'ultra'
  if (recipe.difficulty === 'advanced') return 'quality'
  if (recipe.difficulty === 'simple' && recipe.guide_rank && recipe.guide_rank <= 1) {
    return 'quality'
  }
  const id = recipe.id.toLowerCase()
  if (id.startsWith('mode_quick')) return 'fast'
  if (id.startsWith('mode_quality')) return 'quality'
  if (id.startsWith('golden_') || recipe.name.includes('🏆')) return 'ultra'
  if (recipe.type === 'pipeline' || recipe.type === 'chained') return 'quality'
  return 'balanced'
}

function recipeEstimatedVram(recipe: Recipe): number {
  if (recipe.expected_vram_tier === 'cpu_only') return 0
  if (recipe.expected_vram_tier === 'low_vram') return 4
  if (recipe.expected_vram_tier === 'mid_vram') return 8
  if (recipe.expected_vram_tier === 'high_vram') return 12
  // UI hint only; real VRAM detection is per-model.
  if (recipe.vram_category === 'low') return 4
  if (recipe.vram_category === 'medium') return 8
  if (recipe.vram_category === 'high') return 12
  // Conservative default.
  return recipe.type === 'ensemble' ? 10 : 8
}

function recipeCategory(recipe: Recipe): Preset['category'] {
  const target = (recipe.target || '').toLowerCase()
  if (target === 'vocals') return 'vocals'
  if (target === 'instrumental') return 'instrumental'
  if (target === 'karaoke') return 'utility'
  if (target === 'restoration') return 'utility'
  if (target === 'drums' || target === 'bass') return 'instruments'
  return 'smart'
}

function recipeSimpleGoal(recipe: Recipe): Preset['simpleGoal'] | undefined {
  const explicit = recipe.simple_goal
  if (
    explicit === 'instrumental' ||
    explicit === 'vocals' ||
    explicit === 'karaoke' ||
    explicit === 'cleanup' ||
    explicit === 'instruments'
  ) {
    return explicit
  }
  const target = (recipe.target || '').toLowerCase()
  if (target === 'vocals') return 'vocals'
  if (target === 'instrumental') return 'instrumental'
  if (target === 'karaoke') return 'karaoke'
  if (target === 'restoration') return 'cleanup'
  if (target === 'drums' || target === 'bass') return 'instruments'
  return undefined
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

  const simpleGoal = recipeSimpleGoal(recipe)
  const baseTags = ['recipe', recipe.type]
  if (recipe.target) baseTags.push(recipe.target)
  if (recipe.quality_goal) baseTags.push(recipe.quality_goal)
  if (recipe.difficulty) baseTags.push(recipe.difficulty)
  if (recipe.simple_surface) baseTags.push('simple-surface')

  return {
    id: recipe.id,
    name: recipe.name,
    description: recipe.description || '',
    stems,
    recommended:
      recipe.promotion_status !== 'supported_advanced' &&
      (recipe.simple_surface === true || (typeof recipe.guide_rank === 'number' && recipe.guide_rank <= 2)),
    category: recipeCategory(recipe),
    qualityLevel: recipeQuality(recipe),
    estimatedVram: recipeEstimatedVram(recipe),
    tags: Array.from(new Set(baseTags)),
    simpleVisible:
      recipe.simple_surface === true &&
      recipe.promotion_status !== 'supported_advanced' &&
      recipe.qa_status !== 'pending',
    simpleGoal,
    guideRank: recipe.guide_rank,
    difficulty: recipe.difficulty,
    expectedVramTier: recipe.expected_vram_tier,
    expectedRuntimeTier: recipe.expected_runtime_tier,
    recommendedFor: recipe.recommended_for,
    contraindications: recipe.contraindications,
    workflowSummary: recipe.workflow_summary,
    workflowFamily: recipe.family,
    promotionStatus: recipe.promotion_status,
    qaStatus: recipe.qa_status,
    // Important: execute as a recipe by passing the recipe id as modelId.
    modelId: recipe.id,
    isRecipe: true,
    recipe: {
      type: recipe.type,
      surface: recipe.surface,
      family: recipe.family,
      target: recipe.target,
      warning: recipe.warning,
      source: recipe.source,
      promotion_status: recipe.promotion_status,
      qa_status: recipe.qa_status,
      defaults: recipe.defaults,
      difficulty: recipe.difficulty,
      expectedVramTier: recipe.expected_vram_tier,
      expectedRuntimeTier: recipe.expected_runtime_tier,
      guideRank: recipe.guide_rank,
      recommendedFor: recipe.recommended_for,
      contraindications: recipe.contraindications,
      workflowSummary: recipe.workflow_summary,
      runtime_policy: recipe.runtime_policy,
      export_policy: recipe.export_policy,
      operating_profile: recipe.operating_profile,
      intermediate_outputs: recipe.intermediate_outputs,
      fallback_policy: recipe.fallback_policy,
      requiredModels,
      steps: recipe.steps || [],
    },
    advancedDefaults,
  }
}

export function recipesToPresets(recipes: Recipe[]): Preset[] {
  return recipes.map(recipeToPreset)
}
