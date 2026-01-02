import type { Preset } from '@/presets'

export type SimpleGoal =
  | 'all'
  | 'instrumental'
  | 'vocals'
  | 'karaoke'
  | 'instruments'
  | 'cleanup'
  | 'workflows'

export type SimpleMode = 'all' | Preset['qualityLevel']

function isRecipePreset(preset: Preset): boolean {
  return preset.isRecipe === true
}

function isMultiStepRecipe(preset: Preset): boolean {
  if (!isRecipePreset(preset)) return false
  const type = preset.recipe?.type
  return type === 'pipeline' || type === 'chained'
}

function lowercased(values: string[] | undefined): string[] {
  return (values || []).map(v => v.toLowerCase())
}

export function presetQualityScore(preset: Preset): 1 | 2 | 3 | 4 | 5 {
  switch (preset.qualityLevel) {
    case 'ultra':
      return 5
    case 'quality':
      return 4
    case 'balanced':
      return 3
    case 'fast':
    default:
      return 2
  }
}

export function presetSpeedScore(preset: Preset): 1 | 2 | 3 | 4 | 5 {
  // Inverse-ish of quality; workflows tend to be slower.
  const base: number =
    preset.qualityLevel === 'fast'
      ? 5
      : preset.qualityLevel === 'balanced'
        ? 4
        : preset.qualityLevel === 'quality'
          ? 3
          : 2

  const workflowPenalty = isMultiStepRecipe(preset) ? 1 : 0
  return Math.max(1, Math.min(5, base - workflowPenalty)) as 1 | 2 | 3 | 4 | 5
}

export function presetIsWorkflow(preset: Preset): boolean {
  return isRecipePreset(preset)
}

export function presetGoal(preset: Preset): Exclude<SimpleGoal, 'all'> {
  // Workflows
  if (presetIsWorkflow(preset)) return 'workflows'

  // Karaoke is a goal by itself
  const tags = lowercased(preset.tags)
  const stems = lowercased(preset.stems)

  const looksLikeKaraoke =
    tags.includes('karaoke') ||
    stems.includes('no_vocals') ||
    stems.includes('backing_vocals') ||
    stems.includes('lead_vocal')

  if (looksLikeKaraoke) return 'karaoke'

  // Cleanup utilities
  const looksLikeCleanup =
    preset.category === 'utility' &&
    (tags.includes('de-reverb') || tags.includes('de-noise') || tags.includes('de-bleed') || preset.id.startsWith('de_'))

  if (looksLikeCleanup) return 'cleanup'

  if (preset.category === 'vocals') return 'vocals'
  if (preset.category === 'instrumental') return 'instrumental'
  if (preset.category === 'instruments') return 'instruments'

  // Fallback: treat remaining utility as cleanup.
  if (preset.category === 'utility') return 'cleanup'

  return 'instrumental'
}

export function presetMatchesGoal(preset: Preset, goal: SimpleGoal): boolean {
  if (goal === 'all') return true
  return presetGoal(preset) === goal
}

export function presetMatchesMode(preset: Preset, mode: SimpleMode): boolean {
  if (mode === 'all') return true
  return preset.qualityLevel === mode
}

export function presetSimpleBadges(preset: Preset): string[] {
  const badges: string[] = []
  if (preset.recommended) badges.push('Recommended')

  if (presetIsWorkflow(preset)) badges.push('Multi-step')

  const tags = (preset.tags || []).map(t => t.toLowerCase())
  if (tags.includes('guide-favorite') || tags.includes('best')) badges.push('Best')
  if (tags.includes('bleedless') || tags.includes('clean')) badges.push('Clean')

  return Array.from(new Set(badges))
}

export function presetIsRecommendedForSimple(preset: Preset): boolean {
  if (preset.recommended) return true
  const id = preset.id.toLowerCase()
  if (id.startsWith('golden_')) return true
  if (id.startsWith('mode_quality')) return true
  if (id.startsWith('best_')) return true
  // Good default: show workflows less aggressively unless explicitly special.
  return false
}

export function presetSortScore(preset: Preset): number {
  // Primary sorting: recommended first, then quality, then lower VRAM.
  const rec = presetIsRecommendedForSimple(preset) ? 1000 : 0
  const q = presetQualityScore(preset) * 100
  const vram = typeof preset.estimatedVram === 'number' ? Math.max(0, 20 - preset.estimatedVram) : 0
  return rec + q + vram
}
