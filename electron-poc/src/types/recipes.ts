export type RecipeType = 'ensemble' | 'pipeline' | 'chained' | 'single'

export interface RecipeDefaults {
  overlap?: number
  segment_size?: number
  chunk_size?: number
  shifts?: number
  tta?: boolean
}

export interface RecipePhaseFixParams {
  low_hz?: number
  high_hz?: number
  high_freq_weight?: number
  lowHz?: number
  highHz?: number
  highFreqWeight?: number
}

export interface RecipeEnsembleStep {
  model_id: string
  weight?: number
  role?: string
  note?: string
}

export interface RecipeWorkflowStep {
  model_id?: string
  source_model?: string
  action?: string
  step_name?: string
  name?: string
  step?: number
  input_source?: string
  input_from?: string
  output?: string | string[]
  optional?: boolean
  weight?: number
  note?: string
  [key: string]: unknown
}

export type RecipeStep = RecipeEnsembleStep | RecipeWorkflowStep

export interface Recipe {
  id: string
  name: string
  description?: string
  type: RecipeType
  target?: string
  warning?: string
  source?: string
  vram_category?: 'low' | 'medium' | 'high'
  defaults?: RecipeDefaults

  // Ensemble-only fields (may be present on other types as metadata)
  algorithm?: string
  algorithm_config?: Record<string, unknown>
  phase_fix_params?: RecipePhaseFixParams

  // Post-processing in recipes.json is a high-level name (not always a model dependency)
  post_processing?: string
  post_processing_params?: Record<string, unknown>

  steps: RecipeStep[]
}
