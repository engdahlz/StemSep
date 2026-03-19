export type RecipeType = 'ensemble' | 'pipeline' | 'chained' | 'single'
export type RecipeDifficulty = 'simple' | 'advanced' | 'expert'
export type RecipeVramTier = 'cpu_only' | 'low_vram' | 'mid_vram' | 'high_vram'
export type RecipeRuntimeTier = 'fast' | 'standard' | 'advanced'

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

export interface RecipeAudioQualityThresholds {
  min_correlation?: number
  min_snr_db?: number
  min_si_sdr_db?: number
  max_gain_delta_db?: number
  max_clipped_samples?: number
}

export interface Recipe {
  id: string
  name: string
  description?: string
  type: RecipeType
  surface?: 'single' | 'ensemble' | 'workflow' | 'restoration' | 'special_stem'
  family?: string
  target?: string
  quality_goal?: string
  difficulty?: RecipeDifficulty
  expected_vram_tier?: RecipeVramTier
  expected_runtime_tier?: RecipeRuntimeTier
  guide_rank?: number
  simple_surface?: boolean
  surface_policy?: 'verified_only' | 'advanced_only' | 'manual_review'
  requires_verified_assets?: boolean
  requires_qa_pass?: boolean
  golden_set_id?: string
  audio_quality_thresholds?: RecipeAudioQualityThresholds
  simple_goal?: string
  guide_topics?: string[]
  quality_tradeoff?: string
  recommended_for?: string[]
  contraindications?: string[]
  workflow_summary?: string
  warning?: string
  source?: string
  promotion_status?: 'curated' | 'supported_advanced'
  qa_status?: 'verified' | 'pending' | 'experimental'
  vram_category?: 'low' | 'medium' | 'high'
  operating_profile?: string
  intermediate_outputs?: string[]
  surface_blockers?: string[]
  required_model_statuses?: Array<{
    id: string
    catalog_status?: string | null
    metrics_status?: string | null
    readiness?: string | null
    simple_allowed?: boolean | null
    blocked_reason?: string | null
    runtime_adapter?: string | null
    install_mode?: string | null
    download_mode?: string | null
  }>
  fallback_policy?: {
    mode?: 'none' | 'runtime_fallback' | 'workflow_fallback' | 'profile_fallback'
    reason?: string
    runtime_order?: string[]
    fallback_workflow_id?: string
    fallback_operating_profile?: string
  }
  defaults?: RecipeDefaults

  // Ensemble-only fields (may be present on other types as metadata)
  algorithm?: string
  algorithm_config?: Record<string, unknown>
  phase_fix_params?: RecipePhaseFixParams
  runtime_policy?: {
    required?: string[]
    fallbacks?: string[]
    allow_manual_models?: boolean
    preferred_runtime?: string
  }
  export_policy?: {
    stems?: string[]
    output_format?: 'wav' | 'mp3' | 'flac'
  }

  // Post-processing in recipes.json is a high-level name (not always a model dependency)
  post_processing?: string
  post_processing_params?: Record<string, unknown>

  steps: RecipeStep[]
}
