export interface PreflightRequiredModel {
  id: string
  name?: string | null
  installed?: boolean
  curated?: boolean | null
  support_tier?: string | null
  guide_rank?: number | null
  catalog_status?: string | null
  metrics_status?: string | null
  readiness?: string | null
  simple_allowed?: boolean | null
  blocking_reason?: string | null
  blocked_reason?: string | null
  runtime_engine?: string | null
  runtime_model_type?: string | null
  runtime_adapter?: string | null
  runtime_allowed?: string[]
  runtime_preferred?: string | null
  runtime_variant?: string | null
  runtime_required?: string[]
  runtime_fallbacks?: string[]
  runtime_hosts?: string[]
  runtime_config_ref?: string | null
  runtime_checkpoint_ref?: string | null
  runtime_patch_profile?: string | null
  install_burden?: string | null
  requires_manual_assets?: boolean
  missing_assets?: string[]
  required_files?: string[]
  requires_patch?: boolean
  requires_custom_repo_file?: string[]
  workflow_roles?: string[]
  best_for?: string[]
  artifacts_risk?: string[] | null
  vram_profile?: string | null
  chunk_overlap_policy?: Record<string, unknown> | null
  quality_axes?: Record<string, unknown> | null
  content_fit?: string[]
  operating_profiles?: Record<string, unknown> | null
  quality_tier?: string | null
  target_roles?: string[]
  vram_required?: number | null
}

export interface SeparationPreflightPlan {
  workflow_name?: string | null
  workflow_type?: string | null
  workflow_family?: string | null
  quality_goal?: string | null
  difficulty?: string | null
  surface_policy?: string | null
  requires_verified_assets?: boolean
  requires_qa_pass?: boolean
  golden_set_id?: string | null
  audio_quality_thresholds?: {
    min_correlation?: number
    min_snr_db?: number
    min_si_sdr_db?: number
    max_gain_delta_db?: number
    max_clipped_samples?: number
  } | null
  simple_surface?: boolean
  simple_goal?: string | null
  guide_topics?: string[]
  quality_tradeoff?: string | null
  surface_blockers?: string[]
  guide_rank?: number | null
  expected_vram_tier?: string | null
  expected_runtime_tier?: string | null
  effective_model_id?: string | null
  effective_model_ids?: string[]
  required_models?: PreflightRequiredModel[]
  runtime_blocks?: string[]
  recommended_adjustments?: string[]
  estimated_vram_gb?: number | null
  resolved_device?: string | null
  resolved_overlap?: number | null
  resolved_segment_size?: number | null
  runtime_adapter?: string | null
  missing_runtime_assets?: string[]
  unsupported_patch_profile?: string[]
  phase_fix_compatibility?: string | null
  crossover_validation?: string | null
  recommended_operating_profile?: string | null
  fallback_reason?: string | null
  should_use_simple?: boolean
  should_use_advanced?: boolean
}

export interface SeparationPreflightResolvedRecipe {
  id?: string
  name?: string
  type?: string
  defaults?: Record<string, unknown>
  steps?: Array<Record<string, unknown>>
  [key: string]: unknown
}

export interface SeparationPreflightReport {
  can_proceed?: boolean
  errors?: string[]
  warnings?: string[]
  missing_models?: string[]
  torch_available?: boolean
  plan?: SeparationPreflightPlan
  resolved?: {
    model_id?: string | null
    recipe?: SeparationPreflightResolvedRecipe | null
    stems?: string[]
    device?: string | null
    overlap?: number | null
    segment_size?: number | null
    batch_size?: number | null
    [key: string]: unknown
  }
  [key: string]: unknown
}
