export interface PreflightRequiredModel {
  id: string
  name?: string | null
  installed?: boolean
  guide_rank?: number | null
  readiness?: string | null
  simple_allowed?: boolean | null
  blocking_reason?: string | null
  runtime_variant?: string | null
  quality_tier?: string | null
  target_roles?: string[]
  vram_required?: number | null
}

export interface SeparationPreflightPlan {
  workflow_name?: string | null
  workflow_type?: string | null
  quality_goal?: string | null
  difficulty?: string | null
  simple_surface?: boolean
  simple_goal?: string | null
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
