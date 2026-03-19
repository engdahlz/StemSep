import type {
  RecipeAudioQualityThresholds,
  RecipeDefaults,
  RecipeDifficulty,
  RecipeRuntimeTier,
  RecipeStep,
  RecipeType,
  RecipeVramTier,
} from './types/recipes'

export interface PostProcessingStep {
  type: 'phase_fix' | 'de_reverb' | 'de_bleed' | 'de_noise' | 'de_breath'
  modelId: string
  description: string
  targetStem?: 'vocals' | 'instrumental' | 'all'
}

export interface Preset {
  id: string
  name: string
  description: string
  stems: string[]
  recommended: boolean
  category: 'instrumental' | 'vocals' | 'instruments' | 'utility' | 'ensemble' | 'smart'
  qualityLevel: 'fast' | 'balanced' | 'quality' | 'ultra'
  estimatedVram: number // GB
  tags: string[]
  simpleVisible?: boolean
  simpleGoal?: 'instrumental' | 'vocals' | 'karaoke' | 'cleanup' | 'instruments'
  guideRank?: number
  difficulty?: RecipeDifficulty
  expectedVramTier?: RecipeVramTier
  expectedRuntimeTier?: RecipeRuntimeTier
  recommendedFor?: string[]
  contraindications?: string[]
  workflowSummary?: string
  workflowFamily?: string
  promotionStatus?: 'curated' | 'supported_advanced'
  qaStatus?: 'verified' | 'pending' | 'experimental'
  modelId?: string // For single-model presets
  isRecipe?: boolean // For workflow recipes
  recipe?: {
    type: RecipeType
    surface?: 'single' | 'ensemble' | 'workflow' | 'restoration' | 'special_stem'
    target?: string
    warning?: string
    source?: string
    family?: string
    surface_policy?: 'verified_only' | 'advanced_only' | 'manual_review'
    requires_verified_assets?: boolean
    requires_qa_pass?: boolean
    golden_set_id?: string
    audio_quality_thresholds?: RecipeAudioQualityThresholds
    guide_topics?: string[]
    quality_tradeoff?: string
    promotion_status?: 'curated' | 'supported_advanced'
    qa_status?: 'verified' | 'pending' | 'experimental'
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
    defaults?: RecipeDefaults
    difficulty?: RecipeDifficulty
    expectedVramTier?: RecipeVramTier
    expectedRuntimeTier?: RecipeRuntimeTier
    guideRank?: number
    recommendedFor?: string[]
    contraindications?: string[]
    workflowSummary?: string
    runtime_policy?: {
      required?: string[]
      fallbacks?: string[]
      allow_manual_models?: boolean
      preferred_runtime?: string
    }
    export_policy?: {
      stems?: string[]
      output_format?: 'wav' | 'mp3' | 'flac'
      intermediate_outputs?: string[]
    }
    operating_profile?: string
    intermediate_outputs?: string[]
    fallback_policy?: {
      mode?: 'none' | 'runtime_fallback' | 'workflow_fallback' | 'profile_fallback'
      reason?: string
      runtime_order?: string[]
      fallback_workflow_id?: string
      fallback_operating_profile?: string
    }
    requiredModels: string[]
    steps: RecipeStep[]
  }
  ensembleConfig?: {
    models: { model_id: string; weight?: number }[]
    algorithm: 'average' | 'avg_wave' | 'max_spec' | 'min_spec' | 'phase_fix' | 'frequency_split'
    stemAlgorithms?: {
      vocals?: 'average' | 'max_spec' | 'min_spec'
      instrumental?: 'average' | 'max_spec' | 'min_spec'
    }
    phaseFixEnabled?: boolean
    phaseFixParams?: {
      lowHz: number
      highHz: number
      highFreqWeight: number
    }
  }
  pipelineNote?: string // Notes about post-processing pipeline
  postProcessingSteps?: PostProcessingStep[] // Auto-applicable post-processing
  advancedDefaults?: {
    overlap?: number
    segmentSize?: number
    shifts?: number
    tta?: boolean
    bitrate?: string
  }
}

export const ALL_PRESETS: Preset[] = [
  {
    id: 'best_instrumental',
    name: 'Best Instrumental (Guide Pick)',
    description: 'Guide-oriented ensemble for high-quality instrumentals.',
    stems: ['instrumental', 'vocals'],
    recommended: true,
    category: 'instrumental',
    qualityLevel: 'ultra',
    estimatedVram: 14,
    tags: ['ensemble', 'best', 'guide-favorite'],
    ensembleConfig: {
      models: [
        { model_id: 'unwa-hyperace', weight: 1.0 },
        { model_id: 'bs-roformer-viperx-1297', weight: 1.0 },
      ],
      algorithm: 'max_spec',
    },
    advancedDefaults: { overlap: 4 },
  },

  {
    id: 'best_vocals',
    name: 'Best Vocals (Clean)',
    description: 'Clean vocals with minimal bleed. Great for production workflows.',
    stems: ['vocals', 'instrumental'],
    recommended: true,
    category: 'vocals',
    qualityLevel: 'quality',
    estimatedVram: 6,
    tags: ['clean', 'bleedless', 'best'],
    modelId: 'bs-roformer-viperx-1297',
    advancedDefaults: { overlap: 4 },
  },

  {
    id: 'best_karaoke',
    name: 'Best Karaoke',
    description: 'High-quality vocal removal for karaoke / backing tracks.',
    stems: ['no_vocals', 'vocals'],
    recommended: true,
    category: 'utility',
    qualityLevel: 'quality',
    estimatedVram: 12,
    tags: ['karaoke', 'no-vocals', 'backing-track'],
    ensembleConfig: {
      models: [
        { model_id: 'mel-band-karaoke-becruily', weight: 1.0 },
        { model_id: 'bs-roformer-karaoke-becruily', weight: 1.0 },
      ],
      algorithm: 'max_spec',
    },
    advancedDefaults: { overlap: 4 },
  },

  {
    id: 'fast_preview',
    name: 'Fast Preview (MDX23C)',
    description: 'Fast separation preview for quick checking before running heavier presets.',
    stems: ['vocals', 'instrumental'],
    recommended: false,
    category: 'instrumental',
    qualityLevel: 'fast',
    estimatedVram: 3,
    tags: ['fast', 'preview'],
    modelId: 'mdx23c-instvoc-hq',
    advancedDefaults: { overlap: 2 },
  },

  {
    id: 'high_quality_instrumental',
    name: 'High Quality Instrumental (Fullness)',
    description: 'High-fullness instrumental. If you hear buzzing, use a Phase Fix workflow preset.',
    stems: ['instrumental', 'vocals'],
    recommended: true,
    category: 'instrumental',
    qualityLevel: 'quality',
    estimatedVram: 8,
    tags: ['fullness', 'detail'],
    modelId: 'unwa-inst-v1e-plus',
    advancedDefaults: { overlap: 4 },
  },

  {
    id: 'balanced_instrumental',
    name: 'Balanced Instrumental (Bleedless)',
    description: 'Cleaner/bleedless instrumental option with lower noise.',
    stems: ['instrumental', 'vocals'],
    recommended: false,
    category: 'instrumental',
    qualityLevel: 'balanced',
    estimatedVram: 6,
    tags: ['bleedless', 'balanced'],
    modelId: 'gabox-inst-fv7z',
    advancedDefaults: { overlap: 4 },
  },

  {
    id: 'cleanest_instrumental_maxmin',
    name: 'Cleanest Instrumental (Max/Min)',
    description: 'Per-stem blending to reduce vocal bleed while keeping detail.',
    stems: ['instrumental', 'vocals'],
    recommended: true,
    category: 'instrumental',
    qualityLevel: 'quality',
    estimatedVram: 10,
    tags: ['ensemble', 'max-min', 'clean'],
    ensembleConfig: {
      models: [
        { model_id: 'bs-roformer-viperx-1297', weight: 1.0 },
        { model_id: 'mel-band-roformer-kim', weight: 1.0 },
      ],
      algorithm: 'average',
      stemAlgorithms: {
        vocals: 'max_spec',
        instrumental: 'min_spec',
      },
    },
    advancedDefaults: { overlap: 4 },
  },

  {
    id: 'natural_body_vocals',
    name: 'Natural Body Vocals',
    description: 'Preserves warmth/body in vocals. Some bleed tolerated for richness.',
    stems: ['vocals', 'instrumental'],
    recommended: false,
    category: 'vocals',
    qualityLevel: 'quality',
    estimatedVram: 8,
    tags: ['fullness', 'warm'],
    modelId: 'unwa-revive-3e',
    advancedDefaults: { overlap: 4 },
  },

  {
    id: 'resurrection_vocals',
    name: 'Resurrection Vocals (Backing/Harmonies)',
    description: 'Good at capturing backing vocals and harmonies.',
    stems: ['vocals', 'instrumental'],
    recommended: false,
    category: 'vocals',
    qualityLevel: 'quality',
    estimatedVram: 6,
    tags: ['harmonies', 'backing-vocals'],
    modelId: 'unwa-resurrection-voc',
    advancedDefaults: { overlap: 4 },
  },

  {
    id: 'de_reverb_room',
    name: 'De-Reverb (Room)',
    description: 'Remove room reverb. Best applied to separated vocals, not full mix.',
    stems: ['dry', 'reverb'],
    recommended: false,
    category: 'utility',
    qualityLevel: 'quality',
    estimatedVram: 6,
    tags: ['de-reverb', 'cleanup'],
    modelId: 'anvuew-dereverb-room',
    advancedDefaults: { overlap: 4 },
  },

  {
    id: 'de_noise',
    name: 'De-Noise',
    description: 'Reduce background noise and hiss.',
    stems: ['clean', 'noise'],
    recommended: false,
    category: 'utility',
    qualityLevel: 'balanced',
    estimatedVram: 6,
    tags: ['de-noise', 'cleanup'],
    modelId: 'aufr33-denoise-std',
    advancedDefaults: { overlap: 4 },
  },

  {
    id: 'de_noise_aggressive',
    name: 'De-Noise (Aggressive)',
    description: 'Stronger noise suppression. Can dull transients.',
    stems: ['clean', 'noise'],
    recommended: false,
    category: 'utility',
    qualityLevel: 'quality',
    estimatedVram: 6,
    tags: ['de-noise', 'cleanup', 'aggressive'],
    modelId: 'aufr33-denoise-aggressive',
    advancedDefaults: { overlap: 4 },
  },

  {
    id: 'de_crowd',
    name: 'De-Crowd (Audience Removal)',
    description: 'Targets crowd/ambience noise. Useful on live recordings.',
    stems: ['clean', 'crowd'],
    recommended: false,
    category: 'utility',
    qualityLevel: 'quality',
    estimatedVram: 6,
    tags: ['crowd', 'live', 'cleanup'],
    modelId: 'mel-band-crowd',
    advancedDefaults: { overlap: 4 },
  },

  {
    id: 'de_bleed',
    name: 'De-Bleed (Vocal Residue)',
    description: 'Remove vocal bleed from instrumentals.',
    stems: ['clean', 'bleed'],
    recommended: false,
    category: 'utility',
    qualityLevel: 'quality',
    estimatedVram: 6,
    tags: ['de-bleed', 'cleanup'],
    modelId: 'gabox-denoise-debleed',
    advancedDefaults: { overlap: 4 },
  },

  {
    id: 'drumsep_5stem',
    name: 'DrumSep 5-Stem',
    description: 'Split drums into Kick/Snare/Toms/Hi-Hat/Cymbals.',
    stems: ['kick', 'snare', 'toms', 'hihat', 'cymbals'],
    recommended: true,
    category: 'instruments',
    qualityLevel: 'quality',
    estimatedVram: 6,
    tags: ['drums', 'drumsep'],
    modelId: 'mel-roformer-drumsep-5stem',
    advancedDefaults: { overlap: 4 },
  },

  {
    id: 'guitar_isolation',
    name: 'Guitar Isolation',
    description: 'Extract guitar from the mix.',
    stems: ['guitar', 'other'],
    recommended: false,
    category: 'instruments',
    qualityLevel: 'quality',
    estimatedVram: 6,
    tags: ['guitar', 'isolation'],
    modelId: 'becruily-guitar',
    advancedDefaults: { overlap: 4 },
  },
]

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

export function getPresetsByCategory(category: Preset['category']): Preset[] {
  return ALL_PRESETS.filter(p => p.category === category)
}

export function getRecommendedPresets(): Preset[] {
  return ALL_PRESETS.filter(p => p.recommended)
}

export function getPresetById(id: string): Preset | undefined {
  return ALL_PRESETS.find(p => p.id === id)
}

export function getPresetsByTag(tag: string): Preset[] {
  return ALL_PRESETS.filter(p => p.tags.includes(tag))
}

export function getLowVramPresets(maxVram: number = 6): Preset[] {
  return ALL_PRESETS.filter(p => p.estimatedVram <= maxVram)
}

/**
 * Get all model IDs required for a preset to function.
 * Includes main model, ensemble models, and post-processing models.
 */
export function getRequiredModels(preset: Preset): string[] {
  const models: Set<string> = new Set()

  // Main model
  if (preset.modelId && !preset.isRecipe) {
    models.add(preset.modelId)
  }

  // Recipe step models (recipes execute via modelId=recipe.id, but dependencies are the step models)
  if (preset.isRecipe && preset.recipe?.requiredModels) {
    preset.recipe.requiredModels.forEach(m => models.add(m))
  }

  // Ensemble models
  if (preset.ensembleConfig?.models) {
    preset.ensembleConfig.models.forEach(m => {
      models.add(m.model_id)
    })
  }

  // Post-processing models
  if (preset.postProcessingSteps) {
    preset.postProcessingSteps.forEach(step => {
      if (step.modelId) {
        models.add(step.modelId)
      }
    })
  }

  return Array.from(models)
}
