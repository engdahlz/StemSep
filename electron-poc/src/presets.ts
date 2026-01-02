import type { RecipeDefaults, RecipeStep, RecipeType } from './types/recipes'

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
  modelId?: string // For single-model presets
  isRecipe?: boolean // For workflow recipes
  recipe?: {
    type: RecipeType
    target?: string
    warning?: string
    source?: string
    defaults?: RecipeDefaults
    requiredModels: string[]
    steps: RecipeStep[]
  }
  ensembleConfig?: {
    models: { model_id: string; weight?: number }[]
    algorithm: 'average' | 'avg_wave' | 'max_spec' | 'min_spec' | 'phase_fix'
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
}

// ============================================================================
// EXPERT PRESETS - Based on NotebookLM/MSST/UVR community recommendations
// ============================================================================

export const ALL_PRESETS: Preset[] = [
  // ============================================================================
  // TOP ENSEMBLE - Based on official guide recommendation
  // ============================================================================

  // Defaults referenced by SeparatePage favorites/selection
  {
    id: 'best_instrumental',
    name: 'Best Instrumental (Guide Pick)',
    description: 'Top guide pick: HyperACE + Viperx 1297 with Max Spec blending.',
    stems: ['instrumental', 'vocals'],
    recommended: true,
    category: 'instrumental',
    qualityLevel: 'ultra',
    estimatedVram: 14,
    tags: ['ensemble', 'best', 'guide-favorite'],
    ensembleConfig: {
      models: [
        { model_id: 'unwa-hyperace', weight: 1.0 },
        { model_id: 'bs-roformer-viperx-1297', weight: 1.0 }
      ],
      algorithm: 'max_spec'
    },
    pipelineNote: 'Guide favorite ensemble - best balance of fullness and clarity'
  },

  {
    id: 'best_vocals',
    name: 'Best Vocals (Clean SDR)',
    description: 'Cleanest vocals with minimal instrument bleed (great for RVC / production).',
    stems: ['vocals', 'instrumental'],
    recommended: true,
    category: 'vocals',
    qualityLevel: 'quality',
    estimatedVram: 6,
    tags: ['bleedless', 'clean', 'best'],
    modelId: 'bs-roformer-viperx-1297'
  },

  // HyperACE + Viperx 1297 - Guide's "My favorite ensemble right now" (dca100fb8)
  {
    id: 'hyperace_ultimate',
    name: 'HyperACE Ultimate (Top Rated)',
    description: 'Guide\'s favorite ensemble: HyperACE + Viperx 1297 with Max Spec.',
    stems: ['instrumental', 'vocals'],
    recommended: true,
    category: 'instrumental',
    qualityLevel: 'ultra',
    estimatedVram: 14,
    tags: ['ensemble', 'best', 'guide-favorite', 'hyperace'],
    ensembleConfig: {
      models: [
        { model_id: 'unwa-hyperace', weight: 1.0 },
        { model_id: 'bs-roformer-viperx-1297', weight: 1.0 }
      ],
      algorithm: 'max_spec'
    },
    pipelineNote: 'Official guide recommendation - best balance of fullness and clarity'
  },

  // ============================================================================
  // VOCALS PRESETS (5)
  // ============================================================================

  // Best bleedless vocals - Viperx 1297 for clean SDR
  {
    id: 'studio_clean_vocals',
    name: 'Studio Clean Vocals',
    description: 'Maximum purity with minimal instrument bleed. Best for RVC training.',
    stems: ['vocals', 'instrumental'],
    recommended: true,
    category: 'vocals',
    qualityLevel: 'quality',
    estimatedVram: 6,
    tags: ['bleedless', 'clean', 'rvc', 'production'],
    modelId: 'bs-roformer-viperx-1297'
  },

  // Best fullness vocals - Unwa Revive 3e has Fullness: 21.43
  {
    id: 'natural_body_vocals',
    name: 'Natural Body Vocals',
    description: 'Preserves warmth and body in voice. Some bleed tolerated for richness.',
    stems: ['vocals', 'instrumental'],
    recommended: false,
    category: 'vocals',
    qualityLevel: 'quality',
    estimatedVram: 8,
    tags: ['fullness', 'warm', 'natural', 'listening'],
    modelId: 'unwa-revive-3e'
  },

  // Resurrection vocal - good backing vocal capture
  {
    id: 'resurrection_vocals',
    name: 'Resurrection Vocals',
    description: 'Good at capturing backing vocals and harmonies.',
    stems: ['vocals', 'instrumental'],
    recommended: false,
    category: 'vocals',
    qualityLevel: 'quality',
    estimatedVram: 6,
    tags: ['backing-vocals', 'harmonies', 'resurrection'],
    modelId: 'unwa-resurrection-voc'
  },

  // Ultimate vocals ensemble - Viperx + Resurrection
  {
    id: 'ultimate_vocals',
    name: 'Ultimate Studio Vocals',
    description: 'Ensemble: Viperx 1297 (clean) + Resurrection (warmth).',
    stems: ['vocals', 'instrumental'],
    recommended: true,
    category: 'vocals',
    qualityLevel: 'ultra',
    estimatedVram: 12,
    tags: ['ensemble', 'best', 'studio', 'production'],
    ensembleConfig: {
      models: [
        { model_id: 'bs-roformer-viperx-1297', weight: 1.0 },  // SDR champion
        { model_id: 'unwa-resurrection-voc', weight: 0.8 }      // Backing vocals
      ],
      algorithm: 'max_spec'
    },
    pipelineNote: 'Follow with Anvuew De-Reverb for dry studio acapella'
  },

  // Karaoke / Lead-Back separator - Anvuew karaoke
  {
    id: 'backing_vocals',
    name: 'Lead + Backing Vocals (Karaoke)',
    description: 'Separates lead vocals from backing vocals and harmonies.',
    stems: ['lead_vocal', 'backing_vocals'],
    recommended: false,
    category: 'vocals',
    qualityLevel: 'quality',
    estimatedVram: 8,
    tags: ['karaoke', 'harmonies', 'backing', 'lead'],
    modelId: 'anvuew-karaoke'
  },

  // Low VRAM vocals option
  {
    id: 'low_vram_vocals',
    name: 'Low VRAM Vocals',
    description: 'Good quality vocals for 4-6GB GPUs. Use chunk_size 132300.',
    stems: ['vocals', 'instrumental'],
    recommended: false,
    category: 'vocals',
    qualityLevel: 'balanced',
    estimatedVram: 4,
    tags: ['low-vram', 'fast', '4gb', '6gb'],
    modelId: 'mel-band-roformer-kim'
  },

  // ============================================================================
  // INSTRUMENTAL PRESETS
  // ============================================================================

  // Fast preview - Kim model (fast & low resource)
  {
    id: 'fast_instrumental',
    name: 'Fast Instrumental',
    description: 'Quick preview quality. Good for checking if separation works.',
    stems: ['instrumental', 'vocals'],
    recommended: false,
    category: 'instrumental',
    qualityLevel: 'fast',
    estimatedVram: 4,
    tags: ['fast', 'preview', 'quick'],
    modelId: 'mel-band-roformer-kim'
  },

  // Balanced - Gabox Inst Fv7z (best bleedless)
  {
    id: 'balanced_instrumental',
    name: 'Balanced Instrumental',
    description: 'Good balance between fullness and cleanliness. Least noise.',
    stems: ['instrumental', 'vocals'],
    recommended: false,
    category: 'instrumental',
    qualityLevel: 'balanced',
    estimatedVram: 6,
    tags: ['balanced', 'everyday', 'bleedless'],
    modelId: 'gabox-inst-fv7z'
  },

  // Guide mentions newer instrumental models (e.g. HyperACE v2, Inst_GaboxFv9, inst_fv7b),
  // but they are not currently shipped in the in-app registry. Closest shipped alternatives:
  {
    id: 'gabox_inst_fullness',
    name: 'GaBox Instrumental Fullness',
    description: 'Fuller GaBox instrumental option (shipped). Good when you want more body than fv7z.',
    stems: ['instrumental', 'vocals'],
    recommended: false,
    category: 'instrumental',
    qualityLevel: 'quality',
    estimatedVram: 6,
    tags: ['gabox', 'fullness', 'alternative'],
    modelId: 'gabox-inst-v6n'
  },

  // High Quality - Unwa Inst v1e+ (Fullness: 37.89, industry standard)
  {
    id: 'high_quality_instrumental',
    name: 'High Quality Instrumental',
    description: 'Industry standard. Rich detail but may need Phase Fix.',
    stems: ['instrumental', 'vocals'],
    recommended: true,
    category: 'instrumental',
    qualityLevel: 'quality',
    estimatedVram: 8,
    tags: ['high-quality', 'fullness', 'detail', 'industry'],
    modelId: 'unwa-inst-v1e-plus',
    pipelineNote: 'May benefit from Phase Fix with Becruily Vocal as source',
    postProcessingSteps: [
      {
        type: 'phase_fix',
        modelId: 'becruily-vocal',
        description: 'Phase Fix with Becruily Vocal',
        targetStem: 'instrumental'
      }
    ]
  },

  {
    id: 'phase_fix_instrumental',
    name: 'Phase Fix Instrumental (Recommended)',
    description: 'Fixes Roformer buzzing/noise by phase-correcting a high-fullness instrumental using a clean vocal phase reference (500-5000Hz).',
    stems: ['instrumental', 'vocals'],
    recommended: true,
    category: 'instrumental',
    qualityLevel: 'ultra',
    estimatedVram: 12,
    tags: ['phase-fix', 'buzzing', 'guide-recommended', 'ensemble'],
    ensembleConfig: {
      models: [
        { model_id: 'unwa-inst-v1e-plus', weight: 1.0 },
        { model_id: 'becruily-vocal', weight: 1.0 }
      ],
      algorithm: 'average',
      phaseFixEnabled: true,
      phaseFixParams: {
        lowHz: 500,
        highHz: 5000,
        highFreqWeight: 2.0
      }
    }
  },

  {
    id: 'cleanest_instrumental_maxmin',
    name: 'Cleanest Instrumental (Max/Min)',
    description: 'Per-stem blending: Max Spec for vocals extraction and Min Spec for instrumental to minimize vocal bleed while keeping detail.',
    stems: ['instrumental', 'vocals'],
    recommended: true,
    category: 'instrumental',
    qualityLevel: 'quality',
    estimatedVram: 10,
    tags: ['max-min', 'bleedless', 'ensemble', 'guide-recommended'],
    ensembleConfig: {
      models: [
        { model_id: 'bs-roformer-viperx-1297', weight: 1.0 },
        { model_id: 'mel-band-roformer-kim', weight: 1.0 }
      ],
      algorithm: 'average',
      stemAlgorithms: {
        vocals: 'max_spec',
        instrumental: 'min_spec'
      }
    }
  },

  // Ultra Clean - Gabox Inst Fv7z (Bleedless: 44.95)
  {
    id: 'ultra_clean_instrumental',
    name: 'Ultra Clean Instrumental',
    description: 'Maximum bleedless score. Extremely clean from vocal residue.',
    stems: ['instrumental', 'vocals'],
    recommended: false,
    category: 'instrumental',
    qualityLevel: 'quality',
    estimatedVram: 6,
    tags: ['bleedless', 'clean', 'no-vocals'],
    modelId: 'gabox-inst-fv7z'
  },

  // Ultimate instrumental ensemble
  {
    id: 'ultimate_instrumental',
    name: 'Ultimate Instrumental',
    description: 'Ensemble for maximum detail and texture.',
    stems: ['instrumental', 'vocals'],
    recommended: true,
    category: 'instrumental',
    qualityLevel: 'ultra',
    estimatedVram: 14,
    tags: ['ensemble', 'best', 'ultra', 'restoration'],
    ensembleConfig: {
      models: [
        { model_id: 'unwa-inst-v1e-plus', weight: 1.0 }, // Fullness champion
        { model_id: 'becruily-inst', weight: 1.0 }       // Per guide: best preservation
      ],
      algorithm: 'max_spec'
    },
    pipelineNote: 'Per guide: best instrument preservation ensemble'
  },

  // v1e+ Becruily ensemble - per guide recommendation
  {
    id: 'v1e_becruily_ensemble',
    name: 'v1e+ Becruily Ensemble',
    description: 'Per guide: best instrument preservation with Max Spec.',
    stems: ['instrumental', 'vocals'],
    recommended: true,
    category: 'instrumental',
    qualityLevel: 'ultra',
    estimatedVram: 12,
    tags: ['ensemble', 'preservation', 'guide-recommended'],
    ensembleConfig: {
      models: [
        { model_id: 'unwa-inst-v1e-plus', weight: 1.0 },
        { model_id: 'becruily-inst', weight: 1.0 }
      ],
      algorithm: 'max_spec'
    }
  },

  // Gabox bleedless ensemble
  {
    id: 'gabox_bleedless_ensemble',
    name: 'Gabox Bleedless Ensemble',
    description: 'Gabox Fv7z + Viperx 1297. Minimal noise and vocal bleed.',
    stems: ['instrumental', 'vocals'],
    recommended: false,
    category: 'instrumental',
    qualityLevel: 'quality',
    estimatedVram: 10,
    tags: ['ensemble', 'bleedless', 'clean'],
    ensembleConfig: {
      models: [
        { model_id: 'gabox-inst-fv7z', weight: 1.0 },
        { model_id: 'bs-roformer-viperx-1297', weight: 1.0 }
      ],
      algorithm: 'max_spec'
    }
  },

  // ============================================================================
  // UTILITY PRESETS
  // ============================================================================

  // Best Karaoke - ensemble
  {
    id: 'best_karaoke',
    name: 'Best Karaoke',
    description: 'Cleanest backing tracks with all vocals removed.',
    stems: ['no_vocals', 'vocals'],
    recommended: true,
    category: 'utility',
    qualityLevel: 'quality',
    estimatedVram: 12,
    tags: ['karaoke', 'no-vocals', 'backing-track'],
    ensembleConfig: {
      models: [
        { model_id: 'mel-band-karaoke-becruily' },
        { model_id: 'bs-roformer-karaoke-becruily' }
      ],
      algorithm: 'max_spec'
    }
  },

  // De-Reverb
  {
    id: 'de_reverb',
    name: 'De-Reverb (Remove Echo)',
    description: 'Removes room reverb for dry studio sound. Run on vocals only!',
    stems: ['dry', 'reverb'],
    recommended: false,
    category: 'utility',
    qualityLevel: 'quality',
    estimatedVram: 6,
    tags: ['de-reverb', 'cleanup', 'dry', 'studio'],
    modelId: 'mel-roformer-dereverb-anvuew-v2',
    pipelineNote: 'Only run on separated vocals, NOT full mix (destroys drums)'
  },

  // De-Noise
  {
    id: 'de_noise',
    name: 'De-Noise (Remove Background)',
    description: 'Reduces background noise and hiss.',
    stems: ['clean', 'noise'],
    recommended: false,
    category: 'utility',
    qualityLevel: 'balanced',
    estimatedVram: 6,
    tags: ['de-noise', 'cleanup', 'hiss'],
    modelId: 'aufr33-denoise-std'
  },

  {
    id: 'de_noise_aggressive',
    name: 'De-Noise (Aggressive)',
    description: 'Stronger background suppression than standard denoise. Can dull transients.',
    stems: ['clean', 'noise'],
    recommended: false,
    category: 'utility',
    qualityLevel: 'quality',
    estimatedVram: 6,
    tags: ['de-noise', 'cleanup', 'aggressive'],
    modelId: 'aufr33-denoise-aggressive'
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
    modelId: 'mel-band-crowd'
  },

  // De-Bleed (remove vocal residue from instrumental)
  {
    id: 'de_bleed',
    name: 'De-Bleed (Remove Vocal Residue)',
    description: 'Removes vocal bleed from instrumental tracks.',
    stems: ['clean', 'bleed'],
    recommended: false,
    category: 'utility',
    qualityLevel: 'quality',
    estimatedVram: 6,
    tags: ['de-bleed', 'cleanup', 'instrumental'],
    modelId: 'mel-roformer-debleed'
  },

  // ============================================================================
  // INSTRUMENTS PRESETS (3)
  // ============================================================================

  // DrumSep 5-stem
  {
    id: 'drumsep_5stem',
    name: 'DrumSep 5-Stem',
    description: 'Separates drums into Kick, Snare, Toms, Hi-Hat, Cymbals.',
    stems: ['kick', 'snare', 'toms', 'hihat', 'cymbals'],
    recommended: true,
    category: 'instruments',
    qualityLevel: 'quality',
    estimatedVram: 6,
    tags: ['drums', 'drumsep', 'sampling', 'kit'],
    modelId: 'mel-roformer-drumsep-5stem',
    pipelineNote: 'Use 5-10 shifts (TTA) for best transient quality'
  },

  // Full 6-stem separation
  {
    id: 'full_separation_6stem',
    name: 'Full 6-Stem Separation',
    description: 'Separates into Drums, Bass, Vocals, Guitar, Piano, Other.',
    stems: ['drums', 'bass', 'vocals', 'guitar', 'piano', 'other'],
    recommended: false,
    category: 'instruments',
    qualityLevel: 'balanced',
    estimatedVram: 4,
    tags: ['6-stem', 'full', 'demucs', 'all-instruments'],
    modelId: 'htdemucs-6s',
    pipelineNote: 'Use 5-10 shifts for better quality'
  },

  // Guitar isolation
  {
    id: 'guitar_isolation',
    name: 'Guitar Isolation',
    description: 'Extracts guitar from the mix.',
    stems: ['guitar', 'other'],
    recommended: false,
    category: 'instruments',
    qualityLevel: 'quality',
    estimatedVram: 6,
    tags: ['guitar', 'instrument', 'isolation'],
    modelId: 'becruily-guitar'
  }
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
