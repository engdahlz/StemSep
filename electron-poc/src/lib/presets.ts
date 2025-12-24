// =============================================================================
// PRESET TO MODEL MAPPING
// Updated 2024-12-13 based on official guide recommendations and user's models
// Models must match exact folder names in D:\StemSep Models
// =============================================================================

// Single model presets - maps preset ID to model folder name
export const PRESET_TO_MODEL_MAP: Record<string, string> = {
  // Karaoke & Lead/Backing separation
  'vocals_lead_back_ultimate': 'anvuew-karaoke',  // Top karaoke model
  'karaoke_production': 'anvuew-karaoke',

  // Fast vocal separation  
  'vocals_fast': 'mel-band-roformer-kim',  // Fast Kim model
  'low_vram_workflow': 'mel-band-roformer-kim',

  // High-quality vocals
  'vocal_clean': 'bs-roformer-viperx-1297',  // Best SDR vocal model

  // High-quality instrumental (single model fallback)
  'instrumental_ultimate': 'unwa-inst-v1e-plus',  // Best fullness (v1e+)
  'rock_metal_instrumental': 'unwa-inst-v1e-plus',
  'pop_instrumental': 'gabox-inst-fv7z',  // Less noise for pop

  // Bleedless production
  'bleedless_production': 'unwa-kim-ft2-bleedless',  // FT2 bleedless model

  // Instrument-specific
  'guitar_extraction': 'becruily-guitar',  // Dedicated guitar model
  'drums_isolation': 'mel-roformer-drumsep-5stem',  // 5-stem drumsep

  // Vocal processing
  'speech_dialogue_cleanup': 'unwa-revive-3e',  // Vocal enhancement
  'vocal_dereverb_enhance': 'anvuew-dereverb',  // Dedicated dereverb

  // Special processing
  'denoise_master': 'gabox-denoise-debleed',  // Gabox denoise/debleed
  'instrumental_gabox_progression': 'gabox-inst-fv7z',  // Latest Gabox instrumental
}

// =============================================================================
// ENSEMBLE PRESETS
// Based on official guide recommendations (lines 3605-3700)
// Algorithm options: 'average', 'max_spec', 'min_spec'
// Phase Fix is now a separate checkbox that can be combined with any algorithm
// =============================================================================

export interface EnsembleConfig {
  id: string;
  name: string;
  description: string;
  models: Array<{ model_id: string; weight: number }>;
  algorithm: 'average' | 'max_spec' | 'min_spec';
  phaseFixEnabled?: boolean;
  category: string;
  recommended?: boolean;
}

export const ENSEMBLE_PRESETS: EnsembleConfig[] = [
  {
    id: 'ensemble_hyperace_ultimate',
    name: 'HyperACE Ultimate (Top Rated)',
    description: 'HyperACE + Viperx 1297. Favorite ensemble per guide - best balance of fullness and clarity.',
    models: [
      { model_id: 'unwa-hyperace', weight: 1.0 },
      { model_id: 'bs-roformer-viperx-1297', weight: 1.0 }
    ],
    algorithm: 'max_spec',
    category: 'instrumental',
    recommended: true
  },
  {
    id: 'ensemble_v1e_becruily',
    name: 'v1e+ + Becruily (Best Preservation)',
    description: 'Per guide: best instrument preservation with Max Spec blending.',
    models: [
      { model_id: 'unwa-inst-v1e-plus', weight: 1.0 },
      { model_id: 'becruily-inst', weight: 1.0 }
    ],
    algorithm: 'max_spec',
    category: 'instrumental',
    recommended: true
  },
  {
    id: 'ensemble_gabox_bleedless',
    name: 'Gabox Bleedless (Least Noise)',
    description: 'Gabox Fv7z + Viperx 1297. Minimal noise and vocal bleed.',
    models: [
      { model_id: 'gabox-inst-fv7z', weight: 1.0 },
      { model_id: 'bs-roformer-viperx-1297', weight: 1.0 }
    ],
    algorithm: 'max_spec',
    category: 'instrumental'
  },
  {
    id: 'ensemble_resurrection_clean',
    name: 'Resurrection Clean',
    description: 'Resurrection inst with FT2 bleedless for clean output.',
    models: [
      { model_id: 'unwa-resurrection-inst', weight: 1.0 },
      { model_id: 'unwa-kim-ft2-bleedless', weight: 0.8 }
    ],
    algorithm: 'average',
    category: 'instrumental'
  },
  {
    id: 'ensemble_vocal_ultimate',
    name: 'Vocal Ultimate',
    description: 'Viperx 1297 + Resurrection vocal for maximum vocal quality.',
    models: [
      { model_id: 'bs-roformer-viperx-1297', weight: 1.0 },
      { model_id: 'unwa-resurrection-voc', weight: 0.8 }
    ],
    algorithm: 'max_spec',
    category: 'vocals'
  }
]

// Helper to check if a preset is an ensemble
export function isEnsemblePreset(presetId: string): boolean {
  return ENSEMBLE_PRESETS.some(e => e.id === presetId);
}

// Get ensemble config by ID
export function getEnsembleConfig(presetId: string): EnsembleConfig | undefined {
  return ENSEMBLE_PRESETS.find(e => e.id === presetId);
}

export const ALL_PRESETS = [
  {
    id: 'vocals_lead_back_ultimate',
    name: 'Vocal Separation (Lead + Backing) - Ultimate Quality',
    description: 'Best quality for separating lead vocals, backing vocals, and instrumental. Top community pick.',
    stems: ['Lead Vocals', 'Backing Vocals', 'Instrumental'],
    recommended: true,
    category: 'vocals',
    workflow: 'single',
    vram_required: 8.0,
    speed: 'medium'
  },
  {
    id: 'vocals_fast',
    name: 'Vocal Separation - Fast',
    description: 'Fast vocal/instrumental separation for quick preview or batch processing.',
    stems: ['Vocals', 'Instrumental'],
    recommended: true,
    category: 'vocals',
    workflow: 'single',
    vram_required: 6.0,
    speed: 'very_fast'
  },
  {
    id: 'karaoke_production',
    name: 'Karaoke Production',
    description: 'Professional karaoke creation with lead vocal, backing vocal, and instrumental tracks.',
    stems: ['Lead Vocals', 'Backing Vocals', 'Instrumental'],
    recommended: true,
    category: 'karaoke',
    workflow: 'sequential',
    vram_required: 8.0,
    speed: 'slow'
  },
  {
    id: 'instrumental_ultimate',
    name: 'Instrumental Separation - Ultimate',
    description: 'Best instrumental separation with maximum quality and fullness.',
    stems: ['Instrumental'],
    recommended: true,
    category: 'instrumental',
    workflow: 'single',
    vram_required: 10.0,
    speed: 'slow'
  },
  {
    id: 'mdx_vr_demucs_ensemble',
    name: 'MDX + VR + Demucs Ensemble',
    description: 'Classic ensemble from UVR community. Top quality for difficult sources.',
    stems: ['Vocals', 'Instrumental'],
    recommended: false,
    category: 'advanced',
    workflow: 'ensemble',
    vram_required: 8.0,
    speed: 'slow'
  },
  {
    id: 'bleedless_production',
    name: 'Bleedless Production',
    description: 'Maximum bleed removal for cleanest vocal/instrumental split.',
    stems: ['Vocals', 'Instrumental'],
    recommended: false,
    category: 'advanced',
    workflow: 'sequential',
    vram_required: 8.0,
    speed: 'medium'
  },
  {
    id: 'guitar_extraction',
    name: 'Guitar Extraction',
    description: 'Extract guitar from full mix. Works for both acoustic and electric.',
    stems: ['Guitar', 'Other'],
    recommended: false,
    category: 'instrumental',
    workflow: 'sequential',
    vram_required: 10.0,
    speed: 'medium'
  },
  {
    id: 'drums_isolation',
    name: 'Drums Isolation (6-stem)',
    description: 'Separate drums into kick, snare, hi-hat, toms, cymbals, and other percussion.',
    stems: ['Kick', 'Snare', 'Hi-Hat', 'Toms', 'Cymbals', 'Percussion'],
    recommended: false,
    category: 'instrumental',
    workflow: 'sequential',
    vram_required: 8.0,
    speed: 'slow'
  },
  {
    id: 'rock_metal_instrumental',
    name: 'Rock/Metal Instrumental Extraction',
    description: 'Optimized for heavy rock and metal tracks with dense instrumentation.',
    stems: ['Vocals', 'Instrumental'],
    recommended: false,
    category: 'instrumental',
    workflow: 'ensemble',
    vram_required: 8.0,
    speed: 'slow'
  },
  {
    id: 'pop_instrumental',
    name: 'Pop Instrumental Extraction',
    description: 'Optimized for pop music with cleaner production.',
    stems: ['Vocals', 'Instrumental'],
    recommended: false,
    category: 'instrumental',
    workflow: 'ensemble',
    vram_required: 8.0,
    speed: 'slow'
  },
  {
    id: 'speech_dialogue_cleanup',
    name: 'Speech/Dialogue Cleanup',
    description: 'Extract clean speech/dialogue from audio with background noise or music.',
    stems: ['Speech', 'Background'],
    recommended: false,
    category: 'vocals',
    workflow: 'single',
    vram_required: 6.0,
    speed: 'fast'
  },
  {
    id: 'vocal_dereverb_enhance',
    name: 'Vocal DeReverb + Enhance',
    description: 'Remove reverb from vocals and enhance quality for cleaner sound.',
    stems: ['Vocals Clean'],
    recommended: false,
    category: 'vocals',
    workflow: 'sequential',
    vram_required: 6.0,
    speed: 'fast'
  },
  {
    id: 'denoise_master',
    name: 'Denoise Master Cleanup',
    description: 'Maximum noise removal using highest SDR denoise model.',
    stems: ['Clean Audio'],
    recommended: false,
    category: 'advanced',
    workflow: 'single',
    vram_required: 5.0,
    speed: 'fast'
  },
  {
    id: 'instrumental_gabox_progression',
    name: 'Instrumental - Gabox Series Test',
    description: 'Test Gabox instrumental progression to find best variant for your source.',
    stems: ['Instrumental V6', 'Instrumental V7', 'Instrumental V8'],
    recommended: false,
    category: 'advanced',
    workflow: 'compare',
    vram_required: 6.0,
    speed: 'medium'
  },
  {
    id: 'low_vram_workflow',
    name: 'Low VRAM Workflow (4GB)',
    description: 'Complete workflow for systems with limited VRAM (4-5GB).',
    stems: ['Vocals', 'Instrumental'],
    recommended: false,
    category: 'vocals',
    workflow: 'sequential',
    vram_required: 4.0,
    speed: 'fast'
  },
  // =============================================================================
  // ENSEMBLE PRESETS (NEW)
  // Based on official guide recommendations
  // =============================================================================
  {
    id: 'ensemble_hyperace_ultimate',
    name: 'HyperACE Ultimate (Top Rated)',
    description: 'HyperACE + Viperx 1297. Favorite ensemble per guide - best balance.',
    stems: ['Vocals', 'Instrumental'],
    recommended: true,
    category: 'ensemble',
    workflow: 'ensemble',
    vram_required: 12.0,
    speed: 'slow'
  },
  {
    id: 'ensemble_v1e_becruily',
    name: 'v1e+ + Becruily (Best Preservation)',
    description: 'Per guide: best instrument preservation with Max Spec blending.',
    stems: ['Vocals', 'Instrumental'],
    recommended: true,
    category: 'ensemble',
    workflow: 'ensemble',
    vram_required: 10.0,
    speed: 'slow'
  },
  {
    id: 'ensemble_gabox_bleedless',
    name: 'Gabox Bleedless (Least Noise)',
    description: 'Gabox Fv7z + Viperx 1297. Minimal noise and vocal bleed.',
    stems: ['Vocals', 'Instrumental'],
    recommended: false,
    category: 'ensemble',
    workflow: 'ensemble',
    vram_required: 10.0,
    speed: 'slow'
  },
  {
    id: 'ensemble_resurrection_clean',
    name: 'Resurrection Clean',
    description: 'Resurrection inst + FT2 bleedless for clean output.',
    stems: ['Vocals', 'Instrumental'],
    recommended: false,
    category: 'ensemble',
    workflow: 'ensemble',
    vram_required: 8.0,
    speed: 'medium'
  },
  {
    id: 'ensemble_vocal_ultimate',
    name: 'Vocal Ultimate',
    description: 'Viperx 1297 + Resurrection vocal for maximum vocal quality.',
    stems: ['Vocals', 'Instrumental'],
    recommended: false,
    category: 'ensemble',
    workflow: 'ensemble',
    vram_required: 10.0,
    speed: 'slow'
  },
]

