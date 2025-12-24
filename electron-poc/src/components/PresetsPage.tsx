import { useState } from 'react'
import { Search, SlidersHorizontal, Sparkles } from 'lucide-react'
import { Button } from './ui/button'
import { Input } from './ui/input'
import { Badge } from './ui/badge'
import { PresetCard, Preset } from './PresetCard'

// Extended Preset definition with all metadata
const ALL_PRESETS: Preset[] = [
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
]

export function PresetsPage() {
  const [selectedCategory, setSelectedCategory] = useState<string>('all')
  const [searchQuery, setSearchQuery] = useState('')
  const [favoriteIds, setFavoriteIds] = useState<string[]>(
    ALL_PRESETS.filter(p => p.recommended).map(p => p.id)
  )

  const categories = ['all', 'vocals', 'instrumental', 'karaoke', 'advanced']

  const filteredPresets = ALL_PRESETS.filter(preset => {
    const matchesCategory = selectedCategory === 'all' || preset.category === selectedCategory
    const matchesSearch = preset.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      preset.description.toLowerCase().includes(searchQuery.toLowerCase())
    return matchesCategory && matchesSearch
  })

  const toggleFavorite = (presetId: string) => {
    setFavoriteIds(prev =>
      prev.includes(presetId)
        ? prev.filter(id => id !== presetId)
        : [...prev, presetId]
    )
  }

  return (
    <div className="h-full overflow-auto p-6 md:p-8 space-y-8">
      {/* Header Section */}
      <div className="flex flex-col md:flex-row justify-between items-start md:items-end gap-4">
        <div className="space-y-2">
          <h1 className="text-4xl font-bold tracking-tight bg-gradient-to-r from-white to-white/60 bg-clip-text text-transparent">
            Presets
          </h1>
          <p className="text-muted-foreground max-w-2xl text-lg">
            Curated workflows for specific separation tasks. Optimized combinations of models and settings.
          </p>
        </div>

        {/* Search */}
        <div className="relative w-full md:w-72">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input
            placeholder="Search presets..."
            className="pl-9 bg-secondary/50 border-white/10 focus:border-primary/50 transition-all"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
        </div>
      </div>

      {/* Filters */}
      <div className="flex flex-wrap items-center gap-3 pb-4 border-b border-white/5">
        <div className="flex items-center gap-2 mr-4 text-sm font-medium text-muted-foreground">
          <SlidersHorizontal className="w-4 h-4" />
          <span>Filters:</span>
        </div>

        {categories.map((category) => (
          <Button
            key={category}
            variant={selectedCategory === category ? "default" : "outline"}
            size="sm"
            onClick={() => setSelectedCategory(category)}
            className={`capitalize transition-all duration-300 ${selectedCategory === category
                ? "bg-primary text-primary-foreground shadow-lg shadow-primary/20"
                : "bg-transparent border-white/10 hover:bg-white/5 hover:border-white/20"
              }`}
          >
            {category}
          </Button>
        ))}

        <div className="ml-auto flex items-center gap-2">
          <Badge variant="outline" className="bg-primary/5 border-primary/20 text-primary">
            <Sparkles className="w-3 h-3 mr-1" />
            {filteredPresets.length} Presets
          </Badge>
        </div>
      </div>

      {/* Presets Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
        {filteredPresets.map((preset) => (
          <PresetCard
            key={preset.id}
            preset={preset}
            isFavorite={favoriteIds.includes(preset.id)}
            onToggleFavorite={toggleFavorite}
          />
        ))}
      </div>

      {filteredPresets.length === 0 && (
        <div className="flex flex-col items-center justify-center py-20 text-center space-y-4 opacity-60">
          <div className="p-4 rounded-full bg-white/5">
            <Search className="w-8 h-8 text-muted-foreground" />
          </div>
          <div>
            <h3 className="text-lg font-medium">No presets found</h3>
            <p className="text-sm text-muted-foreground">Try adjusting your search or filters</p>
          </div>
          <Button
            variant="outline"
            onClick={() => {
              setSelectedCategory('all')
              setSearchQuery('')
            }}
          >
            Clear Filters
          </Button>
        </div>
      )}
    </div>
  )
}

