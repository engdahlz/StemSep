import { useState } from 'react'
import { X, Star, AlertCircle, Lock, Zap } from 'lucide-react'
import { Button } from './ui/button'
import { Card, CardContent, CardHeader, CardTitle } from './ui/card'
import { Badge } from './ui/badge'
import { cn } from '@/lib/utils'

import { Preset } from '../presets'

// Estimate minimum VRAM required for a preset
const estimatePresetVRAM = (preset: Preset): number => {
  const baseVRAM: Record<string, number> = {
    'roformer': 6,
    'mdx': 3,
    'demucs': 4,
    'scnet': 3
  }

  const estimateModelId = (modelId: string): number => {
    const type = Object.keys(baseVRAM).find(k => modelId.toLowerCase().includes(k))
    return type ? baseVRAM[type] : 4
  }

  if (preset.isRecipe && preset.recipe?.requiredModels?.length) {
    const required = preset.recipe.requiredModels
    const maxOne = Math.max(...required.map(estimateModelId))
    if (preset.recipe.type === 'ensemble') {
      // Ensemble is closer to additive (not perfectly), so bump slightly.
      return maxOne * (1 + Math.max(0, required.length - 1) * 0.3)
    }
    // Pipelines/chains are sequential, so max is a better approximation.
    return maxOne
  }

  if (preset.ensembleConfig) {
    // Ensemble needs memory for multiple models (roughly additive)
    const modelCount = preset.ensembleConfig.models.length
    return Math.max(...preset.ensembleConfig.models.map(m => {
      return estimateModelId(m.model_id)
    })) * (1 + (modelCount - 1) * 0.3)
  }

  const modelId = preset.modelId || ''
  return modelId ? estimateModelId(modelId) : 4
}

type PresetVRAMStatus = 'safe' | 'warning' | 'locked'

interface PresetBrowserProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  presets: Preset[]
  favoriteIds: string[]
  onToggleFavorite: (presetId: string) => void
  onSelectPreset: (presetId: string) => void
  availability?: Record<string, { available: boolean; model_name: string | null }>
  onNavigateToModels?: () => void
  onShowModelDetails?: (modelId: string) => void
  gpuVRAM?: number // Available GPU VRAM in GB (for filtering)
}

export function PresetBrowser({
  open,
  onOpenChange,
  presets,
  favoriteIds,
  onToggleFavorite,
  onSelectPreset,
  availability,
  onNavigateToModels,
  onShowModelDetails,
  gpuVRAM = 0
}: PresetBrowserProps) {
  const [selectedCategory, setSelectedCategory] = useState<string>('all')

  // Helper to get VRAM status for a preset
  const getVRAMStatus = (preset: Preset): PresetVRAMStatus => {
    if (gpuVRAM <= 0) return 'safe' // Unknown VRAM, allow all
    const required = estimatePresetVRAM(preset)
    if (required > gpuVRAM) return 'locked'
    if (required > gpuVRAM * 0.8) return 'warning'
    return 'safe'
  }

  if (!open) return null

  const categories = ['all', 'vocals', 'instrumental', 'instruments', 'utility', 'ensemble', 'smart']

  const filteredPresets = selectedCategory === 'all'
    ? presets
    : selectedCategory === 'ensemble'
      ? presets.filter(p => !!p.ensembleConfig && !p.isRecipe)
      : presets.filter(p => p.category === selectedCategory)

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-8 backdrop-blur-sm animate-in fade-in duration-200">
      <Card className="w-full max-w-4xl max-h-[85vh] flex flex-col shadow-2xl" style={{ backgroundColor: '#ffffff' }}>
        <CardHeader className="flex flex-row items-center justify-between border-b px-6 py-4">
          <div>
            <CardTitle className="text-xl">Browse Presets</CardTitle>
            <p className="text-sm text-muted-foreground mt-1">Select a configuration for your separation task</p>
          </div>
          <Button variant="ghost" size="icon" onClick={() => onOpenChange(false)}>
            <X className="h-5 w-5" />
          </Button>
        </CardHeader>

        <CardContent className="flex-1 overflow-hidden p-6 flex flex-col gap-6">
          {/* Category Filter */}
          <div className="flex gap-2 overflow-x-auto pb-2">
            {categories.map((category) => (
              <Button
                key={category}
                variant={selectedCategory === category ? "default" : "outline"}
                size="sm"
                onClick={() => setSelectedCategory(category)}
                className="capitalize rounded-full px-4"
              >
                {category}
              </Button>
            ))}
          </div>

          {/* Presets Grid */}
          <div className="overflow-y-auto pr-2 -mr-2 flex-1">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {filteredPresets.map((preset) => {
                const isFavorite = favoriteIds.includes(preset.id)

                let isAvailable = true
                let missingModels: string[] = []
                let requiredModels: { id: string; name: string }[] = []

                if (preset.ensembleConfig) {
                  requiredModels = preset.ensembleConfig.models.map(m => ({
                    id: m.model_id,
                    name: availability?.[m.model_id]?.model_name || m.model_id
                  }))

                  if (availability) {
                    isAvailable = preset.ensembleConfig.models.every(m => availability[m.model_id]?.available === true)
                    missingModels = preset.ensembleConfig.models
                      .filter(m => availability[m.model_id]?.available !== true)
                      .map(m => availability[m.model_id]?.model_name || m.model_id)
                  }
                } else if (preset.modelId) {
                  requiredModels = [{
                    id: preset.modelId,
                    name: availability?.[preset.modelId]?.model_name || preset.modelId
                  }]

                  if (availability) {
                    isAvailable = availability[preset.modelId]?.available === true
                    if (!isAvailable) {
                      missingModels = [availability[preset.modelId]?.model_name || preset.modelId]
                    }
                  }
                }

                if (preset.isRecipe && preset.recipe?.requiredModels?.length) {
                  requiredModels = preset.recipe.requiredModels.map(id => ({
                    id,
                    name: availability?.[id]?.model_name || id
                  }))

                  if (availability) {
                    isAvailable = preset.recipe.requiredModels.every(id => availability[id]?.available === true)
                    missingModels = preset.recipe.requiredModels
                      .filter(id => availability[id]?.available !== true)
                      .map(id => availability[id]?.model_name || id)
                  }
                }

                // Check VRAM status
                const vramStatus = getVRAMStatus(preset)
                const isVRAMLocked = vramStatus === 'locked'
                const requiredVRAM = estimatePresetVRAM(preset)

                return (
                  <div
                    key={preset.id}
                    className={cn(
                      "group relative p-4 rounded-xl border-2 transition-all duration-200 text-left",
                      "hover:border-primary/50 hover:shadow-md",
                      !isAvailable && "opacity-60 grayscale",
                      isVRAMLocked && "opacity-50 border-yellow-500/30"
                    )}
                  >
                    {/* Lock Overlay for VRAM */}
                    {isVRAMLocked && (
                      <div className="absolute inset-0 flex items-center justify-center bg-black/5 rounded-xl z-20">
                        <div className="bg-yellow-500/90 text-white px-3 py-1.5 rounded-full text-xs font-medium flex items-center gap-1.5">
                          <Lock className="w-3 h-3" />
                          Requires {requiredVRAM.toFixed(0)}GB+ VRAM
                        </div>
                      </div>
                    )}
                    {/* Selection Overlay */}
                    <button
                      onClick={() => {
                        if (isAvailable) {
                          onSelectPreset(preset.id)
                          onOpenChange(false)
                        }
                      }}
                      className="absolute inset-0 w-full h-full z-0 focus:outline-none"
                      disabled={!isAvailable}
                    />

                    <div className="relative z-10 pointer-events-none">
                      <div className="flex justify-between items-start mb-2">
                        <div className="flex items-center gap-2">
                          <span className="font-semibold text-lg">{preset.name}</span>
                          {preset.recommended && gpuVRAM > 0 && vramStatus === 'safe' && (
                            <Badge variant="secondary" className="text-xs bg-green-100 text-green-800 hover:bg-green-100 flex items-center gap-1">
                              <Zap className="w-3 h-3" />
                              Recommended
                            </Badge>
                          )}
                          {preset.recommended && (gpuVRAM <= 0 || vramStatus !== 'safe') && (
                            <Badge variant="secondary" className="text-xs bg-yellow-100 text-yellow-800 hover:bg-yellow-100">
                              Recommended
                            </Badge>
                          )}
                          {!!preset.ensembleConfig && !preset.isRecipe && (
                            <Badge variant="secondary" className="text-xs bg-purple-100 text-purple-800 hover:bg-purple-100">
                              Ensemble
                            </Badge>
                          )}
                          {!!preset.isRecipe && (
                            <Badge variant="secondary" className="text-xs bg-blue-100 text-blue-800 hover:bg-blue-100">
                              Workflow{preset.recipe?.type ? `: ${preset.recipe.type}` : ''}
                            </Badge>
                          )}
                        </div>
                        {/* Star Toggle Button - Needs pointer-events-auto */}
                        <button
                          onClick={(e) => {
                            e.stopPropagation()
                            onToggleFavorite(preset.id)
                          }}
                          className="pointer-events-auto p-1.5 rounded-full hover:bg-accent transition-colors -mr-2 -mt-2"
                          title={isFavorite ? "Remove from favorites" : "Add to favorites"}
                        >
                          <Star
                            className={cn(
                              "w-5 h-5 transition-all",
                              isFavorite
                                ? "fill-yellow-400 text-yellow-400"
                                : "text-muted-foreground/30 group-hover:text-muted-foreground"
                            )}
                          />
                        </button>
                      </div>

                      <p className="text-sm text-muted-foreground mb-4 line-clamp-2">
                        {preset.description}
                      </p>

                      <div className="flex items-center justify-between mt-auto">
                        <div className="flex gap-1.5 flex-wrap">
                          {preset.stems.map(stem => (
                            <Badge key={stem} variant="outline" className="text-xs uppercase tracking-wider bg-background/50">
                              {stem}
                            </Badge>
                          ))}
                        </div>

                        {!isAvailable && (
                          <div className="flex flex-col items-end gap-1">
                            <div className="flex items-center text-xs text-destructive font-medium bg-destructive/10 px-2 py-1 rounded">
                              <AlertCircle className="w-3 h-3 mr-1" />
                              Missing: {missingModels.join(', ')}
                            </div>
                          </div>
                        )}
                        {isAvailable && requiredModels.length > 0 && (
                          <div className="text-xs text-muted-foreground/70 flex flex-wrap gap-1 items-center">
                            <span>Using:</span>
                            {requiredModels.map((m, i) => (
                              <span key={m.id} className="flex items-center">
                                <button
                                  onClick={(e) => {
                                    e.stopPropagation()
                                    onShowModelDetails?.(m.id)
                                  }}
                                  className="hover:text-primary hover:underline focus:outline-none transition-colors"
                                >
                                  {m.name}
                                </button>
                                {i < requiredModels.length - 1 && <span className="mr-1">,</span>}
                              </span>
                            ))}
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                )
              })}
            </div>
          </div>

          {availability && Object.values(availability).some(v => !v.available) && (
            <div className="bg-muted/50 p-3 rounded-lg flex items-center justify-between text-sm">
              <div className="flex items-center text-muted-foreground">
                <AlertCircle className="w-4 h-4 mr-2" />
                Some presets are unavailable because their models are missing.
              </div>
              {onNavigateToModels && (
                <Button variant="link" size="sm" onClick={onNavigateToModels}>
                  Manage Models
                </Button>
              )}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
