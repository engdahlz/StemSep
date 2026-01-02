import { useMemo, useState } from 'react'
import { Info } from 'lucide-react'

import { Badge } from '@/components/ui/badge'
import { Card } from '@/components/ui/card'
import { cn } from '@/lib/utils'
import { getRequiredModels, type Preset } from '@/presets'
import { RatingMeter } from './RatingMeter'
import { presetQualityScore, presetSpeedScore, presetSimpleBadges } from '@/lib/simplePresetViewModel'

export function SimplePresetCard({
  preset,
  selected,
  availability,
  onSelect,
}: {
  preset: Preset
  selected: boolean
  availability?: Record<string, any>
  onSelect: () => void
}) {
  const [infoOpen, setInfoOpen] = useState(false)

  const required = getRequiredModels(preset)
  const missingIds = availability
    ? required.filter(id => availability[id]?.available !== true)
    : []

  const badges = presetSimpleBadges(preset)
  const quality = presetQualityScore(preset)
  const speed = presetSpeedScore(preset)
  const isRecipe = Boolean((preset as any)?.isRecipe)

  const recipeType = (preset as any)?.recipe?.type as string | undefined
  const steps = Array.isArray((preset as any)?.recipe?.steps) ? (preset as any).recipe.steps : []
  const stepCount = steps.length

  const tooltipLines = useMemo(() => {
    const lines: string[] = []
    if (preset.description) lines.push(preset.description)
    if (typeof preset.estimatedVram === 'number') lines.push(`VRAM hint: ${preset.estimatedVram}GB`)
    if (preset.stems?.length) lines.push(`Stems: ${preset.stems.join(', ')}`)
    if (isRecipe) {
      if (recipeType) lines.push(`Type: ${recipeType}`)
      if (stepCount > 0) lines.push(`Steps: ${stepCount}`)
    }

    if (required.length > 0) {
      const requiredNames = required.map(id => availability?.[id]?.model_name || id)
      lines.push(`Required models: ${requiredNames.join(', ')}`)
    }
    if (missingIds.length > 0) {
      const missingNames = missingIds.map(id => availability?.[id]?.model_name || id)
      lines.push(`Missing: ${missingNames.join(', ')}`)
    }
    return lines
  }, [availability, isRecipe, missingIds, preset.description, preset.estimatedVram, preset.stems, recipeType, required, stepCount])

  const stepPreview = useMemo(() => {
    if (!isRecipe || steps.length === 0) return ''
    const names = steps
      .map((s: any, idx: number) => s.step_name || s.name || s.action || `step_${idx + 1}`)
      .slice(0, 4)
    const suffix = steps.length > 4 ? '…' : ''
    return `Steps: ${names.join(' → ')}${suffix}`
  }, [isRecipe, steps])

  return (
    <Card
      className={cn(
        'p-3 cursor-pointer transition-colors',
        selected ? 'border-primary/60 bg-primary/5' : 'hover:bg-muted/30'
      )}
      onClick={onSelect}
      role="button"
      aria-pressed={selected}
    >
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="text-sm font-medium truncate">{preset.name}</div>
        </div>

        <div className="flex items-center gap-2 shrink-0">
          {typeof preset.estimatedVram === 'number' && (
            <div className="text-[11px] text-muted-foreground">{preset.estimatedVram}GB</div>
          )}

          <div className="relative" onClick={e => e.stopPropagation()}>
            <button
              type="button"
              aria-label={`Preset info: ${preset.name}`}
              className={cn(
                'h-7 w-7 inline-flex items-center justify-center rounded-md border bg-background/60',
                'text-muted-foreground hover:text-foreground hover:bg-background',
                'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/40'
              )}
              onMouseEnter={() => setInfoOpen(true)}
              onMouseLeave={() => setInfoOpen(false)}
              onFocus={() => setInfoOpen(true)}
              onBlur={() => setInfoOpen(false)}
              onClick={() => setInfoOpen(v => !v)}
              title={tooltipLines.join('\n')}
            >
              <Info className="h-4 w-4" />
            </button>

            {infoOpen && (
              <div
                className={cn(
                  'absolute right-0 mt-2 w-[320px] z-50 rounded-md border bg-popover text-popover-foreground shadow-lg',
                  'p-3 text-xs'
                )}
              >
                <div className="font-medium text-sm mb-1">{preset.name}</div>
                {tooltipLines.length > 0 ? (
                  <div className="space-y-1">
                    {tooltipLines.map((l, i) => (
                      <div key={i} className="text-muted-foreground">
                        {l}
                      </div>
                    ))}
                    {stepPreview && (
                      <div className="text-muted-foreground">{stepPreview}</div>
                    )}
                  </div>
                ) : (
                  <div className="text-muted-foreground">No details.</div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>

      <div className="mt-2 flex flex-wrap gap-1.5">
        {badges.map(b => (
          <Badge key={b} variant={b === 'Best' ? 'default' : 'secondary'} className="text-[10px]">
            {b}
          </Badge>
        ))}
        {isRecipe && (
          <Badge variant="outline" className="text-[10px]" title="Runs multiple steps automatically">
            Multi-step
          </Badge>
        )}
        {missingIds.length > 0 && (
          <Badge variant="destructive" className="text-[10px]">
            Missing {missingIds.length} model{missingIds.length === 1 ? '' : 's'}
          </Badge>
        )}
      </div>

      <div className="mt-2 space-y-1">
        <RatingMeter label="Quality" value={quality} />
        <RatingMeter label="Speed" value={speed} />
      </div>
    </Card>
  )
}
