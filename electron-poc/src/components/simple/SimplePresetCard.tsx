import { useMemo, useState } from 'react'
import { Info } from 'lucide-react'

import { Badge } from '@/components/ui/badge'
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
    if (preset.workflowSummary) lines.push(`Workflow: ${preset.workflowSummary}`)
    if (typeof preset.guideRank === 'number') lines.push(`Guide rank: #${preset.guideRank}`)
    if (typeof preset.estimatedVram === 'number') lines.push(`VRAM hint: ${preset.estimatedVram}GB`)
    if (preset.stems?.length) lines.push(`Stems: ${preset.stems.join(', ')}`)
    if (preset.recommendedFor?.length) lines.push(`Best for: ${preset.recommendedFor.join('; ')}`)
    if (preset.contraindications?.length) lines.push(`Avoid when: ${preset.contraindications.join('; ')}`)
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
    <div
      className={cn(
        'group relative w-full overflow-visible rounded-[1.25rem] border p-4 text-left transition-all duration-300',
        selected
          ? 'border-white bg-white/84 shadow-[0_24px_70px_rgba(141,150,179,0.18)]'
          : 'border-white/55 bg-[rgba(255,255,255,0.52)] hover:border-white/80 hover:bg-[rgba(255,255,255,0.7)]'
      )}
      onClick={onSelect}
      onKeyDown={(event) => {
        if (event.key === 'Enter' || event.key === ' ') {
          event.preventDefault()
          onSelect()
        }
      }}
      role="button"
      tabIndex={0}
      aria-pressed={selected}
    >
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="truncate text-[15px] tracking-[-0.025em] text-slate-800">
            {preset.name}
          </div>
          {preset.description && (
            <p className="mt-1 line-clamp-2 text-[12px] leading-[1.45] text-slate-500">
              {preset.description}
            </p>
          )}
        </div>

        <div className="flex items-center gap-2 shrink-0">
          {typeof preset.estimatedVram === 'number' && (
            <div className="rounded-full border border-white/60 bg-white/60 px-2 py-0.5 text-[11px] text-slate-500">
              {preset.estimatedVram}GB
            </div>
          )}

          <div className="relative" onClick={e => e.stopPropagation()}>
            <button
              type="button"
              aria-label={`Preset info: ${preset.name}`}
              className={cn(
                'inline-flex h-7 w-7 items-center justify-center rounded-lg border border-white/60 bg-white/58',
                'text-slate-500 hover:bg-white/80 hover:text-slate-800',
                'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-slate-300/60'
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
                  'absolute right-0 z-50 mt-2 w-[320px] rounded-2xl border border-white/70 bg-[rgba(255,255,255,0.92)] p-3 text-xs text-slate-800 shadow-[0_24px_80px_rgba(141,150,179,0.22)] backdrop-blur-xl'
                )}
              >
                <div className="mb-1 text-sm font-medium text-slate-800">{preset.name}</div>
                {tooltipLines.length > 0 ? (
                  <div className="space-y-1">
                    {tooltipLines.map((l, i) => (
                      <div key={i} className="text-slate-600">
                        {l}
                      </div>
                    ))}
                    {stepPreview && (
                      <div className="text-slate-600">{stepPreview}</div>
                    )}
                  </div>
                ) : (
                  <div className="text-slate-600">No details.</div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>

      <div className="mt-2 flex flex-wrap gap-1.5">
        {badges.map((b, index) => (
          <Badge
            key={`${b}-${index}`}
            variant={b === 'Best' ? 'default' : 'secondary'}
            className={cn(
              'border-0 text-[10px]',
              b === 'Best'
                ? 'bg-white text-[#111111]'
                : 'bg-slate-900/6 text-slate-600'
            )}
          >
            {b}
          </Badge>
        ))}
        {isRecipe && (
          <Badge
            variant="outline"
            className="border-white/60 bg-white/60 text-[10px] text-slate-600"
            title="Runs multiple steps automatically"
          >
            Multi-step
          </Badge>
        )}
        {typeof preset.guideRank === 'number' && (
          <Badge variant="outline" className="border-white/60 bg-white/60 text-[10px] text-slate-600" title="Guide priority">
            #{preset.guideRank}
          </Badge>
        )}
        {missingIds.length > 0 && (
          <Badge variant="destructive" className="border-0 bg-rose-500/14 text-[10px] text-rose-700">
            Missing {missingIds.length} model{missingIds.length === 1 ? '' : 's'}
          </Badge>
        )}
      </div>

      <div className="mt-3 space-y-1">
        <RatingMeter label="Quality" value={quality} />
        <RatingMeter label="Speed" value={speed} />
      </div>
    </div>
  )
}
