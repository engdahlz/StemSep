import { useMemo } from 'react'

import { CollapsibleSection } from '@/components/ui/collapsible-section'
import type { Preset } from '@/presets'

import {
  presetIsRecommendedForSimple,
  presetSortScore,
} from '@/lib/simplePresetViewModel'

import { SimplePresetCard } from './SimplePresetCard'

export function SimplePresetPicker({
  presets,
  selectedPresetId,
  onSelectPreset,
  availability,
}: {
  presets: Preset[]
  selectedPresetId: string
  onSelectPreset: (id: string) => void
  availability?: Record<string, any>
}) {
  const simpleSurfaceExists = useMemo(() => presets.some(p => p.simpleVisible), [presets])

  const filtered = useMemo(() => {
    return presets
      .filter(p => (simpleSurfaceExists ? p.simpleVisible === true : true))
      .sort((a, b) => presetSortScore(b) - presetSortScore(a))
  }, [presets, simpleSurfaceExists])

  const recommended = useMemo(() => filtered.filter(presetIsRecommendedForSimple).slice(0, 6), [filtered])
  const others = useMemo(() => filtered.filter(p => !presetIsRecommendedForSimple(p)), [filtered])

  return (
    <div className="rounded-[1.65rem] border border-white/60 bg-[rgba(255,255,255,0.52)] p-5 shadow-[0_24px_72px_rgba(141,150,179,0.16)] backdrop-blur-xl">
      <div className="space-y-4">
        <div className="flex items-center justify-between gap-3">
          <div>
            <div className="text-[11px] uppercase tracking-[0.18em] text-slate-400">Presets</div>
            <div className="mt-1 text-[22px] tracking-[-0.04em] text-slate-800">
              Choose the result you want
            </div>
            <div className="mt-1 text-[13px] text-slate-500">
              Simple mode is intentionally preset-only.
            </div>
          </div>
        </div>

        {recommended.length > 0 && (
          <div className="space-y-3">
            <div className="pt-1 text-sm font-medium text-slate-800">Recommended</div>
            <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
              {recommended.map(p => (
                <SimplePresetCard
                  key={p.id}
                  preset={p}
                  selected={p.id === selectedPresetId}
                  availability={availability}
                  onSelect={() => onSelectPreset(p.id)}
                />
              ))}
            </div>
          </div>
        )}

          <CollapsibleSection
            title={`More options (${others.length})`}
            defaultOpen={recommended.length === 0}
            className="border-white/55 bg-white/28"
          >
          {others.length === 0 ? (
            <div className="text-sm text-slate-500">No presets match your filters.</div>
          ) : (
            <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
              {others.map(p => (
                <SimplePresetCard
                  key={p.id}
                  preset={p}
                  selected={p.id === selectedPresetId}
                  availability={availability}
                  onSelect={() => onSelectPreset(p.id)}
                />
              ))}
            </div>
          )}
        </CollapsibleSection>
      </div>
    </div>
  )
}
