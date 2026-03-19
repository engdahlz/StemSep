import { useMemo, useState } from 'react'
import { Zap, Music2, Mic2, Wand2, Layers, Guitar } from 'lucide-react'

import { CollapsibleSection } from '@/components/ui/collapsible-section'
import type { Preset } from '@/presets'
import { cn } from '@/lib/utils'

import {
  type SimpleGoal,
  type SimpleMode,
  presetMatchesGoal,
  presetMatchesMode,
  presetIsRecommendedForSimple,
  presetSortScore,
} from '@/lib/simplePresetViewModel'

import { useLocalStorageState } from '@/hooks/useLocalStorageState'

import { SimplePresetCard } from './SimplePresetCard'

const GOALS: { id: SimpleGoal; label: string; icon: React.ReactNode }[] = [
  { id: 'all', label: 'All', icon: <Layers className="h-4 w-4" /> },
  { id: 'instrumental', label: 'Instrumental', icon: <Music2 className="h-4 w-4" /> },
  { id: 'vocals', label: 'Vocals', icon: <Mic2 className="h-4 w-4" /> },
  { id: 'karaoke', label: 'Karaoke', icon: <Zap className="h-4 w-4" /> },
  { id: 'instruments', label: 'Instruments', icon: <Guitar className="h-4 w-4" /> },
  { id: 'cleanup', label: 'Cleanup', icon: <Wand2 className="h-4 w-4" /> },
  { id: 'workflows', label: 'Multi-step', icon: <Zap className="h-4 w-4" /> },
]

const MODES: { id: SimpleMode; label: string }[] = [
  { id: 'all', label: 'All' },
  { id: 'fast', label: 'Fast' },
  { id: 'balanced', label: 'Balanced' },
  { id: 'quality', label: 'Quality' },
  { id: 'ultra', label: 'Ultra' },
]

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
  const [goal, setGoal] = useState<SimpleGoal>('all')
  const [mode, setMode] = useState<SimpleMode>('all')
  const [search, setSearch] = useState('')
  const [showWorkflows, setShowWorkflows] = useLocalStorageState<boolean>('stemsep.simple.showMultiStep', true)
  const simpleSurfaceExists = useMemo(() => presets.some(p => p.simpleVisible), [presets])

  const filtered = useMemo(() => {
    const s = search.trim().toLowerCase()
    return presets
      .filter(p => (simpleSurfaceExists ? p.simpleVisible === true : true))
      .filter(p => (showWorkflows ? true : !(p as any)?.isRecipe))
      .filter(p => presetMatchesGoal(p, goal))
      .filter(p => presetMatchesMode(p, mode))
      .filter(p => {
        if (!s) return true
        const hay = `${p.name} ${p.description} ${(p.tags || []).join(' ')}`.toLowerCase()
        return hay.includes(s)
      })
      .sort((a, b) => presetSortScore(b) - presetSortScore(a))
  }, [presets, goal, mode, search, showWorkflows, simpleSurfaceExists])

  const recommended = useMemo(() => filtered.filter(presetIsRecommendedForSimple).slice(0, 6), [filtered])
  const others = useMemo(() => filtered.filter(p => !presetIsRecommendedForSimple(p)), [filtered])
  const selectedPreset = useMemo(
    () => presets.find((preset) => preset.id === selectedPresetId) || null,
    [presets, selectedPresetId],
  )

  return (
    <div className="rounded-[1.5rem] border border-white/55 bg-[rgba(255,255,255,0.5)] p-4 shadow-[0_20px_60px_rgba(141,150,179,0.14)] backdrop-blur-xl">
      <div className="space-y-4">
        <div className="flex items-center justify-between gap-3">
          <div>
            <div className="text-sm font-medium text-slate-800">Pick a preset</div>
            <div className="mt-1 text-[12px] text-slate-500">
              Curated workflows for the new configuration surface.
            </div>
          </div>
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => setShowWorkflows(v => !v)}
              title="Show/hide multi-step presets"
              className={cn(
                'rounded-full px-3 py-1.5 text-[12px] tracking-[-0.2px] transition-all',
                showWorkflows
                  ? 'bg-white text-[#111111]'
                  : 'border border-white/60 bg-white/55 text-slate-600 hover:bg-white/75 hover:text-slate-800'
              )}
            >
              Multi-step {showWorkflows ? 'ON' : 'OFF'}
            </button>
          </div>
        </div>

        {simpleSurfaceExists && (
          <div className="text-xs text-slate-500">
            Simple mode is locked to a curated guide-driven set of premium workflows for this surface.
          </div>
        )}

        <div className="space-y-2">
          <div className="text-xs text-slate-500">Goal</div>
          <div className="flex flex-wrap gap-2">
            {GOALS.map(g => (
              <button
                key={g.id}
                type="button"
                onClick={() => setGoal(g.id)}
                className={cn(
                  'inline-flex items-center gap-2 rounded-full px-3.5 py-1.5 text-[13px] tracking-[-0.2px] transition-all',
                  goal === g.id
                    ? 'bg-white text-[#111111]'
                    : 'bg-white/55 text-slate-600 hover:bg-white/75 hover:text-slate-800'
                )}
              >
                <span className={goal === g.id ? 'text-[#111111]' : 'text-slate-500'}>{g.icon}</span>
                {g.label}
              </button>
            ))}
          </div>
        </div>

        <div className="space-y-2">
          <div className="text-xs text-slate-500">Mode</div>
          <div className="flex flex-wrap gap-2">
            {MODES.map(m => (
              <button
                key={m.id}
                type="button"
                onClick={() => setMode(m.id)}
                className={cn(
                  'rounded-full px-3.5 py-1.5 text-[13px] tracking-[-0.2px] transition-all',
                  mode === m.id
                    ? 'bg-white text-[#111111]'
                    : 'bg-white/55 text-slate-600 hover:bg-white/75 hover:text-slate-800'
                )}
              >
                {m.label}
              </button>
            ))}
          </div>
        </div>

        <div className="flex items-center gap-2">
          <input
            value={search}
            onChange={e => setSearch(e.target.value)}
            placeholder="Search presets…"
            className="h-11 w-full rounded-2xl border border-white/60 bg-white/62 px-4 text-[14px] text-slate-800 outline-none placeholder:text-slate-400 focus:border-white focus:bg-white/82"
          />
          <button
            type="button"
            onClick={() => {
              setGoal('all')
              setMode('all')
              setSearch('')
            }}
            className="rounded-2xl border border-white/60 bg-white/55 px-4 py-3 text-[13px] text-slate-600 transition-all hover:bg-white/75 hover:text-slate-800"
          >
            Reset
          </button>
        </div>

        <div className="flex flex-wrap items-center justify-between gap-3 rounded-[1.15rem] border border-white/55 bg-white/42 px-4 py-3">
          <div className="min-w-0">
            <div className="text-[11px] uppercase tracking-[0.18em] text-slate-500">
              Selected Preset
            </div>
            <div className="mt-1 truncate text-[15px] tracking-[-0.02em] text-slate-800">
              {selectedPreset?.name || "None selected"}
            </div>
          </div>
          {selectedPreset && (
            <div className="rounded-full border border-emerald-300/55 bg-emerald-50/80 px-3 py-1 text-[12px] text-emerald-700">
              Active
            </div>
          )}
        </div>
        {recommended.length > 0 && (
          <div className="space-y-2">
            <div className="pt-1 text-sm font-medium text-slate-800">Recommended</div>
            <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
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
            <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
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
