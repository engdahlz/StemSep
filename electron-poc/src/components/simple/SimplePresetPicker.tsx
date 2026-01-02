import { useMemo, useState } from 'react'
import { Zap, Music2, Mic2, Wand2, Layers, Guitar } from 'lucide-react'

import { Button } from '@/components/ui/button'
import { Card } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { CollapsibleSection } from '@/components/ui/collapsible-section'
import type { Preset } from '@/presets'

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

  const filtered = useMemo(() => {
    const s = search.trim().toLowerCase()
    return presets
      .filter(p => (showWorkflows ? true : !(p as any)?.isRecipe))
      .filter(p => presetMatchesGoal(p, goal))
      .filter(p => presetMatchesMode(p, mode))
      .filter(p => {
        if (!s) return true
        const hay = `${p.name} ${p.description} ${(p.tags || []).join(' ')}`.toLowerCase()
        return hay.includes(s)
      })
      .sort((a, b) => presetSortScore(b) - presetSortScore(a))
  }, [presets, goal, mode, search, showWorkflows])

  const recommended = useMemo(() => filtered.filter(presetIsRecommendedForSimple).slice(0, 6), [filtered])
  const others = useMemo(() => filtered.filter(p => !presetIsRecommendedForSimple(p)), [filtered])

  return (
    <div className="space-y-4">
      <Card className="p-4 space-y-3">
        <div className="flex items-center justify-between gap-3">
          <div className="text-sm font-medium">Pick a preset</div>
          <div className="flex items-center gap-2">
            <Button
              variant={showWorkflows ? 'default' : 'outline'}
              size="sm"
              onClick={() => setShowWorkflows(v => !v)}
              title="Show/hide multi-step presets"
            >
              Multi-step {showWorkflows ? 'ON' : 'OFF'}
            </Button>
          </div>
        </div>

        <div className="space-y-2">
          <div className="text-xs text-muted-foreground">Goal</div>
          <div className="flex flex-wrap gap-2">
            {GOALS.map(g => (
              <Button
                key={g.id}
                variant={goal === g.id ? 'default' : 'outline'}
                size="sm"
                onClick={() => setGoal(g.id)}
              >
                <span className="mr-2 text-muted-foreground">{g.icon}</span>
                {g.label}
              </Button>
            ))}
          </div>
        </div>

        <div className="space-y-2">
          <div className="text-xs text-muted-foreground">Mode</div>
          <div className="flex flex-wrap gap-2">
            {MODES.map(m => (
              <Button
                key={m.id}
                variant={mode === m.id ? 'default' : 'outline'}
                size="sm"
                onClick={() => setMode(m.id)}
              >
                {m.label}
              </Button>
            ))}
          </div>
        </div>

        <div className="flex items-center gap-2">
          <Input
            value={search}
            onChange={e => setSearch(e.target.value)}
            placeholder="Search presets…"
          />
          <Button
            variant="outline"
            onClick={() => {
              setGoal('all')
              setMode('all')
              setSearch('')
            }}
          >
            Reset
          </Button>
        </div>
      </Card>

      {recommended.length > 0 && (
        <div className="space-y-2">
          <div className="text-sm font-medium">Recommended</div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
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
      >
        {others.length === 0 ? (
          <div className="text-sm text-muted-foreground">No presets match your filters.</div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
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
  )
}
