import { useState, useEffect } from 'react'
import { Mic2, Music2, Disc3, ChevronRight } from 'lucide-react'

interface Workflow {
    description: string
    strategy: string
    pre_processing: string[]
    model_priority: string
    post_processing: string[]
    recommended_models: string[]
    chain_order?: string[]
    notes?: string[]
}

interface WorkflowSelectorProps {
    onSelect: (workflowType: 'live' | 'studio', workflow: Workflow) => void
    selectedWorkflow?: 'live' | 'studio' | null
}

export function WorkflowSelector({ onSelect, selectedWorkflow }: WorkflowSelectorProps) {
    const [workflows, setWorkflows] = useState<Record<string, Workflow>>({})
    const [loading, setLoading] = useState(true)

    useEffect(() => {
        const loadWorkflows = async () => {
            try {
                const result = await window.electronAPI.getWorkflows()
                if (result?.workflows) {
                    setWorkflows(result.workflows)
                }
            } catch (error) {
                console.error('Failed to load workflows:', error)
            } finally {
                setLoading(false)
            }
        }
        loadWorkflows()
    }, [])

    if (loading) {
        return (
            <div className="flex items-center justify-center p-8">
                <div className="animate-spin h-6 w-6 border-2 border-primary border-t-transparent rounded-full" />
            </div>
        )
    }

    const workflowCards = [
        {
            type: 'live' as const,
            title: 'Live / Restoration',
            icon: Mic2,
            gradient: 'from-orange-500/20 to-red-500/20',
            border: 'border-orange-500/30',
            accent: 'text-orange-400',
        },
        {
            type: 'studio' as const,
            title: 'Studio / Production',
            icon: Music2,
            gradient: 'from-blue-500/20 to-purple-500/20',
            border: 'border-blue-500/30',
            accent: 'text-blue-400',
        },
    ]

    return (
        <div className="space-y-4">
            <div className="flex items-center gap-2 text-sm text-neutral-400">
                <Disc3 className="w-4 h-4" />
                <span>What type of audio are you working with?</span>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {workflowCards.map((card) => {
                    const workflow = workflows[card.type]
                    const isSelected = selectedWorkflow === card.type
                    const Icon = card.icon

                    return (
                        <button
                            key={card.type}
                            onClick={() => workflow && onSelect(card.type, workflow)}
                            className={`
                relative p-5 rounded-xl border-2 text-left transition-all duration-200
                bg-gradient-to-br ${card.gradient}
                ${isSelected ? `${card.border} ring-2 ring-offset-2 ring-offset-neutral-900` : 'border-neutral-700 hover:border-neutral-600'}
                ${isSelected ? `ring-${card.type === 'live' ? 'orange' : 'blue'}-500/50` : ''}
              `}
                        >
                            <div className="flex items-start justify-between">
                                <div className="flex items-center gap-3">
                                    <div className={`p-2 rounded-lg bg-neutral-800/50 ${card.accent}`}>
                                        <Icon className="w-5 h-5" />
                                    </div>
                                    <div>
                                        <h3 className="font-semibold text-white">{card.title}</h3>
                                        <p className="text-sm text-neutral-400 mt-0.5">
                                            {workflow?.description || 'Loading...'}
                                        </p>
                                    </div>
                                </div>
                                <ChevronRight className={`w-5 h-5 transition-transform ${isSelected ? 'text-white translate-x-1' : 'text-neutral-500'}`} />
                            </div>

                            {workflow && (
                                <div className="mt-4 space-y-2">
                                    <div className="flex flex-wrap gap-1.5">
                                        {workflow.recommended_models.slice(0, 2).map((model) => (
                                            <span
                                                key={model}
                                                className="px-2 py-0.5 text-xs rounded-full bg-neutral-800/80 text-neutral-300 border border-neutral-700"
                                            >
                                                {model}
                                            </span>
                                        ))}
                                    </div>
                                    <p className={`text-xs ${card.accent}`}>
                                        Strategy: {workflow.strategy.replace('_', ' ')}
                                    </p>
                                </div>
                            )}

                            {isSelected && (
                                <div className="absolute top-2 right-2">
                                    <div className={`w-2 h-2 rounded-full ${card.type === 'live' ? 'bg-orange-400' : 'bg-blue-400'} animate-pulse`} />
                                </div>
                            )}
                        </button>
                    )
                })}
            </div>
        </div>
    )
}
