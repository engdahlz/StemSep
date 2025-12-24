import * as React from "react"
import { Check, ChevronsUpDown, Cpu } from "lucide-react"

import { cn } from "@/lib/utils"
import { Button } from "./ui/button"
import { Badge } from "./ui/badge"

interface Model {
    id: string
    name: string
    architecture: string
    description?: string
    installed: boolean
    stems?: string[]
    vram_required?: number
}

interface ModelSelectorProps {
    selectedModelId: string
    onSelectModel: (modelId: string) => void
    models: Model[]
    className?: string
}

export function ModelSelector({
    selectedModelId,
    onSelectModel,
    models,
    className
}: ModelSelectorProps) {
    const [isOpen, setIsOpen] = React.useState(false)
    const [searchQuery, setSearchQuery] = React.useState("")
    const dropdownRef = React.useRef<HTMLDivElement>(null)

    const selectedModel = models.find((model) => model.id === selectedModelId)
    const installedModels = models.filter(m => m.installed)

    // Filter models by search query
    const filteredModels = installedModels.filter(model =>
        model.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
        model.architecture.toLowerCase().includes(searchQuery.toLowerCase())
    )

    // Group by architecture
    const groupedModels = React.useMemo(() => {
        const groups: Record<string, Model[]> = {}
        filteredModels.forEach(model => {
            const arch = model.architecture || 'Other'
            if (!groups[arch]) groups[arch] = []
            groups[arch].push(model)
        })
        return groups
    }, [filteredModels])

    // Close dropdown when clicking outside
    React.useEffect(() => {
        const handleClickOutside = (event: MouseEvent) => {
            if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
                setIsOpen(false)
            }
        }

        document.addEventListener("mousedown", handleClickOutside)
        return () => {
            document.removeEventListener("mousedown", handleClickOutside)
        }
    }, [])

    return (
        <div className={cn("relative", className)} ref={dropdownRef}>
            <Button
                variant="outline"
                role="combobox"
                aria-expanded={isOpen}
                onClick={() => setIsOpen(!isOpen)}
                className="w-full justify-between h-auto py-3 px-4 bg-background/50 backdrop-blur-sm border-border/50 hover:bg-accent/50"
            >
                <div className="flex flex-col items-start text-left gap-1 overflow-hidden w-full">
                    <div className="flex items-center justify-between w-full">
                        <span className="font-semibold truncate flex items-center gap-2">
                            {selectedModel ? selectedModel.name : "Choose a model..."}
                        </span>
                    </div>
                    {selectedModel && (
                        <div className="flex items-center gap-2 w-full">
                            <span className="text-xs text-muted-foreground truncate opacity-80 font-normal">
                                {selectedModel.architecture}
                            </span>
                            {selectedModel.stems && (
                                <div className="flex gap-1">
                                    {selectedModel.stems.slice(0, 3).map(stem => (
                                        <Badge key={stem} variant="secondary" className="text-[9px] h-3.5 px-1">
                                            {stem}
                                        </Badge>
                                    ))}
                                    {selectedModel.stems.length > 3 && (
                                        <Badge variant="secondary" className="text-[9px] h-3.5 px-1">
                                            +{selectedModel.stems.length - 3}
                                        </Badge>
                                    )}
                                </div>
                            )}
                        </div>
                    )}
                </div>
                <ChevronsUpDown className="ml-2 h-4 w-4 shrink-0 opacity-50" />
            </Button>

            {isOpen && (
                <div className="absolute top-full left-0 w-full mt-2 z-50 rounded-md border bg-popover text-popover-foreground shadow-md outline-none animate-in fade-in-0 zoom-in-95">
                    {/* Search Input */}
                    <div className="p-2 border-b">
                        <input
                            type="text"
                            placeholder="Search models..."
                            value={searchQuery}
                            onChange={(e) => setSearchQuery(e.target.value)}
                            className="w-full px-3 py-2 text-sm rounded-md border bg-background focus:outline-none focus:ring-2 focus:ring-primary"
                            autoFocus
                        />
                    </div>

                    <div className="max-h-[300px] overflow-y-auto p-1">
                        {filteredModels.length === 0 ? (
                            <div className="px-3 py-6 text-center text-sm text-muted-foreground">
                                No models found
                            </div>
                        ) : (
                            Object.entries(groupedModels).map(([arch, archModels]) => (
                                <div key={arch}>
                                    {/* Architecture Group Header */}
                                    <div className="px-2 py-1.5 text-xs font-semibold text-muted-foreground mt-1 flex items-center gap-1">
                                        <Cpu className="w-3 h-3" />
                                        {arch}
                                    </div>
                                    {archModels.map((model) => (
                                        <div
                                            key={model.id}
                                            onClick={() => {
                                                onSelectModel(model.id)
                                                setIsOpen(false)
                                                setSearchQuery("")
                                            }}
                                            className={cn(
                                                "relative flex cursor-pointer select-none items-center rounded-sm px-2 py-2 text-sm outline-none transition-colors hover:bg-accent hover:text-accent-foreground",
                                                selectedModelId === model.id && "bg-accent text-accent-foreground"
                                            )}
                                        >
                                            <div className="flex flex-col items-start gap-0.5 w-full">
                                                <div className="flex items-center justify-between w-full">
                                                    <span className="font-medium">{model.name}</span>
                                                    <div className="flex items-center gap-2">
                                                        {model.vram_required && (
                                                            <span className="text-[10px] text-muted-foreground">
                                                                {model.vram_required}GB
                                                            </span>
                                                        )}
                                                        {selectedModelId === model.id && (
                                                            <Check className="h-4 w-4 text-primary" />
                                                        )}
                                                    </div>
                                                </div>
                                                {model.stems && (
                                                    <div className="flex gap-1 mt-0.5">
                                                        {model.stems.map(stem => (
                                                            <Badge key={stem} variant="secondary" className="text-[9px] h-3.5 px-1">
                                                                {stem}
                                                            </Badge>
                                                        ))}
                                                    </div>
                                                )}
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            ))
                        )}
                    </div>
                </div>
            )}
        </div>
    )
}
