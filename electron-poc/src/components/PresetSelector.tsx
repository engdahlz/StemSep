import * as React from "react"
import { Check, ChevronsUpDown, Sparkles, DownloadCloud } from "lucide-react"

import { cn } from "@/lib/utils"
import { Button } from "./ui/button"
import { Preset } from "../presets"
import { Badge } from "./ui/badge"

interface PresetSelectorProps {
    selectedPresetId: string
    onSelectPreset: (presetId: string) => void
    presets: Preset[]
    className?: string
    availability?: Record<string, { available: boolean }>
}

export function PresetSelector({
    selectedPresetId,
    onSelectPreset,
    presets,
    className,
    availability
}: PresetSelectorProps) {
    const [isOpen, setIsOpen] = React.useState(false)
    const dropdownRef = React.useRef<HTMLDivElement>(null)

    const selectedPreset = presets.find((preset) => preset.id === selectedPresetId)

    // Check if preset models are installed
    const isInstalled = (preset: Preset) => {
        if (!availability) return true
        if (preset.ensembleConfig) {
            return preset.ensembleConfig.models.every(m => availability[m.model_id]?.available !== false)
        }
        return preset.modelId ? availability[preset.modelId]?.available !== false : true
    }

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
                            {selectedPreset ? selectedPreset.name : "Select preset..."}
                            {selectedPreset?.isRecipe && (
                                <Badge variant="outline" className="text-[10px] h-4 px-1 bg-yellow-500/10 text-yellow-500 border-yellow-500/20 flex gap-1">
                                    <Sparkles className="w-2 h-2" /> Smart
                                </Badge>
                            )}
                            {!selectedPreset?.isRecipe && selectedPreset?.ensembleConfig && (
                                <Badge variant="secondary" className="text-[10px] h-4 px-1 bg-purple-500/10 text-purple-500 border-purple-500/20">Ensemble</Badge>
                            )}
                        </span>
                        {selectedPreset && !isInstalled(selectedPreset) && <DownloadCloud className="h-4 w-4 text-muted-foreground" />}
                    </div>
                    {selectedPreset && (
                        <span className="text-xs text-muted-foreground truncate w-full opacity-80 font-normal">
                            {selectedPreset.description}
                        </span>
                    )}
                </div>
                <ChevronsUpDown className="ml-2 h-4 w-4 shrink-0 opacity-50" />
            </Button>

            {isOpen && (
                <div className="absolute top-full left-0 w-full mt-2 z-50 rounded-md border bg-popover text-popover-foreground shadow-md outline-none animate-in fade-in-0 zoom-in-95">
                    <div className="max-h-[300px] overflow-y-auto p-1">
                        {/* Smart Recipes Group */}
                        {presets.some(p => p.isRecipe) && (
                            <div className="px-2 py-1.5 text-xs font-semibold text-muted-foreground mb-1 flex items-center gap-1">
                                <Sparkles className="w-3 h-3 text-yellow-500" /> Smart Recipes
                            </div>
                        )}
                        {presets.filter(p => p.isRecipe).map((preset) => (
                            <div
                                key={preset.id}
                                onClick={() => {
                                    onSelectPreset(preset.id)
                                    setIsOpen(false)
                                }}
                                className={cn(
                                    "relative flex cursor-default select-none items-center rounded-sm px-2 py-1.5 text-sm outline-none hover:bg-accent hover:text-accent-foreground data-[disabled]:pointer-events-none data-[disabled]:opacity-50",
                                    selectedPresetId === preset.id && "bg-accent text-accent-foreground"
                                )}
                            >
                                <div className="flex flex-col items-start gap-1 w-full">
                                    <div className="flex items-center justify-between w-full">
                                        <span className="font-medium flex items-center gap-2 text-yellow-500">
                                            <Sparkles className="w-3 h-3" />
                                            {preset.name}
                                        </span>
                                        <div className="flex items-center gap-2">
                                            {!isInstalled(preset) && <span title="Requires download"><DownloadCloud className="h-3 w-3 text-muted-foreground" /></span>}
                                            {selectedPresetId === preset.id && (
                                                <Check className="h-4 w-4 text-primary" />
                                            )}
                                        </div>
                                    </div>
                                    <span className="text-xs text-muted-foreground line-clamp-1">
                                        {preset.description}
                                    </span>
                                </div>
                            </div>
                        ))}

                        {/* Ensembles Group */}
                        {presets.some(p => p.ensembleConfig && !p.isRecipe) && (
                            <div className="px-2 py-1.5 text-xs font-semibold text-muted-foreground mt-2">
                                Ensembles
                            </div>
                        )}
                        {presets.filter(p => p.ensembleConfig && !p.isRecipe).map((preset) => (
                            <div
                                key={preset.id}
                                onClick={() => {
                                    onSelectPreset(preset.id)
                                    setIsOpen(false)
                                }}
                                className={cn(
                                    "relative flex cursor-default select-none items-center rounded-sm px-2 py-1.5 text-sm outline-none hover:bg-accent hover:text-accent-foreground data-[disabled]:pointer-events-none data-[disabled]:opacity-50",
                                    selectedPresetId === preset.id && "bg-accent text-accent-foreground"
                                )}
                            >
                                <div className="flex flex-col items-start gap-1 w-full">
                                    <div className="flex items-center justify-between w-full">
                                        <span className="font-medium flex items-center gap-2">
                                            {preset.name}
                                            <Badge variant="secondary" className="text-[10px] h-4 px-1 bg-purple-500/10 text-purple-500 border-purple-500/20">Ensemble</Badge>
                                        </span>
                                        <div className="flex items-center gap-2">
                                            {!isInstalled(preset) && <span title="Requires download"><DownloadCloud className="h-3 w-3 text-muted-foreground" /></span>}
                                            {selectedPresetId === preset.id && (
                                                <Check className="h-4 w-4 text-primary" />
                                            )}
                                        </div>
                                    </div>
                                    <span className="text-xs text-muted-foreground line-clamp-1">
                                        {preset.description}
                                    </span>
                                </div>
                            </div>
                        ))}

                        {/* Standard Models Group */}
                        {presets.some(p => !p.ensembleConfig && !p.isRecipe) && (
                            <div className="px-2 py-1.5 text-xs font-semibold text-muted-foreground mt-2">
                                Standard Models
                            </div>
                        )}
                        {presets.filter(p => !p.ensembleConfig && !p.isRecipe).map((preset) => (
                            <div
                                key={preset.id}
                                onClick={() => {
                                    onSelectPreset(preset.id)
                                    setIsOpen(false)
                                }}
                                className={cn(
                                    "relative flex cursor-default select-none items-center rounded-sm px-2 py-1.5 text-sm outline-none hover:bg-accent hover:text-accent-foreground data-[disabled]:pointer-events-none data-[disabled]:opacity-50",
                                    selectedPresetId === preset.id && "bg-accent text-accent-foreground"
                                )}
                            >
                                <div className="flex flex-col items-start gap-1 w-full">
                                    <div className="flex items-center justify-between w-full">
                                        <span className="font-medium">{preset.name}</span>
                                        <div className="flex items-center gap-2">
                                            {!isInstalled(preset) && <span title="Requires download"><DownloadCloud className="h-3 w-3 text-muted-foreground" /></span>}
                                            {selectedPresetId === preset.id && (
                                                <Check className="h-4 w-4 text-primary" />
                                            )}
                                        </div>
                                    </div>
                                    <span className="text-xs text-muted-foreground line-clamp-1">
                                        {preset.description}
                                    </span>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            )}
        </div>
    )
}
