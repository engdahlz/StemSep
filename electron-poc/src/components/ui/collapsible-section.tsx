import { useState } from 'react'
import { ChevronDown, ChevronRight } from 'lucide-react'
import { cn } from '../../lib/utils'

interface CollapsibleSectionProps {
    title: string
    icon?: React.ReactNode
    defaultOpen?: boolean
    children: React.ReactNode
    className?: string
    badge?: React.ReactNode
}

export function CollapsibleSection({
    title,
    icon,
    defaultOpen = false,
    children,
    className,
    badge
}: CollapsibleSectionProps) {
    const [isOpen, setIsOpen] = useState(defaultOpen)

    return (
        <div className={cn("rounded-lg border border-border overflow-hidden", className)}>
            <button
                onClick={() => setIsOpen(!isOpen)}
                className={cn(
                    "w-full flex items-center justify-between px-4 py-3 text-left",
                    "bg-secondary/30 hover:bg-secondary/50 transition-colors",
                    isOpen && "border-b border-border"
                )}
            >
                <div className="flex items-center gap-2">
                    {icon && <span className="text-muted-foreground">{icon}</span>}
                    <span className="text-sm font-medium">{title}</span>
                    {badge}
                </div>
                {isOpen ? (
                    <ChevronDown className="h-4 w-4 text-muted-foreground" />
                ) : (
                    <ChevronRight className="h-4 w-4 text-muted-foreground" />
                )}
            </button>

            <div className={cn(
                "transition-all duration-200 ease-in-out overflow-hidden",
                isOpen ? "max-h-[1000px] opacity-100" : "max-h-0 opacity-0"
            )}>
                <div className="p-4 space-y-4 bg-background/50">
                    {children}
                </div>
            </div>
        </div>
    )
}
