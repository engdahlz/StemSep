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
        <div
            className={cn(
                "overflow-hidden rounded-[1.35rem] border border-white/55 bg-[rgba(255,255,255,0.44)] shadow-[0_20px_60px_rgba(141,150,179,0.14)] backdrop-blur-xl",
                className
            )}
        >
            <button
                onClick={() => setIsOpen(!isOpen)}
                className={cn(
                    "stemsep-config-row flex w-full items-center justify-between px-4 py-3.5 text-left transition-all duration-300",
                    isOpen && "border-b border-white/45 bg-white/46"
                )}
            >
                <div className="flex items-center gap-2">
                    {icon && <span className="text-slate-500">{icon}</span>}
                    <span className="text-sm font-medium text-slate-800">{title}</span>
                    {badge}
                </div>
                {isOpen ? (
                    <ChevronDown className="h-4 w-4 text-slate-500" />
                ) : (
                    <ChevronRight className="h-4 w-4 text-slate-500" />
                )}
            </button>

            <div className={cn(
                "transition-all duration-200 ease-in-out overflow-hidden",
                isOpen ? "max-h-[1000px] opacity-100" : "max-h-0 opacity-0"
            )}>
                <div className="space-y-4 bg-white/20 p-4">
                    {children}
                </div>
            </div>
        </div>
    )
}
