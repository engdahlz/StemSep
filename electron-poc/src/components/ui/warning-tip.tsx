import { AlertTriangle, Info } from 'lucide-react'
import { cn } from '../../lib/utils'

interface WarningTipProps {
    type: 'info' | 'warning' | 'danger'
    title?: string
    children: React.ReactNode
    className?: string
}

export function WarningTip({ type, title, children, className }: WarningTipProps) {
    const styles = {
        info: {
            bg: 'bg-blue-500/10 border-blue-500/30',
            icon: <Info className="h-4 w-4 text-blue-400 shrink-0" />,
            text: 'text-blue-200'
        },
        warning: {
            bg: 'bg-amber-500/10 border-amber-500/30',
            icon: <AlertTriangle className="h-4 w-4 text-amber-400 shrink-0" />,
            text: 'text-amber-200'
        },
        danger: {
            bg: 'bg-red-500/10 border-red-500/30',
            icon: <AlertTriangle className="h-4 w-4 text-red-400 shrink-0" />,
            text: 'text-red-200'
        }
    }

    const style = styles[type]

    return (
        <div className={cn(
            "flex gap-2 p-3 rounded-lg border text-xs",
            style.bg,
            style.text,
            className
        )}>
            {style.icon}
            <div>
                {title && <p className="font-medium mb-0.5">{title}</p>}
                <p className="leading-relaxed opacity-90">{children}</p>
            </div>
        </div>
    )
}

// Pre-built warning messages
export function TTAWarning() {
    return (
        <WarningTip type="warning" title="TTA Enabled">
            Test Time Augmentation processes audio multiple times with different augmentations,
            then averages results. This approximately doubles processing time but can improve
            quality on complex sources.
        </WarningTip>
    )
}

export function CPUOnlyWarning() {
    return (
        <WarningTip type="warning" title="CPU Processing">
            No GPU detected. Processing will use CPU only, which is significantly slower
            (typically 10-50x). Consider using a smaller segment size to reduce memory usage.
        </WarningTip>
    )
}

export function LowVRAMWarning({ available, required }: { available: number; required: number }) {
    return (
        <WarningTip type="danger" title="Insufficient VRAM">
            This configuration requires approximately {required.toFixed(1)}GB VRAM, but only
            {available.toFixed(1)}GB is available. Consider reducing chunk size, disabling TTA,
            or using a smaller model to avoid out-of-memory errors.
        </WarningTip>
    )
}

export function HighQualityTip() {
    return (
        <WarningTip type="info" title="Quality Tip">
            For best results with complex audio, try enabling TTA or using an ensemble preset.
            This increases processing time but significantly improves separation quality.
        </WarningTip>
    )
}

export function EnsembleTip() {
    return (
        <WarningTip type="info" title="Ensemble Mode">
            Ensemble presets combine multiple models to improve separation quality.
            Processing time scales linearly with the number of models used.
        </WarningTip>
    )
}
