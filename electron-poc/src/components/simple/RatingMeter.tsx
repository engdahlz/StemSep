import { cn } from '@/lib/utils'

export function RatingMeter({
  label,
  value,
  max = 5,
  className,
}: {
  label: string
  value: number
  max?: number
  className?: string
}) {
  const v = Math.max(0, Math.min(max, value))
  const filled = Math.round(v)

  return (
    <div className={cn('flex items-center gap-2', className)}>
      <span className="w-12 shrink-0 text-[11px] text-slate-500">{label}</span>
      <div className="flex items-center gap-1">
        {Array.from({ length: max }).map((_, i) => (
          <span
            key={i}
            className={cn(
              'h-1.5 w-4 rounded-full transition-colors',
              i < filled ? 'bg-gradient-to-r from-slate-700 to-slate-500' : 'bg-white/72'
            )}
          />
        ))}
      </div>
    </div>
  )
}
