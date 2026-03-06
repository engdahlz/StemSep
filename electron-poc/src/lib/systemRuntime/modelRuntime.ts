export function modelRequiresFnoRuntime(model: any): boolean {
  if (!model) return false
  const variant = String(model?.runtime?.variant || '').toLowerCase()
  const architecture = String(model?.architecture || '').toLowerCase()
  const id = String(model?.id || '').toLowerCase()
  const name = String(model?.name || '').toLowerCase()
  return (
    variant.includes('fno') ||
    architecture.includes('fno') ||
    id.includes('fno') ||
    name.includes('fno')
  )
}

