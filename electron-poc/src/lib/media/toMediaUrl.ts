export function toMediaUrl(filePath: string): string {
  if (/^(blob:|data:|https?:)/i.test(filePath)) return filePath
  const resolver = window.electronAPI?.resolveMediaUrl
  if (resolver) {
    const resolved = resolver(filePath)
    if (typeof resolved === "string" && resolved.trim()) return resolved
  }
  const normalized = String(filePath || "").replace(/\\/g, "/")
  return `media://${encodeURI(normalized)}`
}
