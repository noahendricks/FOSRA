// Stub for @/util/locale
export const Locale = {
  pluralize: (count: number, singular: string, plural?: string) => 
    count === 1 ? singular : (plural ?? singular + "s"),
  titlecase: (str: string) => str.charAt(0).toUpperCase() + str.slice(1),
  bytes: (bytes: number) => {
    const units = ["B", "KB", "MB", "GB"]
    let i = 0
    while (bytes >= 1024 && i < units.length - 1) {
      bytes /= 1024
      i++
    }
    return `${bytes.toFixed(1)} ${units[i]}`
  },
  number: (n: number) => n.toLocaleString(),
  date: (d: Date) => d.toLocaleDateString(),
  time: (d: Date) => d.toLocaleTimeString(),
  datetime: (d: Date) => d.toLocaleString(),
  relativeTime: (d: Date) => {
    const diff = Date.now() - d.getTime()
    const seconds = Math.floor(diff / 1000)
    if (seconds < 60) return "just now"
    const minutes = Math.floor(seconds / 60)
    if (minutes < 60) return `${minutes}m ago`
    const hours = Math.floor(minutes / 60)
    if (hours < 24) return `${hours}h ago`
    const days = Math.floor(hours / 24)
    return `${days}d ago`
  },
}