// Stub for @/util/locale
export const Locale = {
  pluralize: (count: number, singular: string, plural?: string) => 
    count === 1 ? singular : (plural ?? singular + "s"),
  titlecase: (str: string) => str.charAt(0).toUpperCase() + str.slice(1),
  truncate: (str: string, maxLength: number, suffix = "…") => {
    if (str.length <= maxLength) return str
    return str.slice(0, maxLength - suffix.length) + suffix
  },
  truncateMiddle: (str: string, maxLength: number, suffix = "…") => {
    if (str.length <= maxLength) return str
    const half = Math.floor((maxLength - suffix.length) / 2)
    return str.slice(0, half) + suffix + str.slice(-half)
  },
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
  date: (d: Date | number) => new Date(d).toLocaleDateString(),
  time: (d: Date | number) => new Date(d).toLocaleTimeString(),
  datetime: (d: Date | number) => new Date(d).toLocaleString(),
  todayTimeOrDateTime: (d: Date | number) => {
    const date = new Date(d)
    const now = new Date()
    if (date.getDate() === now.getDate() && date.getMonth() === now.getMonth() && date.getFullYear() === now.getFullYear()) {
      return date.toLocaleTimeString()
    }
    return date.toLocaleString()
  },
  duration: (ms: number) => {
    const seconds = Math.floor(ms / 1000)
    if (seconds < 60) return `${seconds}s`
    const minutes = Math.floor(seconds / 60)
    if (minutes < 60) return `${minutes}m ${seconds % 60}s`
    const hours = Math.floor(minutes / 60)
    return `${hours}h ${minutes % 60}m`
  },
  relativeTime: (d: Date | number) => {
    const date = new Date(d)
    const diff = Date.now() - date.getTime()
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
