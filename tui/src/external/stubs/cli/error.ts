// Stub for @/cli/error
export function FormatError(error: Error): string {
  return error.message || String(error)
}

export const FormatUnknownError = (error: unknown): string => {
  if (error instanceof Error) return error.message
  if (typeof error === "string") return error
  return String(error)
}
