// Stub for @/cli/error
export class FormatError extends Error {
  constructor(message: string) {
    super(message)
    this.name = "FormatError"
  }
}

export const FormatUnknownError = (error: unknown): string => {
  if (error instanceof Error) return error.message
  if (typeof error === "string") return error
  return String(error)
}