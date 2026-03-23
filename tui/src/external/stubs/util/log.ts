// Stub for @/util/log
export const Log = {
  Default: {
    error: (...args: unknown[]) => console.error("[ERROR]", ...args),
    warn: (...args: unknown[]) => console.warn("[WARN]", ...args),
    info: (...args: unknown[]) => console.info("[INFO]", ...args),
    debug: (...args: unknown[]) => console.debug("[DEBUG]", ...args),
  },
}