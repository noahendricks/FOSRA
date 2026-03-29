// stub for @/util/defer
// async disposable pattern for cleanup
export function defer(fn: () => Promise<void>): AsyncDisposable {
  return {
    [Symbol.asyncDispose]: fn,
  }
}
