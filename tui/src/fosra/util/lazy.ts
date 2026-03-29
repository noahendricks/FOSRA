// stub for @/util/lazy
export function lazy<T>(fn: () => T): () => T {
  let value: T | undefined
  let computed = false
  return () => {
    if (!computed) {
      value = fn()
      computed = true
    }
    return value!
  }
}
