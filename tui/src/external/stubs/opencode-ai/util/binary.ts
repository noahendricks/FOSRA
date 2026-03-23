// Stub for @opencode-ai/util/binary
type CompareFn<T> = (a: T, b: T) => number

export const Binary = {
  search: <T>(arr: T[], item: T, compare: CompareFn<T>): number => {
    let low = 0
    let high = arr.length - 1
    while (low <= high) {
      const mid = Math.floor((low + high) / 2)
      const cmp = compare(arr[mid], item)
      if (cmp < 0) low = mid + 1
      else if (cmp > 0) high = mid - 1
      else return mid
    }
    return -1
  },
  
  insert: <T>(arr: T[], item: T, compare: CompareFn<T>): number => {
    let low = 0
    let high = arr.length
    while (low < high) {
      const mid = Math.floor((low + high) / 2)
      const cmp = compare(arr[mid], item)
      if (cmp < 0) low = mid + 1
      else high = mid
    }
    arr.splice(low, 0, item)
    return low
  },
}