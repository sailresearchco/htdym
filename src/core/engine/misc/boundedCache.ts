// A Map that forgets its oldest entry past `capacity`, for pure caches
// whose working set is small but whose key space is not (a sweep walks
// through meshes and batch shapes and rarely comes back). First in,
// first out: a hit costs one lookup, and an evicted entry that was still
// wanted is only recomputed.
export class BoundedCache<K, V> {
  private map = new Map<K, V>();
  constructor(private capacity: number) {}

  get(key: K): V | undefined {
    return this.map.get(key);
  }

  set(key: K, value: V): void {
    this.map.set(key, value);
    if (this.map.size > this.capacity) this.map.delete(this.map.keys().next().value as K);
  }
}
