export class TtlCache<T> {
  private value: T | null = null
  private expiresAt = 0

  constructor(private readonly ttlMs: number) {}

  get(): T | null {
    if (this.expiresAt <= Date.now()) {
      this.value = null
      return null
    }
    return this.value
  }

  set(next: T) {
    this.value = next
    this.expiresAt = Date.now() + this.ttlMs
  }

  clear() {
    this.value = null
    this.expiresAt = 0
  }
}

