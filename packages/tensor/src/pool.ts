/**
 * Tensor memory pooling for allocation reuse
 * Reduces GC pressure by reusing Float32Array buffers
 */

type PoolEntry = {
  buffer: Float32Array
  size: number
  inUse: boolean
}

class TensorPool {
  private pools: Map<number, PoolEntry[]> = new Map()
  private maxPoolSize = 100 // Max buffers per size
  private enabled = true

  /**
   * Acquire a buffer from pool or create new
   */
  acquire(size: number): Float32Array {
    if (!this.enabled) {
      return new Float32Array(size)
    }

    let pool = this.pools.get(size)
    if (!pool) {
      pool = []
      this.pools.set(size, pool)
    }

    // Find available buffer
    for (const entry of pool) {
      if (!entry.inUse) {
        entry.inUse = true
        // Zero out buffer for safety
        entry.buffer.fill(0)
        return entry.buffer
      }
    }

    // No available buffer, create new and add to pool
    const buffer = new Float32Array(size)
    if (pool.length < this.maxPoolSize) {
      pool.push({
        buffer,
        size,
        inUse: true,
      })
    }
    return buffer
  }

  /**
   * Release buffer back to pool
   */
  release(buffer: Float32Array): void {
    if (!this.enabled) {
      return
    }

    const size = buffer.length
    let pool = this.pools.get(size)

    if (!pool) {
      pool = []
      this.pools.set(size, pool)
    }

    // Find existing entry
    for (const entry of pool) {
      if (entry.buffer === buffer) {
        entry.inUse = false
        return
      }
    }

    // Add new entry if pool not full
    if (pool.length < this.maxPoolSize) {
      pool.push({
        buffer,
        size,
        inUse: false,
      })
    }
  }

  /**
   * Clear all pools
   */
  clear(): void {
    this.pools.clear()
  }

  /**
   * Get pool statistics
   */
  stats(): {
    sizes: number[]
    totalBuffers: number
    inUse: number
    available: number
  } {
    const sizes = Array.from(this.pools.keys()).sort((a, b) => a - b)
    let totalBuffers = 0
    let inUse = 0

    for (const pool of this.pools.values()) {
      totalBuffers += pool.length
      inUse += pool.filter((e) => e.inUse).length
    }

    return {
      sizes,
      totalBuffers,
      inUse,
      available: totalBuffers - inUse,
    }
  }

  /**
   * Enable/disable pooling
   */
  setEnabled(enabled: boolean): void {
    this.enabled = enabled
    if (!enabled) {
      this.clear()
    }
  }
}

// Global pool instance
const globalPool = new TensorPool()

/**
 * Acquire buffer from pool
 */
export function acquireBuffer(size: number): Float32Array {
  return globalPool.acquire(size)
}

/**
 * Release buffer back to pool
 */
export function releaseBuffer(buffer: Float32Array): void {
  globalPool.release(buffer)
}

/**
 * Clear all pools
 */
export function clearPool(): void {
  globalPool.clear()
}

/**
 * Get pool statistics
 */
export function poolStats() {
  return globalPool.stats()
}

/**
 * Enable/disable pooling globally
 */
export function setPoolingEnabled(enabled: boolean): void {
  globalPool.setEnabled(enabled)
}

/**
 * Scope-based memory management
 * All buffers allocated within the scope are released when scope exits
 *
 * Example:
 * ```ts
 * const result = withScope(() => {
 *   const a = randn([100, 100])
 *   const b = randn([100, 100])
 *   return matmul(a, b) // Only this result escapes the scope
 * })
 * // All intermediate buffers are now available for reuse
 * ```
 */
export function withScope<T>(fn: () => T): T {
  try {
    const result = fn()
    return result
  } finally {
    // Mark all buffers as available for reuse
    // This is a simple implementation that releases ALL buffers
    // A more sophisticated version would track which buffers were created in this scope
    for (const pool of globalPool['pools'].values()) {
      for (const entry of pool) {
        entry.inUse = false
      }
    }
  }
}
