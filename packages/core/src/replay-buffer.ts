import type { ReplayBuffer, TrainingExample } from './types'

export class PrioritizedReplayBuffer implements ReplayBuffer {
  private buffer: TrainingExample[] = []
  private priorities: Float32Array
  private maxSize: number
  private alpha: number
  private position = 0

  constructor(maxSize = 1000, alpha = 0.6) {
    this.maxSize = maxSize
    this.alpha = alpha
    this.priorities = new Float32Array(maxSize)
  }

  add(example: TrainingExample): void {
    const priority = Math.max(...this.priorities, 1.0)

    if (this.buffer.length < this.maxSize) {
      this.buffer.push(example)
      this.priorities[this.buffer.length - 1] = priority ** this.alpha
    } else {
      this.buffer[this.position] = example
      this.priorities[this.position] = priority ** this.alpha
      this.position = (this.position + 1) % this.maxSize
    }
  }

  sample(n: number): TrainingExample[] {
    const size = this.buffer.length
    if (size === 0) return []

    const sampleSize = Math.min(n, size)
    const samples: TrainingExample[] = []

    const totalPriority = this.priorities.slice(0, size).reduce((a, b) => a + b, 0)

    for (let i = 0; i < sampleSize; i++) {
      let rand = Math.random() * totalPriority
      let index = 0

      for (let j = 0; j < size; j++) {
        rand -= this.priorities[j]!
        if (rand <= 0) {
          index = j
          break
        }
      }

      samples.push(this.buffer[index]!)
    }

    return samples
  }

  updatePriority(index: number, priority: number): void {
    if (index >= 0 && index < this.buffer.length) {
      this.priorities[index] = priority ** this.alpha
    }
  }

  size(): number {
    return this.buffer.length
  }

  clear(): void {
    this.buffer = []
    this.priorities = new Float32Array(this.maxSize)
    this.position = 0
  }
}

export class UniformReplayBuffer implements ReplayBuffer {
  private buffer: TrainingExample[] = []
  private maxSize: number
  private position = 0

  constructor(maxSize = 1000) {
    this.maxSize = maxSize
  }

  add(example: TrainingExample): void {
    if (this.buffer.length < this.maxSize) {
      this.buffer.push(example)
    } else {
      this.buffer[this.position] = example
      this.position = (this.position + 1) % this.maxSize
    }
  }

  sample(n: number): TrainingExample[] {
    const size = this.buffer.length
    if (size === 0) return []

    const sampleSize = Math.min(n, size)
    const samples: TrainingExample[] = []
    const indices = new Set<number>()

    while (indices.size < sampleSize) {
      indices.add(Math.floor(Math.random() * size))
    }

    for (const idx of indices) {
      samples.push(this.buffer[idx]!)
    }

    return samples
  }

  size(): number {
    return this.buffer.length
  }

  clear(): void {
    this.buffer = []
    this.position = 0
  }
}
