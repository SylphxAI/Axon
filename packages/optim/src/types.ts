/**
 * Pure functional optimizer types
 * PyTorch-like API but with pure functions
 */

import type { Tensor } from '@neuronline/tensor'

/**
 * Optimizer state (immutable)
 * Contains internal state for optimizers (momentum, adaptive learning rates, etc.)
 */
export type OptimizerState = {
  readonly step: number
  readonly params: readonly Tensor[]
  readonly state: Record<string, unknown>
}

/**
 * Optimizer configuration
 */
export type OptimizerConfig = {
  readonly lr: number // Learning rate
  readonly [key: string]: unknown // Optimizer-specific config
}

/**
 * Update result from optimizer
 */
export type UpdateResult = {
  readonly params: readonly Tensor[]
  readonly state: OptimizerState
}
