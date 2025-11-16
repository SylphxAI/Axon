/**
 * Dropout layer - Pure functional implementation
 */

import type { Tensor } from '@neuronline/tensor'
import { mul, scalar } from '@neuronline/tensor'

/**
 * Dropout configuration
 */
export type DropoutConfig = {
  readonly p: number // Dropout probability
  readonly training: boolean // Training mode
}

/**
 * Forward pass through Dropout
 * During training: randomly zero out elements with probability p
 * During inference: return input unchanged (scaled during training)
 */
export function forward(input: Tensor, config: DropoutConfig): Tensor {
  if (!config.training || config.p === 0) {
    return input
  }

  // Create dropout mask
  const mask = new Float32Array(input.data.length)
  const scale = 1 / (1 - config.p)

  for (let i = 0; i < mask.length; i++) {
    mask[i] = Math.random() > config.p ? scale : 0
  }

  // Apply mask
  const maskTensor: Tensor = {
    data: mask,
    shape: input.shape,
    requiresGrad: false,
  }

  return mul(input, maskTensor)
}
