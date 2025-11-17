/**
 * Dropout layer - Pure functional implementation
 */

import type { Tensor } from '@sylphx/tensor'
import { mul, acquireBuffer } from '@sylphx/tensor'

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
  const mask = acquireBuffer(input.data.length)
  const scale = 1 / (1 - config.p)

  // Unroll by 8 for better performance
  let i = 0
  const len8 = mask.length - 7
  for (; i < len8; i += 8) {
    mask[i] = Math.random() > config.p ? scale : 0
    mask[i + 1] = Math.random() > config.p ? scale : 0
    mask[i + 2] = Math.random() > config.p ? scale : 0
    mask[i + 3] = Math.random() > config.p ? scale : 0
    mask[i + 4] = Math.random() > config.p ? scale : 0
    mask[i + 5] = Math.random() > config.p ? scale : 0
    mask[i + 6] = Math.random() > config.p ? scale : 0
    mask[i + 7] = Math.random() > config.p ? scale : 0
  }

  // Handle remainder
  for (; i < mask.length; i++) {
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
