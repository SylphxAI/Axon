/**
 * Dropout layer - Pure functional implementation
 */

import type { Tensor } from '@sylphx/tensor'
import { mul, acquireBuffer } from '@sylphx/tensor'
import type { Layer } from './types'

export type DropoutConfig = {
  readonly p: number // Dropout probability
}

export type DropoutState = {
  readonly p: number
}

/**
 * Dropout layer factory
 * During training: randomly zero out elements with probability p
 * During inference: return input unchanged (scaled during training)
 */
export const Dropout = (p: number): Layer<DropoutState> => ({
  init: (): DropoutState => ({ p }),

  forward: (input: Tensor, state: DropoutState): Tensor => {
    // For inference, return input unchanged
    // In production, you'd pass a training flag
    // For now, simplified version that always applies dropout
    if (state.p === 0) {
      return input
    }

    // Create dropout mask
    const mask = acquireBuffer(input.data.length)
    const scale = 1 / (1 - state.p)

    // Unroll by 8 for better performance
    let i = 0
    const len8 = mask.length - 7
    for (; i < len8; i += 8) {
      mask[i] = Math.random() > state.p ? scale : 0
      mask[i + 1] = Math.random() > state.p ? scale : 0
      mask[i + 2] = Math.random() > state.p ? scale : 0
      mask[i + 3] = Math.random() > state.p ? scale : 0
      mask[i + 4] = Math.random() > state.p ? scale : 0
      mask[i + 5] = Math.random() > state.p ? scale : 0
      mask[i + 6] = Math.random() > state.p ? scale : 0
      mask[i + 7] = Math.random() > state.p ? scale : 0
    }

    // Handle remainder
    for (; i < mask.length; i++) {
      mask[i] = Math.random() > state.p ? scale : 0
    }

    // Apply mask
    const maskTensor: Tensor = {
      data: mask,
      shape: input.shape,
      requiresGrad: false,
    }

    return mul(input, maskTensor)
  },
})
