/**
 * Linear (Dense/Fully-Connected) layer - Pure functional v2
 */

import type { Tensor } from '@sylphx/tensor'
import { heNormal, uniform, matmul, add, transpose } from '@sylphx/tensor'
import type { Layer } from './types'

/**
 * Linear layer state
 */
export type LinearState = {
  readonly weight: Tensor  // [outFeatures, inFeatures]
  readonly bias: Tensor    // [outFeatures]
}

/**
 * Linear layer constructor
 * Returns { init, forward } pair
 *
 * @param inFeatures - Input dimension
 * @param outFeatures - Output dimension
 * @returns Layer with init and forward functions
 *
 * @example
 * const layer = Linear(2, 8)
 * const state = layer.init()
 * const output = layer.forward(input, state)
 */
export const Linear = (
  inFeatures: number,
  outFeatures: number
): Layer<LinearState> => ({
  /**
   * Initialize layer state
   * Weight: He initialization (good for ReLU)
   * Bias: Uniform[-bound, bound] where bound = 1/sqrt(fan_in)
   */
  init: () => {
    const bound = 1 / Math.sqrt(inFeatures)

    return {
      weight: heNormal([outFeatures, inFeatures], { requiresGrad: true }),
      bias: uniform([outFeatures], -bound, bound, { requiresGrad: true }),
    }
  },

  /**
   * Forward pass
   * output = input @ weight.T + bias
   */
  forward: (input: Tensor, state: LinearState): Tensor => {
    // input: [batch, inFeatures]
    // weight: [outFeatures, inFeatures]
    // output: [batch, outFeatures]

    const weightT = transpose(state.weight)  // [inFeatures, outFeatures]
    const output = matmul(input, weightT)    // [batch, outFeatures]
    return add(output, state.bias)           // Broadcasting bias
  }
})
