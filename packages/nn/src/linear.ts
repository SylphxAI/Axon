/**
 * Linear (Dense/Fully-Connected) layer - Pure functional implementation
 * PyTorch-like API but with pure functions
 */

import type { Tensor } from '@neuronline/tensor'
import { heNormal, uniform, matmul, add, transpose } from '@neuronline/tensor'

/**
 * Linear layer state (immutable)
 * Like nn.Linear in PyTorch but as data
 */
export type LinearState = {
  readonly weight: Tensor // [outFeatures, inFeatures]
  readonly bias: Tensor // [outFeatures]
}

/**
 * Initialize linear layer
 * Pure function - returns initial state
 *
 * Weight: He initialization (good for ReLU)
 * Bias: Uniform[-bound, bound] where bound = 1/sqrt(fan_in)
 */
export function init(inFeatures: number, outFeatures: number): LinearState {
  const bound = 1 / Math.sqrt(inFeatures)

  return {
    weight: heNormal([outFeatures, inFeatures], { requiresGrad: true }),
    bias: uniform([outFeatures], -bound, bound, { requiresGrad: true }),
  }
}

/**
 * Forward pass through linear layer
 * Pure function: input → output
 */
export function forward(input: Tensor, state: LinearState): Tensor {
  // input: [batch, inFeatures]
  // weight: [outFeatures, inFeatures]
  // output: [batch, outFeatures]

  // Transpose weight for correct dimensions
  const weightT = transpose(state.weight) // [inFeatures, outFeatures]
  const output = matmul(input, weightT) // [batch, outFeatures]
  return add(output, state.bias) // Broadcasting bias
}

/**
 * Get all parameters (for optimizer)
 * Pure function - returns array of tensors
 */
export function parameters(state: LinearState): Tensor[] {
  return [state.weight, state.bias]
}

/**
 * Update layer weights (from optimizer)
 * Pure function - returns new state
 */
export function updateWeights(
  state: LinearState,
  newWeight: Tensor,
  newBias: Tensor
): LinearState {
  return {
    weight: newWeight,
    bias: newBias,
  }
}
