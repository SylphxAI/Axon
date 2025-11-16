/**
 * Batch Normalization layer - Pure functional implementation
 */

import type { Tensor } from '@neuronline/tensor'
import { ones, zeros, mul, add, sub, scalar } from '@neuronline/tensor'

/**
 * BatchNorm layer state
 */
export type BatchNormState = {
  readonly gamma: Tensor // Scale parameter
  readonly beta: Tensor // Shift parameter
  readonly runningMean: Tensor // Running mean for inference
  readonly runningVar: Tensor // Running variance for inference
  readonly momentum: number
  readonly epsilon: number
}

/**
 * Initialize BatchNorm layer
 */
export function init(numFeatures: number, momentum: number = 0.1, epsilon: number = 1e-5): BatchNormState {
  return {
    gamma: ones([numFeatures], { requiresGrad: true }),
    beta: zeros([numFeatures], { requiresGrad: true }),
    runningMean: zeros([numFeatures]),
    runningVar: ones([numFeatures]),
    momentum,
    epsilon,
  }
}

/**
 * Forward pass through BatchNorm
 *
 * Training mode: Normalize using batch statistics, update running stats
 * Inference mode: Normalize using running statistics
 */
export function forward(
  input: Tensor,
  state: BatchNormState,
  training: boolean = true
): { output: Tensor; state: BatchNormState } {
  if (training) {
    // Calculate batch mean and variance
    const batchMean = calculateMean(input)
    const batchVar = calculateVariance(input, batchMean)

    // Normalize
    const normalized = normalize(input, batchMean, batchVar, state.epsilon)

    // Scale and shift
    const output = scaleAndShift(normalized, state.gamma, state.beta)

    // Update running statistics
    const newRunningMean = updateRunningMean(
      state.runningMean,
      batchMean,
      state.momentum
    )
    const newRunningVar = updateRunningVar(
      state.runningVar,
      batchVar,
      state.momentum
    )

    const newState: BatchNormState = {
      ...state,
      runningMean: newRunningMean,
      runningVar: newRunningVar,
    }

    return { output, state: newState }
  } else {
    // Use running statistics for inference
    const normalized = normalize(
      input,
      state.runningMean,
      state.runningVar,
      state.epsilon
    )
    const output = scaleAndShift(normalized, state.gamma, state.beta)

    return { output, state }
  }
}

// Helper functions

function calculateMean(input: Tensor): Tensor {
  // Simplified: mean across batch dimension
  // Full implementation would handle different dimensions
  const numSamples = input.shape[0]!
  const numFeatures = input.data.length / numSamples

  const mean = new Float32Array(numFeatures)

  for (let i = 0; i < numSamples; i++) {
    for (let j = 0; j < numFeatures; j++) {
      mean[j] += input.data[i * numFeatures + j]! / numSamples
    }
  }

  return {
    data: mean,
    shape: [numFeatures],
    requiresGrad: false,
  }
}

function calculateVariance(input: Tensor, mean: Tensor): Tensor {
  const numSamples = input.shape[0]!
  const numFeatures = mean.data.length

  const variance = new Float32Array(numFeatures)

  for (let i = 0; i < numSamples; i++) {
    for (let j = 0; j < numFeatures; j++) {
      const diff = input.data[i * numFeatures + j]! - mean.data[j]!
      variance[j] += (diff * diff) / numSamples
    }
  }

  return {
    data: variance,
    shape: [numFeatures],
    requiresGrad: false,
  }
}

function normalize(
  input: Tensor,
  mean: Tensor,
  variance: Tensor,
  epsilon: number
): Tensor {
  const numSamples = input.shape[0]!
  const numFeatures = mean.data.length

  const normalized = new Float32Array(input.data.length)

  for (let i = 0; i < numSamples; i++) {
    for (let j = 0; j < numFeatures; j++) {
      const idx = i * numFeatures + j
      normalized[idx] =
        (input.data[idx]! - mean.data[j]!) /
        Math.sqrt(variance.data[j]! + epsilon)
    }
  }

  return {
    data: normalized,
    shape: input.shape,
    requiresGrad: input.requiresGrad,
  }
}

function scaleAndShift(input: Tensor, gamma: Tensor, beta: Tensor): Tensor {
  // output = gamma * input + beta
  const scaled = mul(input, gamma)
  return add(scaled, beta)
}

function updateRunningMean(
  runningMean: Tensor,
  batchMean: Tensor,
  momentum: number
): Tensor {
  // running_mean = (1 - momentum) * running_mean + momentum * batch_mean
  const keepOld = mul(runningMean, scalar(1 - momentum))
  const addNew = mul(batchMean, scalar(momentum))
  return add(keepOld, addNew)
}

function updateRunningVar(
  runningVar: Tensor,
  batchVar: Tensor,
  momentum: number
): Tensor {
  // running_var = (1 - momentum) * running_var + momentum * batch_var
  const keepOld = mul(runningVar, scalar(1 - momentum))
  const addNew = mul(batchVar, scalar(momentum))
  return add(keepOld, addNew)
}
