/**
 * Batch Normalization layer - Pure functional implementation
 */

import type { Tensor } from '@neuronline/tensor'
import { ones, zeros, mul, add, scalar, acquireBuffer } from '@neuronline/tensor'

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

  const mean = acquireBuffer(numFeatures)
  const invNumSamples = 1.0 / numSamples

  // Loop unrolling for better performance
  for (let i = 0; i < numSamples; i++) {
    const rowOffset = i * numFeatures

    // Unroll by 8 for better ILP
    let j = 0
    const numFeatures8 = numFeatures - 7
    for (; j < numFeatures8; j += 8) {
      mean[j] = mean[j]! + input.data[rowOffset + j]! * invNumSamples
      mean[j + 1] = mean[j + 1]! + input.data[rowOffset + j + 1]! * invNumSamples
      mean[j + 2] = mean[j + 2]! + input.data[rowOffset + j + 2]! * invNumSamples
      mean[j + 3] = mean[j + 3]! + input.data[rowOffset + j + 3]! * invNumSamples
      mean[j + 4] = mean[j + 4]! + input.data[rowOffset + j + 4]! * invNumSamples
      mean[j + 5] = mean[j + 5]! + input.data[rowOffset + j + 5]! * invNumSamples
      mean[j + 6] = mean[j + 6]! + input.data[rowOffset + j + 6]! * invNumSamples
      mean[j + 7] = mean[j + 7]! + input.data[rowOffset + j + 7]! * invNumSamples
    }

    // Handle remainder
    for (; j < numFeatures; j++) {
      mean[j] = mean[j]! + input.data[rowOffset + j]! * invNumSamples
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
  const invNumSamples = 1.0 / numSamples

  const variance = acquireBuffer(numFeatures)

  // Loop unrolling for better performance
  for (let i = 0; i < numSamples; i++) {
    const rowOffset = i * numFeatures

    // Unroll by 8 for better ILP
    let j = 0
    const numFeatures8 = numFeatures - 7
    for (; j < numFeatures8; j += 8) {
      const diff0 = input.data[rowOffset + j]! - mean.data[j]!
      const diff1 = input.data[rowOffset + j + 1]! - mean.data[j + 1]!
      const diff2 = input.data[rowOffset + j + 2]! - mean.data[j + 2]!
      const diff3 = input.data[rowOffset + j + 3]! - mean.data[j + 3]!
      const diff4 = input.data[rowOffset + j + 4]! - mean.data[j + 4]!
      const diff5 = input.data[rowOffset + j + 5]! - mean.data[j + 5]!
      const diff6 = input.data[rowOffset + j + 6]! - mean.data[j + 6]!
      const diff7 = input.data[rowOffset + j + 7]! - mean.data[j + 7]!

      variance[j] = variance[j]! + diff0 * diff0 * invNumSamples
      variance[j + 1] = variance[j + 1]! + diff1 * diff1 * invNumSamples
      variance[j + 2] = variance[j + 2]! + diff2 * diff2 * invNumSamples
      variance[j + 3] = variance[j + 3]! + diff3 * diff3 * invNumSamples
      variance[j + 4] = variance[j + 4]! + diff4 * diff4 * invNumSamples
      variance[j + 5] = variance[j + 5]! + diff5 * diff5 * invNumSamples
      variance[j + 6] = variance[j + 6]! + diff6 * diff6 * invNumSamples
      variance[j + 7] = variance[j + 7]! + diff7 * diff7 * invNumSamples
    }

    // Handle remainder
    for (; j < numFeatures; j++) {
      const diff = input.data[rowOffset + j]! - mean.data[j]!
      variance[j] = variance[j]! + diff * diff * invNumSamples
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

  const normalized = acquireBuffer(input.data.length)

  // Pre-compute inverse standard deviations for better performance
  const invStd = acquireBuffer(numFeatures)
  for (let j = 0; j < numFeatures; j++) {
    invStd[j] = 1.0 / Math.sqrt(variance.data[j]! + epsilon)
  }

  // Loop unrolling for better performance
  for (let i = 0; i < numSamples; i++) {
    const rowOffset = i * numFeatures

    // Unroll by 8 for better ILP
    let j = 0
    const numFeatures8 = numFeatures - 7
    for (; j < numFeatures8; j += 8) {
      normalized[rowOffset + j] = (input.data[rowOffset + j]! - mean.data[j]!) * invStd[j]!
      normalized[rowOffset + j + 1] = (input.data[rowOffset + j + 1]! - mean.data[j + 1]!) * invStd[j + 1]!
      normalized[rowOffset + j + 2] = (input.data[rowOffset + j + 2]! - mean.data[j + 2]!) * invStd[j + 2]!
      normalized[rowOffset + j + 3] = (input.data[rowOffset + j + 3]! - mean.data[j + 3]!) * invStd[j + 3]!
      normalized[rowOffset + j + 4] = (input.data[rowOffset + j + 4]! - mean.data[j + 4]!) * invStd[j + 4]!
      normalized[rowOffset + j + 5] = (input.data[rowOffset + j + 5]! - mean.data[j + 5]!) * invStd[j + 5]!
      normalized[rowOffset + j + 6] = (input.data[rowOffset + j + 6]! - mean.data[j + 6]!) * invStd[j + 6]!
      normalized[rowOffset + j + 7] = (input.data[rowOffset + j + 7]! - mean.data[j + 7]!) * invStd[j + 7]!
    }

    // Handle remainder
    for (; j < numFeatures; j++) {
      normalized[rowOffset + j] = (input.data[rowOffset + j]! - mean.data[j]!) * invStd[j]!
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
