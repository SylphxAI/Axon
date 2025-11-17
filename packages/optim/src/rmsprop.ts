/**
 * RMSprop optimizer
 * Pure functional implementation
 */

import type { Tensor } from '@sylphx/tensor'
import { sub, mul, add, scalar, zeros, acquireBuffer } from '@sylphx/tensor'
import type { OptimizerState, UpdateResult } from './types'

export type RMSpropConfig = {
  readonly lr: number
  readonly alpha?: number // Smoothing constant (default: 0.99)
  readonly epsilon?: number // Numerical stability (default: 1e-8)
  readonly weightDecay?: number
}

/**
 * Initialize RMSprop optimizer state
 */
export function init(params: readonly Tensor[], config: RMSpropConfig): OptimizerState {
  // Initialize squared gradient average
  const sqAvg = params.map((p) => zeros(p.shape))

  return {
    step: 0,
    params,
    state: {
      config,
      sqAvg,
    },
  }
}

/**
 * Perform RMSprop update step
 */
export function step(
  optState: OptimizerState,
  params: readonly Tensor[],
  grads: Map<Tensor, Tensor>
): UpdateResult {
  const config = optState.state.config as RMSpropConfig
  const { lr, alpha = 0.99, epsilon = 1e-8, weightDecay = 0 } = config

  const sqAvg = optState.state.sqAvg as Tensor[]

  const newParams: Tensor[] = []
  const newSqAvg: Tensor[] = []

  for (let i = 0; i < params.length; i++) {
    const param = params[i]!
    const grad = grads.get(param)

    if (!grad) {
      newParams.push(param)
      newSqAvg.push(sqAvg[i]!)
      continue
    }

    // Weight decay
    let dParam = grad
    if (weightDecay !== 0) {
      dParam = add(dParam, mul(param, scalar(weightDecay)))
    }

    // Update squared gradient average: v_t = alpha * v_{t-1} + (1 - alpha) * grad^2
    const gradSquared = elementwiseSquare(dParam)
    const newV = add(
      mul(sqAvg[i]!, scalar(alpha)),
      mul(gradSquared, scalar(1 - alpha))
    )
    newSqAvg.push(newV)

    // Update: param = param - lr * grad / (sqrt(v_t) + epsilon)
    const denominator = elementwiseSqrtPlusEpsilon(newV, epsilon)
    const update = elementwiseDivide(dParam, denominator)
    const newParam = sub(param, mul(update, scalar(lr)))
    newParams.push(newParam)
  }

  const newState: OptimizerState = {
    step: optState.step + 1,
    params: newParams,
    state: {
      config,
      sqAvg: newSqAvg,
    },
  }

  return {
    params: newParams,
    state: newState,
  }
}

// Helper functions

function elementwiseSquare(t: Tensor): Tensor {
  const data = acquireBuffer(t.data.length)

  // Unroll by 8 for better performance
  let i = 0
  const len8 = t.data.length - 7
  for (; i < len8; i += 8) {
    const v0 = t.data[i]!
    const v1 = t.data[i + 1]!
    const v2 = t.data[i + 2]!
    const v3 = t.data[i + 3]!
    const v4 = t.data[i + 4]!
    const v5 = t.data[i + 5]!
    const v6 = t.data[i + 6]!
    const v7 = t.data[i + 7]!
    data[i] = v0 * v0
    data[i + 1] = v1 * v1
    data[i + 2] = v2 * v2
    data[i + 3] = v3 * v3
    data[i + 4] = v4 * v4
    data[i + 5] = v5 * v5
    data[i + 6] = v6 * v6
    data[i + 7] = v7 * v7
  }

  // Handle remainder
  for (; i < t.data.length; i++) {
    data[i] = t.data[i]! * t.data[i]!
  }

  return { ...t, data, requiresGrad: false }
}

function elementwiseSqrtPlusEpsilon(t: Tensor, epsilon: number): Tensor {
  const data = acquireBuffer(t.data.length)

  // Unroll by 4 for sqrt (transcendental function)
  let i = 0
  const len4 = t.data.length - 3
  for (; i < len4; i += 4) {
    data[i] = Math.sqrt(t.data[i]!) + epsilon
    data[i + 1] = Math.sqrt(t.data[i + 1]!) + epsilon
    data[i + 2] = Math.sqrt(t.data[i + 2]!) + epsilon
    data[i + 3] = Math.sqrt(t.data[i + 3]!) + epsilon
  }

  // Handle remainder
  for (; i < t.data.length; i++) {
    data[i] = Math.sqrt(t.data[i]!) + epsilon
  }

  return { ...t, data, requiresGrad: false }
}

function elementwiseDivide(a: Tensor, b: Tensor): Tensor {
  const data = acquireBuffer(a.data.length)

  // Unroll by 8
  let i = 0
  const len8 = a.data.length - 7
  for (; i < len8; i += 8) {
    data[i] = a.data[i]! / b.data[i]!
    data[i + 1] = a.data[i + 1]! / b.data[i + 1]!
    data[i + 2] = a.data[i + 2]! / b.data[i + 2]!
    data[i + 3] = a.data[i + 3]! / b.data[i + 3]!
    data[i + 4] = a.data[i + 4]! / b.data[i + 4]!
    data[i + 5] = a.data[i + 5]! / b.data[i + 5]!
    data[i + 6] = a.data[i + 6]! / b.data[i + 6]!
    data[i + 7] = a.data[i + 7]! / b.data[i + 7]!
  }

  // Handle remainder
  for (; i < a.data.length; i++) {
    data[i] = a.data[i]! / b.data[i]!
  }

  return { ...a, data, requiresGrad: false }
}
