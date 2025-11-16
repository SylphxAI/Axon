/**
 * RMSprop optimizer
 * Pure functional implementation
 */

import type { Tensor } from '@neuronline/tensor'
import { sub, mul, add, scalar, zeros } from '@neuronline/tensor'
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
  const data = new Float32Array(t.data.length)
  for (let i = 0; i < t.data.length; i++) {
    data[i] = t.data[i]! * t.data[i]!
  }
  return { ...t, data, requiresGrad: false }
}

function elementwiseSqrtPlusEpsilon(t: Tensor, epsilon: number): Tensor {
  const data = new Float32Array(t.data.length)
  for (let i = 0; i < t.data.length; i++) {
    data[i] = Math.sqrt(t.data[i]!) + epsilon
  }
  return { ...t, data, requiresGrad: false }
}

function elementwiseDivide(a: Tensor, b: Tensor): Tensor {
  const data = new Float32Array(a.data.length)
  for (let i = 0; i < a.data.length; i++) {
    data[i] = a.data[i]! / b.data[i]!
  }
  return { ...a, data, requiresGrad: false }
}
