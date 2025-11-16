/**
 * AdaGrad optimizer
 * Pure functional implementation
 */

import type { Tensor } from '@neuronline/tensor'
import { sub, mul, add, scalar, zeros } from '@neuronline/tensor'
import type { OptimizerState, UpdateResult } from './types'

export type AdaGradConfig = {
  readonly lr: number
  readonly epsilon?: number // Numerical stability (default: 1e-8)
  readonly weightDecay?: number
}

/**
 * Initialize AdaGrad optimizer state
 */
export function init(params: readonly Tensor[], config: AdaGradConfig): OptimizerState {
  // Initialize accumulated squared gradients
  const accum = params.map((p) => zeros(p.shape))

  return {
    step: 0,
    params,
    state: {
      config,
      accum,
    },
  }
}

/**
 * Perform AdaGrad update step
 */
export function step(
  optState: OptimizerState,
  params: readonly Tensor[],
  grads: Map<Tensor, Tensor>
): UpdateResult {
  const config = optState.state.config as AdaGradConfig
  const { lr, epsilon = 1e-8, weightDecay = 0 } = config

  const accum = optState.state.accum as Tensor[]

  const newParams: Tensor[] = []
  const newAccum: Tensor[] = []

  for (let i = 0; i < params.length; i++) {
    const param = params[i]!
    const grad = grads.get(param)

    if (!grad) {
      newParams.push(param)
      newAccum.push(accum[i]!)
      continue
    }

    // Weight decay
    let dParam = grad
    if (weightDecay !== 0) {
      dParam = add(dParam, mul(param, scalar(weightDecay)))
    }

    // Accumulate squared gradients: accum_t = accum_{t-1} + grad^2
    const gradSquared = elementwiseSquare(dParam)
    const newAcc = add(accum[i]!, gradSquared)
    newAccum.push(newAcc)

    // Update: param = param - lr * grad / (sqrt(accum_t) + epsilon)
    const denominator = elementwiseSqrtPlusEpsilon(newAcc, epsilon)
    const update = elementwiseDivide(dParam, denominator)
    const newParam = sub(param, mul(update, scalar(lr)))
    newParams.push(newParam)
  }

  const newState: OptimizerState = {
    step: optState.step + 1,
    params: newParams,
    state: {
      config,
      accum: newAccum,
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
