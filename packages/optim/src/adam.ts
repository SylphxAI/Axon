/**
 * Adam optimizer (Adaptive Moment Estimation)
 * Pure functional implementation
 */

import type { Tensor } from '@neuronline/tensor'
import { sub, mul, add, scalar, zeros, acquireBuffer } from '@neuronline/tensor'
import type { OptimizerState, UpdateResult } from './types'

export type AdamConfig = {
  readonly lr: number
  readonly beta1?: number // First moment decay (default: 0.9)
  readonly beta2?: number // Second moment decay (default: 0.999)
  readonly epsilon?: number // Numerical stability (default: 1e-8)
  readonly weightDecay?: number
  readonly amsgrad?: boolean
}

/**
 * Initialize Adam optimizer state
 * Pure function - returns initial state
 */
export function init(params: readonly Tensor[], config: AdamConfig): OptimizerState {
  // Initialize first and second moment buffers
  const m = params.map((p) => zeros(p.shape)) // First moment (mean)
  const v = params.map((p) => zeros(p.shape)) // Second moment (variance)
  const vMax = config.amsgrad ? params.map((p) => zeros(p.shape)) : undefined

  return {
    step: 0,
    params,
    state: {
      config,
      m,
      v,
      vMax,
    },
  }
}

/**
 * Perform Adam update step
 * Pure function - returns new parameters and state
 */
export function step(
  optState: OptimizerState,
  params: readonly Tensor[],
  grads: Map<Tensor, Tensor>
): UpdateResult {
  const config = optState.state.config as AdamConfig
  const {
    lr,
    beta1 = 0.9,
    beta2 = 0.999,
    epsilon = 1e-8,
    weightDecay = 0,
    amsgrad = false,
  } = config

  const m = optState.state.m as Tensor[]
  const v = optState.state.v as Tensor[]
  const vMax = optState.state.vMax as Tensor[] | undefined

  const newParams: Tensor[] = []
  const newM: Tensor[] = []
  const newV: Tensor[] = []
  const newVMax: Tensor[] | undefined = amsgrad ? [] : undefined

  const t = optState.step + 1

  for (let i = 0; i < params.length; i++) {
    const param = params[i]!
    const grad = grads.get(param)

    if (!grad) {
      newParams.push(param)
      newM.push(m[i]!)
      newV.push(v[i]!)
      if (amsgrad) newVMax!.push(vMax![i]!)
      continue
    }

    // Weight decay
    let dParam = grad
    if (weightDecay !== 0) {
      dParam = add(dParam, mul(param, scalar(weightDecay)))
    }

    // Update biased first moment: m_t = beta1 * m_{t-1} + (1 - beta1) * grad
    const mNew = add(mul(m[i]!, scalar(beta1)), mul(dParam, scalar(1 - beta1)))
    newM.push(mNew)

    // Update biased second moment: v_t = beta2 * v_{t-1} + (1 - beta2) * grad^2
    const gradSquared = elementwiseSquare(dParam)
    const vNew = add(mul(v[i]!, scalar(beta2)), mul(gradSquared, scalar(1 - beta2)))
    newV.push(vNew)

    // Compute bias-corrected moments
    const mHat = mul(mNew, scalar(1 / (1 - Math.pow(beta1, t))))
    let vHat = mul(vNew, scalar(1 / (1 - Math.pow(beta2, t))))

    // AMSGrad: use max of past v_hat
    if (amsgrad) {
      vHat = elementwiseMax(vHat, vMax![i]!)
      newVMax!.push(vHat)
    }

    // Update: param = param - lr * m_hat / (sqrt(v_hat) + epsilon)
    const denominator = elementwiseSqrtPlusEpsilon(vHat, epsilon)
    const update = elementwiseDivide(mHat, denominator)
    const newParam = sub(param, mul(update, scalar(lr)))
    newParams.push(newParam)
  }

  const newState: OptimizerState = {
    step: t,
    params: newParams,
    state: {
      config,
      m: newM,
      v: newV,
      vMax: newVMax,
    },
  }

  return {
    params: newParams,
    state: newState,
  }
}

// Helper functions for element-wise operations with memory pooling

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

function elementwiseMax(a: Tensor, b: Tensor): Tensor {
  const data = acquireBuffer(a.data.length)

  // Unroll by 8
  let i = 0
  const len8 = a.data.length - 7
  for (; i < len8; i += 8) {
    data[i] = Math.max(a.data[i]!, b.data[i]!)
    data[i + 1] = Math.max(a.data[i + 1]!, b.data[i + 1]!)
    data[i + 2] = Math.max(a.data[i + 2]!, b.data[i + 2]!)
    data[i + 3] = Math.max(a.data[i + 3]!, b.data[i + 3]!)
    data[i + 4] = Math.max(a.data[i + 4]!, b.data[i + 4]!)
    data[i + 5] = Math.max(a.data[i + 5]!, b.data[i + 5]!)
    data[i + 6] = Math.max(a.data[i + 6]!, b.data[i + 6]!)
    data[i + 7] = Math.max(a.data[i + 7]!, b.data[i + 7]!)
  }

  // Handle remainder
  for (; i < a.data.length; i++) {
    data[i] = Math.max(a.data[i]!, b.data[i]!)
  }

  return { ...a, data, requiresGrad: false }
}
