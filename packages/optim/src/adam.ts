/**
 * Adam optimizer (Adaptive Moment Estimation)
 * Pure functional implementation
 */

import type { Tensor } from '@neuronline/tensor'
import { sub, mul, add, scalar, zeros } from '@neuronline/tensor'
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

// Helper functions for element-wise operations

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

function elementwiseMax(a: Tensor, b: Tensor): Tensor {
  const data = new Float32Array(a.data.length)
  for (let i = 0; i < a.data.length; i++) {
    data[i] = Math.max(a.data[i]!, b.data[i]!)
  }
  return { ...a, data, requiresGrad: false }
}
