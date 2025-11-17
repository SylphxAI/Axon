/**
 * Adam optimizer (Adaptive Moment Estimation) - Pure functional v2
 */

import type { Tensor } from '@sylphx/tensor'
import { sub, mul, add, scalar, zeros, div, sqrt, square } from '@sylphx/tensor'
import type { Optimizer } from './types'

/**
 * Adam configuration
 */
export type AdamConfig = {
  readonly lr: number          // Learning rate
  readonly beta1?: number       // First moment decay (default: 0.9)
  readonly beta2?: number       // Second moment decay (default: 0.999)
  readonly epsilon?: number     // Numerical stability (default: 1e-8)
  readonly weightDecay?: number // L2 regularization (default: 0)
}

/**
 * Adam optimizer state
 */
export type AdamState = {
  readonly step: number
  readonly m: Tensor[]       // First moment estimates
  readonly v: Tensor[]       // Second moment estimates
  readonly config: Required<AdamConfig>
}

/**
 * Adam optimizer constructor
 * Returns { init, step } pair
 *
 * @param config - Adam configuration
 * @returns Optimizer with init and step functions
 *
 * @example
 * const optimizer = Adam({ lr: 0.01 })
 * const state = optimizer.init(params)
 * const result = optimizer.step(params, grads, state)
 */
export const Adam = (config: AdamConfig): Optimizer<AdamState> => {
  const fullConfig: Required<AdamConfig> = {
    lr: config.lr,
    beta1: config.beta1 ?? 0.9,
    beta2: config.beta2 ?? 0.999,
    epsilon: config.epsilon ?? 1e-8,
    weightDecay: config.weightDecay ?? 0
  }

  return {
    /**
     * Initialize Adam state
     * Creates first and second moment buffers
     */
    init: (params: Tensor[]): AdamState => ({
      step: 0,
      m: params.map(p => zeros(p.shape)),
      v: params.map(p => zeros(p.shape)),
      config: fullConfig
    }),

    /**
     * Perform Adam optimization step
     * Updates parameters using adaptive learning rates
     */
    step: (params: Tensor[], grads: Tensor[], state: AdamState) => {
      const { lr, beta1, beta2, epsilon, weightDecay } = state.config

      const newParams: Tensor[] = []
      const newM: Tensor[] = []
      const newV: Tensor[] = []

      const t = state.step + 1

      for (let i = 0; i < params.length; i++) {
        const param = params[i]!
        let grad = grads[i]!

        // Apply weight decay if specified
        if (weightDecay !== 0) {
          grad = add(grad, mul(param, scalar(weightDecay)))
        }

        // Update biased first moment estimate
        // m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
        const m = add(
          mul(state.m[i]!, scalar(beta1)),
          mul(grad, scalar(1 - beta1))
        )

        // Update biased second moment estimate
        // v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
        const v = add(
          mul(state.v[i]!, scalar(beta2)),
          mul(square(grad), scalar(1 - beta2))
        )

        // Compute bias-corrected first moment estimate
        // m_hat = m_t / (1 - beta1^t)
        const mHat = div(m, scalar(1 - Math.pow(beta1, t)))

        // Compute bias-corrected second moment estimate
        // v_hat = v_t / (1 - beta2^t)
        const vHat = div(v, scalar(1 - Math.pow(beta2, t)))

        // Compute update
        // theta_t = theta_{t-1} - lr * m_hat / (sqrt(v_hat) + epsilon)
        const update = mul(
          div(mHat, add(sqrt(vHat), scalar(epsilon))),
          scalar(lr)
        )

        const newParam = sub(param, update)

        newM.push(m)
        newV.push(v)
        newParams.push(newParam)
      }

      return {
        params: newParams,
        state: {
          step: t,
          m: newM,
          v: newV,
          config: state.config
        }
      }
    }
  }
}
