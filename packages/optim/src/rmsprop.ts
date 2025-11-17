/**
 * RMSprop optimizer
 * Pure functional implementation
 */

import type { Tensor } from '@sylphx/tensor'
import { sub, mul, add, scalar, zeros, square, sqrt, div } from '@sylphx/tensor'
import type { Optimizer } from './types'

export type RMSpropConfig = {
  readonly lr: number
  readonly alpha?: number // Smoothing constant (default: 0.99)
  readonly epsilon?: number // Numerical stability (default: 1e-8)
  readonly weightDecay?: number
}

export type RMSpropState = {
  readonly config: RMSpropConfig
  readonly sqAvg: Tensor[]
}

/**
 * RMSprop optimizer factory
 * Returns optimizer with init and step functions
 */
export const RMSprop = (config: RMSpropConfig): Optimizer<RMSpropState> => {
  const fullConfig: RMSpropConfig = {
    alpha: 0.99,
    epsilon: 1e-8,
    weightDecay: 0,
    ...config,
  }

  return {
    init: (params: Tensor[]): RMSpropState => ({
      config: fullConfig,
      sqAvg: params.map((p) => zeros(p.shape)),
    }),

    step: (params: Tensor[], grads: Tensor[], state: RMSpropState) => {
      const { lr, alpha = 0.99, epsilon = 1e-8, weightDecay = 0 } = state.config

      const newParams: Tensor[] = []
      const newSqAvg: Tensor[] = []

      for (let i = 0; i < params.length; i++) {
        const param = params[i]!
        const grad = grads[i]!

        // Weight decay
        let dParam = grad
        if (weightDecay !== 0) {
          dParam = add(dParam, mul(param, scalar(weightDecay)))
        }

        // Update squared gradient average: v_t = alpha * v_{t-1} + (1 - alpha) * grad^2
        const gradSquared = square(dParam)
        const newV = add(
          mul(state.sqAvg[i]!, scalar(alpha)),
          mul(gradSquared, scalar(1 - alpha))
        )
        newSqAvg.push(newV)

        // Update: param = param - lr * grad / (sqrt(v_t) + epsilon)
        const sqrtV = sqrt(newV)
        const denominator = add(sqrtV, scalar(epsilon))
        const update = div(dParam, denominator)
        const newParam = sub(param, mul(update, scalar(lr)))
        newParams.push(newParam)
      }

      return {
        params: newParams,
        state: {
          config: state.config,
          sqAvg: newSqAvg,
        },
      }
    },
  }
}
