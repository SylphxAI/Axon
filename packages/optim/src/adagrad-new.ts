/**
 * AdaGrad optimizer
 * Pure functional implementation
 */

import type { Tensor } from '@sylphx/tensor'
import { sub, mul, add, scalar, zeros, square, sqrt, div } from '@sylphx/tensor'
import type { Optimizer } from './types'

export type AdaGradConfig = {
  readonly lr: number
  readonly epsilon?: number // Numerical stability (default: 1e-8)
  readonly weightDecay?: number
}

export type AdaGradState = {
  readonly config: AdaGradConfig
  readonly accum: Tensor[]
}

/**
 * AdaGrad optimizer factory
 * Returns optimizer with init and step functions
 */
export const AdaGrad = (config: AdaGradConfig): Optimizer<AdaGradState> => {
  const fullConfig: AdaGradConfig = {
    epsilon: 1e-8,
    weightDecay: 0,
    ...config,
  }

  return {
    init: (params: Tensor[]): AdaGradState => ({
      config: fullConfig,
      accum: params.map((p) => zeros(p.shape)),
    }),

    step: (params: Tensor[], grads: Tensor[], state: AdaGradState) => {
      const { lr, epsilon = 1e-8, weightDecay = 0 } = state.config

      const newParams: Tensor[] = []
      const newAccum: Tensor[] = []

      for (let i = 0; i < params.length; i++) {
        const param = params[i]!
        const grad = grads[i]!

        // Weight decay
        let dParam = grad
        if (weightDecay !== 0) {
          dParam = add(dParam, mul(param, scalar(weightDecay)))
        }

        // Accumulate squared gradients: accum_t = accum_{t-1} + grad^2
        const gradSquared = square(dParam)
        const newAcc = add(state.accum[i]!, gradSquared)
        newAccum.push(newAcc)

        // Update: param = param - lr * grad / (sqrt(accum_t) + epsilon)
        const sqrtAcc = sqrt(newAcc)
        const denominator = add(sqrtAcc, scalar(epsilon))
        const update = div(dParam, denominator)
        const newParam = sub(param, mul(update, scalar(lr)))
        newParams.push(newParam)
      }

      return {
        params: newParams,
        state: {
          config: state.config,
          accum: newAccum,
        },
      }
    },
  }
}
