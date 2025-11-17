/**
 * SGD (Stochastic Gradient Descent) optimizer
 * Pure functional implementation
 */

import type { Tensor } from '@sylphx/tensor'
import { sub, mul, add, scalar, zeros } from '@sylphx/tensor'
import type { Optimizer } from './types'

export type SGDConfig = {
  readonly lr: number
  readonly momentum?: number
  readonly dampening?: number
  readonly weightDecay?: number
  readonly nesterov?: boolean
}

export type SGDState = {
  readonly config: SGDConfig
  readonly momentumBuffers?: Tensor[]
}

/**
 * SGD optimizer factory
 * Returns optimizer with init and step functions
 */
export const SGD = (config: SGDConfig): Optimizer<SGDState> => {
  const fullConfig: SGDConfig = {
    momentum: 0,
    dampening: 0,
    weightDecay: 0,
    nesterov: false,
    ...config,
  }

  return {
    init: (params: Tensor[]): SGDState => {
      const state: SGDState = { config: fullConfig }

      // Initialize momentum buffers if needed
      if (fullConfig.momentum && fullConfig.momentum > 0) {
        return {
          ...state,
          momentumBuffers: params.map((p) => zeros(p.shape)),
        }
      }

      return state
    },

    step: (params: Tensor[], grads: Tensor[], state: SGDState) => {
      const { lr, momentum = 0, dampening = 0, weightDecay = 0, nesterov = false } = state.config

      const newParams: Tensor[] = []
      let momentumBuffers = state.momentumBuffers

      for (let i = 0; i < params.length; i++) {
        const param = params[i]!
        const grad = grads[i]!

        // Weight decay
        let dParam = grad
        if (weightDecay !== 0) {
          dParam = add(dParam, mul(param, scalar(weightDecay)))
        }

        // Momentum
        if (momentum !== 0) {
          if (!momentumBuffers) {
            momentumBuffers = params.map((p) => zeros(p.shape))
          }

          const buf = momentumBuffers[i]!

          // v_t = momentum * v_{t-1} + (1 - dampening) * grad
          const newBuf = add(
            mul(buf, scalar(momentum)),
            mul(dParam, scalar(1 - dampening))
          )
          momentumBuffers = [
            ...momentumBuffers.slice(0, i),
            newBuf,
            ...momentumBuffers.slice(i + 1),
          ]

          if (nesterov) {
            // param = param - lr * (momentum * v_t + grad)
            dParam = add(mul(newBuf, scalar(momentum)), dParam)
          } else {
            // param = param - lr * v_t
            dParam = newBuf
          }
        }

        // Update: param = param - lr * dParam
        const newParam = sub(param, mul(dParam, scalar(lr)))
        newParams.push(newParam)
      }

      return {
        params: newParams,
        state: {
          config: state.config,
          momentumBuffers,
        },
      }
    },
  }
}
