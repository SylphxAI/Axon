/**
 * SGD (Stochastic Gradient Descent) optimizer
 * Pure functional implementation
 */

import type { Tensor } from '@neuronline/tensor'
import { sub, mul, add, scalar, zeros } from '@neuronline/tensor'
import type { OptimizerState, UpdateResult } from './types'

export type SGDConfig = {
  readonly lr: number
  readonly momentum?: number
  readonly dampening?: number
  readonly weightDecay?: number
  readonly nesterov?: boolean
}

/**
 * Initialize SGD optimizer state
 * Pure function - returns initial state
 */
export function init(params: readonly Tensor[], config: SGDConfig): OptimizerState {
  const state: Record<string, unknown> = {
    config,
  }

  // Initialize momentum buffers if needed
  if (config.momentum && config.momentum > 0) {
    const momentumBuffers = params.map((p) => zeros(p.shape))
    state.momentumBuffers = momentumBuffers
  }

  return {
    step: 0,
    params,
    state,
  }
}

/**
 * Perform SGD update step
 * Pure function - returns new parameters and state
 */
export function step(
  optState: OptimizerState,
  params: readonly Tensor[],
  grads: Map<Tensor, Tensor>
): UpdateResult {
  const config = optState.state.config as SGDConfig
  const { lr, momentum = 0, dampening = 0, weightDecay = 0, nesterov = false } = config

  const newParams: Tensor[] = []
  let momentumBuffers = optState.state.momentumBuffers as Tensor[] | undefined

  for (let i = 0; i < params.length; i++) {
    const param = params[i]!
    const grad = grads.get(param)

    if (!grad) {
      newParams.push(param)
      continue
    }

    // Weight decay
    let dParam = grad
    if (weightDecay !== 0) {
      dParam = add(dParam, mul(param, scalar(weightDecay)))
    }

    // Momentum
    if (momentum !== 0) {
      if (!momentumBuffers) {
        momentumBuffers = optState.params.map((p) => zeros(p.shape))
      }

      const buf = momentumBuffers[i]!

      // v_t = momentum * v_{t-1} + (1 - dampening) * grad
      const newBuf = add(
        mul(buf, scalar(momentum)),
        mul(dParam, scalar(1 - dampening))
      )
      momentumBuffers[i] = newBuf

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

  const newState: OptimizerState = {
    step: optState.step + 1,
    params: newParams,
    state: {
      ...optState.state,
      momentumBuffers,
    },
  }

  return {
    params: newParams,
    state: newState,
  }
}
