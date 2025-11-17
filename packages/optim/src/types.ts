/**
 * Pure Functional Optimizer API (v2)
 */

import type { Tensor } from '@sylphx/tensor'

/**
 * Optimizer interface
 * An optimizer is a pair of pure functions
 */
export type Optimizer<State = any> = {
  /**
   * Initialize optimizer state
   */
  init: (params: Tensor[]) => State

  /**
   * Perform optimization step
   * Returns new parameters and new optimizer state
   */
  step: (params: Tensor[], grads: Tensor[], state: State) => {
    params: Tensor[]
    state: State
  }
}
