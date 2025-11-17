/**
 * Pure Functional Layer API (v2)
 *
 * Layer = { init, forward }
 * - init() creates initial state
 * - forward(input, state) transforms input
 */

import type { Tensor } from '@sylphx/tensor'

/**
 * Layer interface
 * A layer is a pair of pure functions
 */
export type Layer<State = any> = {
  /**
   * Initialize layer state
   * Pure function - returns new state
   */
  init: () => State

  /**
   * Forward pass
   * Pure function - transforms input using state
   */
  forward: (input: Tensor, state: State) => Tensor
}

/**
 * Model state is an array of layer states
 */
export type ModelState = any[]
