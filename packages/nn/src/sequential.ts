/**
 * Sequential - Layer combinator
 * Composes multiple layers into a single layer
 */

import type { Tensor } from '@sylphx/tensor'
import type { Layer } from './types'

/**
 * Sequential layer combinator
 * Chains multiple layers together
 *
 * @param layers - Variable number of layers to compose
 * @returns Single layer that applies all layers in sequence
 *
 * @example
 * const model = Sequential(
 *   Linear(2, 8),
 *   Tanh(),
 *   Linear(8, 1)
 * )
 *
 * const state = model.init()  // [linearState1, {}, linearState2]
 * const output = model.forward(input, state)
 */
export const Sequential = (...layers: Layer[]): Layer => ({
  /**
   * Initialize all layers
   * Returns array of layer states
   */
  init: () => layers.map(layer => layer.init()),

  /**
   * Forward pass through all layers
   * Threads input through each layer in sequence
   */
  forward: (input: Tensor, states: any[]) => {
    return layers.reduce((x, layer, i) => {
      return layer.forward(x, states[i])
    }, input)
  }
})
