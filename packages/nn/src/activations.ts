/**
 * Activation layers - Pure functional v2
 * Stateless layers (init returns {})
 */

import type { Tensor } from '@sylphx/tensor'
import * as F from '@sylphx/functional'
import type { Layer } from './types'

/**
 * ReLU activation layer
 * f(x) = max(0, x)
 */
export const ReLU = (): Layer<{}> => ({
  init: () => ({}),
  forward: (input: Tensor, _state: {}) => F.relu(input)
})

/**
 * Leaky ReLU activation layer
 * f(x) = max(alpha * x, x)
 */
export const LeakyReLU = (alpha = 0.01): Layer<{ alpha: number }> => ({
  init: () => ({ alpha }),
  forward: (input: Tensor, state: { alpha: number }) => F.leakyRelu(input, state.alpha)
})

/**
 * Tanh activation layer
 * f(x) = tanh(x)
 */
export const Tanh = (): Layer<{}> => ({
  init: () => ({}),
  forward: (input: Tensor, _state: {}) => F.tanh(input)
})

/**
 * Sigmoid activation layer
 * f(x) = 1 / (1 + exp(-x))
 */
export const Sigmoid = (): Layer<{}> => ({
  init: () => ({}),
  forward: (input: Tensor, _state: {}) => F.sigmoid(input)
})

/**
 * Softmax activation layer
 * f(x) = exp(x) / sum(exp(x))
 * Supports tensors of any dimension (applies along last dimension)
 *
 * @param dim Dimension to apply softmax (default: -1, last dimension)
 */
export const Softmax = (dim: number = -1): Layer<{ dim: number }> => ({
  init: () => ({ dim }),
  forward: (input: Tensor, state: { dim: number }) => F.softmax(input, state.dim)
})
