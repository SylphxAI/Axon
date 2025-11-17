/**
 * @sylphx/nn
 * Pure functional neural network layers
 *
 * All layers follow the pattern: { init, forward }
 * - init() returns initial state
 * - forward(input, state) transforms input
 *
 * No classes, no mutation, just pure functions!
 */

export * from './types'
export * from './linear-new'
export * from './activations'
export * from './sequential'
