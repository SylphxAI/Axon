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
export * from './dropout-new'
export * from './conv2d-new'

// RNN layers (separate utilities for sequence processing)
export * as LSTM from './lstm'
export * as GRU from './gru'

// Stateful layers (return { output, state })
export * as BatchNorm from './batchnorm'
