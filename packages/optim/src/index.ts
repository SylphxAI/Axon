/**
 * @sylphx/optim
 * Pure functional optimizers
 *
 * All optimizers follow the pattern: { init, step }
 * - init(params) returns initial state
 * - step(params, grads, state) returns new params and state
 */

export * from './types'
export * from './adam-new'
export * from './sgd-new'
export * from './rmsprop-new'
export * from './adagrad-new'
