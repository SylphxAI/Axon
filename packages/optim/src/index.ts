/**
 * @sylphx/optim
 * Pure functional optimizers for neural networks
 */

// Types
export type { OptimizerState, OptimizerConfig, UpdateResult } from './types'

// SGD
export type { SGDConfig } from './sgd'
export * as sgd from './sgd'

// Adam
export type { AdamConfig } from './adam'
export * as adam from './adam'

// RMSprop
export type { RMSpropConfig } from './rmsprop'
export * as rmsprop from './rmsprop'

// AdaGrad
export type { AdaGradConfig } from './adagrad'
export * as adagrad from './adagrad'
