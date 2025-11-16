/**
 * @neuronline/optim
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
