/**
 * @neuronline/nn
 * Pure functional neural network layers
 */

export * as linear from './linear'
export type { LinearState } from './linear'

// Conv2D
export * as conv2d from './conv2d'
export type { Conv2DState } from './conv2d'

// Dropout
export type { DropoutConfig } from './dropout'
export * as dropout from './dropout'

// BatchNorm
export type { BatchNormState } from './batchnorm'
export * as batchnorm from './batchnorm'

// LSTM
export type { LSTMState, LSTMHidden } from './lstm'
export * as lstm from './lstm'

// GRU
export type { GRUState, GRUHidden } from './gru'
export * as gru from './gru'

// Serialization
export type {
  SerializedModel,
  SerializedLayer,
  SerializedParam,
} from './serialization'
export {
  serializeTensor,
  deserializeTensor,
  saveModel,
  loadModel,
  saveModelToFile,
  loadModelFromFile,
  getModelSummary,
} from './serialization'
