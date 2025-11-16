// Neural Network (General-Purpose ML)
export { NeuralNetwork } from './neural-network'
export type {
  NeuralNetworkConfig,
  TrainingData,
  TrainingOptions,
  TrainingMetrics,
} from './neural-network'

// Activation Functions
export { getActivation, leakyRelu, linear } from './activation'
export type { ActivationFunction as NNActivationFunction } from './activation'

// Loss Functions
export {
  getLoss,
  binaryCrossEntropy,
  mse,
  huberLoss,
  categoricalCrossEntropy,
  categoricalCrossEntropyGradient,
} from './loss'
export type { LossFunction } from './loss'

// Optimizers
export {
  createOptimizer,
  SGDOptimizer,
  MomentumOptimizer,
  AdamOptimizer,
  RMSpropOptimizer,
} from './optimizer'
export type { Optimizer, OptimizerConfig, OptimizerState } from './optimizer'

// Layers
export { createLayer, layerForward, layerBackward, updateLayerWeights, serializeLayer, deserializeLayer } from './layer'
export type { LayerConfig, LayerState } from './layer'

// Online Learning (Legacy - will deprecate)
export { OnlineLearner } from './online-learner'
export type { OnlineLearnerConfig } from './online-learner'

export { FTRLModel } from './ftrl'
export { SGDModel } from './sgd'

export { PrioritizedReplayBuffer, UniformReplayBuffer } from './replay-buffer'

export * from './bandit'
export * from './functional-learner'
export * from './math'
export * from './types'
