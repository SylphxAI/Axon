import { crossEntropyLoss, dot, sigmoid } from './math'
import type { TrainingExample, Vector } from './types'

export interface FunctionalModelState {
  readonly weights: Vector
  readonly momentum: Vector
  readonly z: Vector
  readonly n: Vector
  readonly learningRate: number
  readonly regularization: number
  readonly totalUpdates: number
  readonly totalLoss: number
}

export interface PredictionResult {
  readonly probability: number
  readonly loss?: number
}

export function createModelState(
  inputSize: number,
  learningRate = 0.01,
  regularization = 0.01
): FunctionalModelState {
  const scale = Math.sqrt(2.0 / inputSize)
  const weights = new Float32Array(inputSize)
  for (let i = 0; i < inputSize; i++) {
    weights[i] = (Math.random() - 0.5) * 2 * scale
  }

  return {
    weights,
    momentum: new Float32Array(inputSize),
    z: new Float32Array(inputSize),
    n: new Float32Array(inputSize),
    learningRate,
    regularization,
    totalUpdates: 0,
    totalLoss: 0,
  }
}

export function predict(state: FunctionalModelState, features: Vector): PredictionResult {
  const logit = dot(state.weights, features)
  return {
    probability: sigmoid(logit),
  }
}

export function learn(state: FunctionalModelState, example: TrainingExample): FunctionalModelState {
  const prediction = predict(state, example.features)
  const loss = crossEntropyLoss(prediction.probability, example.label)
  const error = prediction.probability - example.label

  const newWeights = new Float32Array(state.weights.length)
  const newMomentum = new Float32Array(state.weights.length)
  const momentumFactor = 0.9
  const gradientClipping = 1.0

  for (let i = 0; i < state.weights.length; i++) {
    let gradient = error * example.features[i]! + state.regularization * state.weights[i]!
    gradient = Math.max(-gradientClipping, Math.min(gradientClipping, gradient))

    newMomentum[i] = momentumFactor * state.momentum[i]! - state.learningRate * gradient
    newWeights[i] = state.weights[i]! + newMomentum[i]!
  }

  return {
    ...state,
    weights: newWeights,
    momentum: newMomentum,
    totalUpdates: state.totalUpdates + 1,
    totalLoss: state.totalLoss * 0.99 + loss * 0.01,
  }
}

export function batchLearn(
  state: FunctionalModelState,
  examples: TrainingExample[]
): FunctionalModelState {
  return examples.reduce((currentState, example) => learn(currentState, example), state)
}

export function getMetrics(state: FunctionalModelState) {
  return {
    totalUpdates: state.totalUpdates,
    averageLoss: state.totalLoss,
    learningRate: state.learningRate,
    weightNorm: Math.sqrt(dot(state.weights, state.weights)),
  }
}

export function exportModel(state: FunctionalModelState): {
  weights: number[]
  config: {
    learningRate: number
    regularization: number
  }
  stats: {
    totalUpdates: number
    averageLoss: number
  }
} {
  return {
    weights: Array.from(state.weights),
    config: {
      learningRate: state.learningRate,
      regularization: state.regularization,
    },
    stats: {
      totalUpdates: state.totalUpdates,
      averageLoss: state.totalLoss,
    },
  }
}

export function importModel(
  data: { weights: number[]; config?: { learningRate?: number; regularization?: number } },
  inputSize: number
): FunctionalModelState {
  const weights = new Float32Array(data.weights)
  return {
    weights,
    momentum: new Float32Array(inputSize),
    z: new Float32Array(inputSize),
    n: new Float32Array(inputSize),
    learningRate: data.config?.learningRate ?? 0.01,
    regularization: data.config?.regularization ?? 0.01,
    totalUpdates: 0,
    totalLoss: 0,
  }
}
