export type Vector = Float32Array

export interface TrainingExample {
  features: Vector
  label: number
  weight?: number
  timestamp?: number
}

export interface ModelConfig {
  inputSize: number
  learningRate: number
  regularization?: number
  gradientClipping?: number
  memorySize?: number
}

export interface OnlineModel {
  predict(features: Vector): number
  update(example: TrainingExample): void
  getWeights(): Vector
  reset(): void
}

export interface ReplayBuffer {
  add(example: TrainingExample): void
  sample(n: number): TrainingExample[]
  size(): number
  clear(): void
}

export type ActivationFunction = (x: number) => number

export interface LearningMetrics {
  loss: number
  accuracy?: number
  predictions: number
  updates: number
  lastUpdate: number
}
