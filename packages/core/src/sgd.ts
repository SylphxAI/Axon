import { clip, dot, sigmoid } from './math'
import type { ModelConfig, OnlineModel, TrainingExample, Vector } from './types'

export class SGDModel implements OnlineModel {
  private weights: Vector
  private learningRate: number
  private regularization: number
  private gradientClipping: number
  private momentum: Vector
  private momentumFactor: number

  constructor(config: ModelConfig) {
    this.weights = new Float32Array(config.inputSize)
    this.momentum = new Float32Array(config.inputSize)
    this.learningRate = config.learningRate
    this.regularization = config.regularization ?? 0.01
    this.gradientClipping = config.gradientClipping ?? 1.0
    this.momentumFactor = 0.9

    const scale = Math.sqrt(2.0 / config.inputSize)
    for (let i = 0; i < this.weights.length; i++) {
      this.weights[i] = (Math.random() - 0.5) * 2 * scale
    }
  }

  predict(features: Vector): number {
    const logit = dot(this.weights, features)
    return sigmoid(logit)
  }

  update(example: TrainingExample): void {
    const { features, label } = example
    const prediction = this.predict(features)
    const error = prediction - label

    const gradient = new Float32Array(features.length)
    for (let i = 0; i < features.length; i++) {
      gradient[i] = error * features[i]! + this.regularization * this.weights[i]!
      gradient[i] = clip(gradient[i]!, -this.gradientClipping, this.gradientClipping)
    }

    for (let i = 0; i < this.weights.length; i++) {
      this.momentum[i] = this.momentumFactor * this.momentum[i]! - this.learningRate * gradient[i]!
      this.weights[i] = this.weights[i]! + this.momentum[i]!
    }
  }

  getWeights(): Vector {
    return this.weights.slice()
  }

  reset(): void {
    this.weights.fill(0)
    this.momentum.fill(0)
    const scale = Math.sqrt(2.0 / this.weights.length)
    for (let i = 0; i < this.weights.length; i++) {
      this.weights[i] = (Math.random() - 0.5) * 2 * scale
    }
  }
}
