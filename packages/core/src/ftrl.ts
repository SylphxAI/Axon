import { dot, sigmoid } from './math'
import type { ModelConfig, OnlineModel, TrainingExample, Vector } from './types'

export class FTRLModel implements OnlineModel {
  private weights: Vector
  private z: Vector
  private n: Vector
  private alpha: number
  private beta: number
  private lambda1: number
  private lambda2: number

  constructor(config: ModelConfig) {
    this.weights = new Float32Array(config.inputSize)
    this.z = new Float32Array(config.inputSize)
    this.n = new Float32Array(config.inputSize)

    this.alpha = config.learningRate
    this.beta = 1.0
    this.lambda1 = config.regularization ?? 0.1
    this.lambda2 = config.regularization ?? 1.0
  }

  predict(features: Vector): number {
    const logit = dot(this.weights, features)
    return sigmoid(logit)
  }

  update(example: TrainingExample): void {
    const { features, label } = example
    const prediction = this.predict(features)
    const gradient = prediction - label

    for (let i = 0; i < features.length; i++) {
      const featureValue = features[i]!
      if (featureValue === 0) continue

      const g = gradient * featureValue
      const sigma = (Math.sqrt(this.n[i]! + g * g) - Math.sqrt(this.n[i]!)) / this.alpha

      this.z[i] = this.z[i]! + g - sigma * this.weights[i]!
      this.n[i] = this.n[i]! + g * g

      const sign = this.z[i]! < 0 ? -1 : 1
      const absZ = Math.abs(this.z[i]!)

      if (absZ <= this.lambda1) {
        this.weights[i] = 0
      } else {
        this.weights[i] =
          -(sign * (absZ - this.lambda1)) /
          ((this.beta + Math.sqrt(this.n[i]!)) / this.alpha + this.lambda2)
      }
    }
  }

  getWeights(): Vector {
    return this.weights.slice()
  }

  reset(): void {
    this.weights.fill(0)
    this.z.fill(0)
    this.n.fill(0)
  }
}
