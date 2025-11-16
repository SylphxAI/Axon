import { FTRLModel } from './ftrl'
import { crossEntropyLoss } from './math'
import { UniformReplayBuffer } from './replay-buffer'
import { SGDModel } from './sgd'
import type { LearningMetrics, OnlineModel, TrainingExample, Vector } from './types'
import type { ReplayBuffer } from './types'

export interface OnlineLearnerConfig {
  inputSize: number
  learningRate?: number
  algorithm?: 'ftrl' | 'sgd'
  regularization?: number
  replayBufferSize?: number
  replayBatchSize?: number
  updateFrequency?: number
}

export class OnlineLearner {
  private model: OnlineModel
  private replayBuffer: ReplayBuffer
  private replayBatchSize: number
  private updateFrequency: number
  private updateCounter = 0
  private metrics: LearningMetrics

  constructor(config: OnlineLearnerConfig) {
    const {
      inputSize,
      learningRate = 0.01,
      algorithm = 'ftrl',
      regularization = 0.1,
      replayBufferSize = 1000,
      replayBatchSize = 5,
      updateFrequency = 1,
    } = config

    const modelConfig = {
      inputSize,
      learningRate,
      regularization,
      gradientClipping: 1.0,
    }

    this.model = algorithm === 'ftrl' ? new FTRLModel(modelConfig) : new SGDModel(modelConfig)

    this.replayBuffer = new UniformReplayBuffer(replayBufferSize)
    this.replayBatchSize = replayBatchSize
    this.updateFrequency = updateFrequency

    this.metrics = {
      loss: 0,
      predictions: 0,
      updates: 0,
      lastUpdate: Date.now(),
    }
  }

  predict(features: Vector): number {
    this.metrics.predictions++
    return this.model.predict(features)
  }

  learn(example: TrainingExample): void {
    const prediction = this.model.predict(example.features)
    const loss = crossEntropyLoss(prediction, example.label)

    this.metrics.loss = this.metrics.loss * 0.99 + loss * 0.01

    this.model.update(example)
    this.metrics.updates++
    this.metrics.lastUpdate = Date.now()

    this.replayBuffer.add(example)

    this.updateCounter++
    if (this.updateCounter >= this.updateFrequency) {
      this.replayUpdate()
      this.updateCounter = 0
    }
  }

  private replayUpdate(): void {
    if (this.replayBuffer.size() < this.replayBatchSize) return

    const samples = this.replayBuffer.sample(this.replayBatchSize)
    for (const sample of samples) {
      this.model.update(sample)
    }
  }

  getMetrics(): LearningMetrics {
    return { ...this.metrics }
  }

  getWeights(): Vector {
    return this.model.getWeights()
  }

  reset(): void {
    this.model.reset()
    this.replayBuffer.clear()
    this.updateCounter = 0
    this.metrics = {
      loss: 0,
      predictions: 0,
      updates: 0,
      lastUpdate: Date.now(),
    }
  }

  export(): {
    weights: number[]
    metrics: LearningMetrics
  } {
    return {
      weights: Array.from(this.model.getWeights()),
      metrics: this.getMetrics(),
    }
  }

  import(data: { weights: number[] }): void {
    this.reset()
    const weights = new Float32Array(data.weights)
    if (this.model instanceof FTRLModel) {
      const ftrlModel = this.model as FTRLModel
      const modelWeights = ftrlModel.getWeights()
      for (let i = 0; i < Math.min(weights.length, modelWeights.length); i++) {
        modelWeights[i] = weights[i]!
      }
    } else if (this.model instanceof SGDModel) {
      const sgdModel = this.model as SGDModel
      const modelWeights = sgdModel.getWeights()
      for (let i = 0; i < Math.min(weights.length, modelWeights.length); i++) {
        modelWeights[i] = weights[i]!
      }
    }
  }
}
