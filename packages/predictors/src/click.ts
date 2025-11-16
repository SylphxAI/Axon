import { OnlineLearner } from '@sylphx/neuronline-core'
import type { OnlineLearnerConfig, TrainingExample } from '@sylphx/neuronline-core'
import { clickToVector } from './feature-extractor'
import type { ClickContext } from './feature-extractor'

export interface ClickPredictorConfig extends Partial<OnlineLearnerConfig> {
  vectorSize?: number
  threshold?: number
}

export interface ClickEvent {
  context: ClickContext
  clicked: boolean
  targetId?: string
}

export class ClickPredictor {
  private learner: OnlineLearner
  private vectorSize: number
  private threshold: number
  private history: ClickEvent[] = []
  private maxHistory = 100

  constructor(config: ClickPredictorConfig = {}) {
    this.vectorSize = config.vectorSize ?? 16
    this.threshold = config.threshold ?? 0.5

    this.learner = new OnlineLearner({
      inputSize: this.vectorSize,
      learningRate: config.learningRate ?? 0.01,
      algorithm: config.algorithm ?? 'ftrl',
      regularization: config.regularization ?? 0.1,
      replayBufferSize: config.replayBufferSize ?? 500,
      replayBatchSize: config.replayBatchSize ?? 5,
    })
  }

  predict(context: ClickContext): number {
    const features = clickToVector(context, this.vectorSize)
    return this.learner.predict(features)
  }

  willClick(context: ClickContext): boolean {
    return this.predict(context) > this.threshold
  }

  learn(event: ClickEvent): void {
    const features = clickToVector(event.context, this.vectorSize)
    const example: TrainingExample = {
      features,
      label: event.clicked ? 1 : 0,
      timestamp: event.context.timestamp ?? Date.now(),
    }

    this.learner.learn(example)

    this.history.push(event)
    if (this.history.length > this.maxHistory) {
      this.history.shift()
    }
  }

  getMetrics() {
    return this.learner.getMetrics()
  }

  getAccuracy(): number {
    if (this.history.length === 0) return 0

    let correct = 0
    for (const event of this.history) {
      const prediction = this.willClick(event.context)
      if (prediction === event.clicked) {
        correct++
      }
    }

    return correct / this.history.length
  }

  export() {
    return {
      ...this.learner.export(),
      vectorSize: this.vectorSize,
      threshold: this.threshold,
      history: this.history.slice(-10),
    }
  }

  import(data: { weights: number[]; vectorSize?: number; threshold?: number }) {
    this.learner.import(data)
    if (data.vectorSize) this.vectorSize = data.vectorSize
    if (data.threshold) this.threshold = data.threshold
  }

  reset(): void {
    this.learner.reset()
    this.history = []
  }
}
