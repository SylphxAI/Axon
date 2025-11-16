import { OnlineLearner } from '@sylphx/neuronline-core'
import type { OnlineLearnerConfig, TrainingExample } from '@sylphx/neuronline-core'
import { sequenceToVector } from './feature-extractor'
import type { SequenceContext } from './feature-extractor'

export interface SequencePredictorConfig extends Partial<OnlineLearnerConfig> {
  vectorSize?: number
  maxSequenceLength?: number
}

export class SequencePredictor {
  private learner: OnlineLearner
  private vectorSize: number
  private maxSequenceLength: number
  private actionSequence: string[] = []
  private timestamps: number[] = []

  constructor(config: SequencePredictorConfig = {}) {
    this.vectorSize = config.vectorSize ?? 32
    this.maxSequenceLength = config.maxSequenceLength ?? 10

    this.learner = new OnlineLearner({
      inputSize: this.vectorSize,
      learningRate: config.learningRate ?? 0.01,
      algorithm: config.algorithm ?? 'sgd',
      regularization: config.regularization ?? 0.01,
      replayBufferSize: config.replayBufferSize ?? 1000,
      replayBatchSize: config.replayBatchSize ?? 10,
    })
  }

  addAction(action: string, timestamp = Date.now()): void {
    this.actionSequence.push(action)
    this.timestamps.push(timestamp)

    if (this.actionSequence.length > this.maxSequenceLength) {
      this.actionSequence.shift()
      this.timestamps.shift()
    }
  }

  predictNext(possibleActions: string[]): Map<string, number> {
    const predictions = new Map<string, number>()

    for (const action of possibleActions) {
      const testSequence = [...this.actionSequence, action]
      const context: SequenceContext = {
        actions: testSequence,
        timestamps: this.timestamps,
        maxLength: this.maxSequenceLength,
      }

      const features = sequenceToVector(context, this.vectorSize)
      const score = this.learner.predict(features)
      predictions.set(action, score)
    }

    return predictions
  }

  getMostLikely(possibleActions: string[]): string | null {
    const predictions = this.predictNext(possibleActions)
    let maxScore = -1
    let bestAction: string | null = null

    for (const [action, score] of predictions) {
      if (score > maxScore) {
        maxScore = score
        bestAction = action
      }
    }

    return bestAction
  }

  learn(sequence: string[], outcome: number): void {
    const context: SequenceContext = {
      actions: sequence,
      maxLength: this.maxSequenceLength,
    }

    const features = sequenceToVector(context, this.vectorSize)
    const example: TrainingExample = {
      features,
      label: outcome,
      timestamp: Date.now(),
    }

    this.learner.learn(example)
  }

  learnFromCurrent(nextAction: string, wasCorrect: boolean): void {
    const sequence = [...this.actionSequence, nextAction]
    this.learn(sequence, wasCorrect ? 1 : 0)
  }

  getMetrics() {
    return this.learner.getMetrics()
  }

  reset(): void {
    this.learner.reset()
    this.actionSequence = []
    this.timestamps = []
  }

  export() {
    return {
      ...this.learner.export(),
      vectorSize: this.vectorSize,
      maxSequenceLength: this.maxSequenceLength,
      actionSequence: this.actionSequence,
      timestamps: this.timestamps,
    }
  }

  import(data: {
    weights: number[]
    vectorSize?: number
    maxSequenceLength?: number
    actionSequence?: string[]
    timestamps?: number[]
  }) {
    this.learner.import(data)
    if (data.vectorSize) this.vectorSize = data.vectorSize
    if (data.maxSequenceLength) this.maxSequenceLength = data.maxSequenceLength
    if (data.actionSequence) this.actionSequence = data.actionSequence
    if (data.timestamps) this.timestamps = data.timestamps
  }
}
