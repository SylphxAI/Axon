import { describe, expect, test } from 'bun:test'
import { OnlineLearner } from './online-learner'
import type { TrainingExample } from './types'

describe('OnlineLearner', () => {
  test('initialization', () => {
    const learner = new OnlineLearner({
      inputSize: 10,
      learningRate: 0.01,
    })

    expect(learner.getMetrics().predictions).toBe(0)
    expect(learner.getMetrics().updates).toBe(0)
  })

  test('prediction', () => {
    const learner = new OnlineLearner({
      inputSize: 3,
    })

    const features = new Float32Array([1, 0.5, 0.3])
    const prediction = learner.predict(features)

    expect(prediction).toBeGreaterThanOrEqual(0)
    expect(prediction).toBeLessThanOrEqual(1)
    expect(learner.getMetrics().predictions).toBe(1)
  })

  test('learning simple pattern', () => {
    const learner = new OnlineLearner({
      inputSize: 2,
      learningRate: 0.15,
      algorithm: 'sgd',
      regularization: 0.001,
    })

    const positiveExamples: TrainingExample[] = [
      { features: new Float32Array([1, 1]), label: 1 },
      { features: new Float32Array([1, 0.8]), label: 1 },
      { features: new Float32Array([0.9, 1]), label: 1 },
    ]

    const negativeExamples: TrainingExample[] = [
      { features: new Float32Array([0, 0]), label: 0 },
      { features: new Float32Array([0.1, 0]), label: 0 },
      { features: new Float32Array([0, 0.1]), label: 0 },
    ]

    for (let epoch = 0; epoch < 150; epoch++) {
      for (const example of [...positiveExamples, ...negativeExamples]) {
        learner.learn(example)
      }
    }

    const positivePrediction = learner.predict(new Float32Array([1, 1]))
    const negativePrediction = learner.predict(new Float32Array([0, 0]))

    expect(positivePrediction).toBeGreaterThan(negativePrediction)
    expect(Math.abs(positivePrediction - negativePrediction)).toBeGreaterThan(0.1)
  })

  test('FTRL algorithm', () => {
    const learner = new OnlineLearner({
      inputSize: 2,
      algorithm: 'ftrl',
      learningRate: 0.1,
    })

    const example: TrainingExample = {
      features: new Float32Array([1, 0]),
      label: 1,
    }

    learner.learn(example)
    expect(learner.getMetrics().updates).toBe(1)
  })

  test('export and import', () => {
    const learner = new OnlineLearner({
      inputSize: 3,
    })

    const example: TrainingExample = {
      features: new Float32Array([1, 0.5, 0.3]),
      label: 1,
    }

    learner.learn(example)
    const exported = learner.export()

    const newLearner = new OnlineLearner({
      inputSize: 3,
    })
    newLearner.import(exported)

    const features = new Float32Array([1, 0.5, 0.3])
    const prediction1 = learner.predict(features)
    const prediction2 = newLearner.predict(features)

    expect(Math.abs(prediction1 - prediction2)).toBeLessThan(0.01)
  })

  test('metrics tracking', () => {
    const learner = new OnlineLearner({
      inputSize: 2,
    })

    expect(learner.getMetrics().loss).toBe(0)

    learner.learn({
      features: new Float32Array([1, 1]),
      label: 1,
    })

    const metrics = learner.getMetrics()
    expect(metrics.updates).toBe(1)
    expect(metrics.loss).toBeGreaterThan(0)
  })

  test('reset functionality', () => {
    const learner = new OnlineLearner({
      inputSize: 2,
    })

    learner.learn({
      features: new Float32Array([1, 1]),
      label: 1,
    })

    learner.reset()

    const metrics = learner.getMetrics()
    expect(metrics.updates).toBe(0)
    expect(metrics.predictions).toBe(0)
  })
})
