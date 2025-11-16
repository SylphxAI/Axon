import { describe, expect, test } from 'bun:test'
import { NeuralNetwork } from './neural-network'

describe('NeuralNetwork', () => {
  test('creates network with correct architecture', () => {
    const nn = new NeuralNetwork({
      layers: [2, 3, 1],
      activation: 'relu',
    })

    expect(nn).toBeDefined()
  })

  test('forward pass produces output of correct size', () => {
    const nn = new NeuralNetwork({
      layers: [2, 3, 1],
      activation: 'relu',
    })

    const output = nn.run([0.5, 0.5])
    expect(output.length).toBe(1)
  })

  test('learns XOR problem (non-linear)', () => {
    // XOR is the classic test for non-linear learning
    // Linear models cannot learn XOR
    const nn = new NeuralNetwork({
      layers: [2, 8, 1], // 2 inputs, 8 hidden neurons (more neurons for better learning), 1 output
      activation: 'tanh', // tanh often works better for XOR
      outputActivation: 'sigmoid',
      loss: 'binary-crossentropy', // better for binary classification
      optimizer: 'adam',
      learningRate: 0.3, // higher learning rate
    })

    // XOR training data
    const data = [
      { input: [0, 0], output: [0] },
      { input: [0, 1], output: [1] },
      { input: [1, 0], output: [1] },
      { input: [1, 1], output: [0] },
    ]

    // Train
    const metrics = nn.train(data, {
      epochs: 2000, // more epochs
      verbose: false,
    })

    // Check that loss decreased
    const initialLoss = metrics[0]?.loss ?? 1
    const finalLoss = metrics[metrics.length - 1]?.loss ?? 1
    expect(finalLoss).toBeLessThan(initialLoss)

    // Test predictions
    const test00 = nn.run([0, 0])[0]!
    const test01 = nn.run([0, 1])[0]!
    const test10 = nn.run([1, 0])[0]!
    const test11 = nn.run([1, 1])[0]!

    // Should learn XOR correctly (threshold at 0.5)
    // Use a more lenient threshold for testing
    expect(test00).toBeLessThan(0.5) // 0 XOR 0 = 0
    expect(test01).toBeGreaterThan(0.5) // 0 XOR 1 = 1
    expect(test10).toBeGreaterThan(0.5) // 1 XOR 0 = 1
    expect(test11).toBeLessThan(0.5) // 1 XOR 1 = 0
  })

  test('learns AND problem (simple)', () => {
    const nn = new NeuralNetwork({
      layers: [2, 1],
      activation: 'sigmoid',
      loss: 'mse',
      learningRate: 0.5,
    })

    const data = [
      { input: [0, 0], output: [0] },
      { input: [0, 1], output: [0] },
      { input: [1, 0], output: [0] },
      { input: [1, 1], output: [1] },
    ]

    nn.train(data, {
      epochs: 500,
      verbose: false,
    })

    const test11 = nn.run([1, 1])[0]!
    const test00 = nn.run([0, 0])[0]!

    expect(test11).toBeGreaterThan(0.5) // 1 AND 1 = 1
    expect(test00).toBeLessThan(0.5) // 0 AND 0 = 0
  })

  test('works with different optimizers', () => {
    const optimizers = ['sgd', 'adam', 'momentum', 'rmsprop']

    for (const optimizer of optimizers) {
      const nn = new NeuralNetwork({
        layers: [2, 4, 1],
        activation: 'relu',
        outputActivation: 'sigmoid',
        optimizer,
        learningRate: 0.1,
      })

      const data = [
        { input: [0, 0], output: [0] },
        { input: [1, 1], output: [1] },
      ]

      // Should not throw
      expect(() => nn.train(data, { epochs: 10, verbose: false })).not.toThrow()
    }
  })

  test('works with different activations', () => {
    const activations = ['relu', 'sigmoid', 'tanh', 'leakyrelu']

    for (const activation of activations) {
      const nn = new NeuralNetwork({
        layers: [2, 3, 1],
        activation,
      })

      const output = nn.run([0.5, 0.5])
      expect(output.length).toBe(1)
    }
  })

  test('handles multi-output networks', () => {
    const nn = new NeuralNetwork({
      layers: [2, 4, 3], // 3 outputs
      activation: 'relu',
    })

    const output = nn.run([0.5, 0.5])
    expect(output.length).toBe(3)
  })

  test('handles deep networks', () => {
    const nn = new NeuralNetwork({
      layers: [10, 20, 15, 10, 5, 1], // 5 hidden layers
      activation: 'relu',
    })

    const output = nn.run(new Float32Array(10).fill(0.5))
    expect(output.length).toBe(1)
  })

  test('validation split works', () => {
    const nn = new NeuralNetwork({
      layers: [2, 4, 1],
      activation: 'relu',
      outputActivation: 'sigmoid',
    })

    const data = Array.from({ length: 100 }, () => ({
      input: [Math.random(), Math.random()],
      output: [Math.random() > 0.5 ? 1 : 0],
    }))

    const metrics = nn.train(data, {
      epochs: 10,
      validation: 0.2, // 20% validation
      verbose: false,
    })

    // Should have validation metrics
    expect(metrics[0]?.validationLoss).toBeDefined()
  })

  test('early stopping works', () => {
    const nn = new NeuralNetwork({
      layers: [2, 4, 1],
      activation: 'relu',
    })

    const data = Array.from({ length: 100 }, () => ({
      input: [Math.random(), Math.random()],
      output: [Math.random()],
    }))

    const metrics = nn.train(data, {
      epochs: 1000,
      validation: 0.2,
      earlyStop: true,
      patience: 5,
      verbose: false,
    })

    // Should stop before 1000 epochs
    expect(metrics.length).toBeLessThan(1000)
  })

  test('serialize network', () => {
    const nn = new NeuralNetwork({
      layers: [2, 3, 1],
      activation: 'relu',
    })

    const json = nn.toJSON()
    expect(json).toBeDefined()
    expect(typeof json).toBe('object')
  })

  test('network summary', () => {
    const nn = new NeuralNetwork({
      layers: [10, 20, 10, 1],
      activation: 'relu',
    })

    // Should not throw
    expect(() => nn.summary()).not.toThrow()
  })

  test('batch training', () => {
    const nn = new NeuralNetwork({
      layers: [2, 4, 1],
      activation: 'relu',
    })

    const data = Array.from({ length: 100 }, () => ({
      input: [Math.random(), Math.random()],
      output: [Math.random()],
    }))

    const metrics = nn.train(data, {
      epochs: 10,
      batchSize: 10,
      verbose: false,
    })

    expect(metrics.length).toBe(10)
  })

  test('accepts Float32Array and number[] inputs', () => {
    const nn = new NeuralNetwork({
      layers: [2, 1],
      activation: 'sigmoid',
    })

    const output1 = nn.run([0.5, 0.5])
    const output2 = nn.run(new Float32Array([0.5, 0.5]))

    expect(output1[0]).toBeCloseTo(output2[0]!, 6)
  })

  test('throws on incorrect input size', () => {
    const nn = new NeuralNetwork({
      layers: [2, 1],
      activation: 'sigmoid',
    })

    expect(() => nn.run([0.5])).toThrow('Input size mismatch')
    expect(() => nn.run([0.5, 0.5, 0.5])).toThrow('Input size mismatch')
  })
})
