/**
 * Multi-Layer Neural Network with Backpropagation
 */

import { getActivation, type ActivationFunction } from './activation'
import { getLoss, type LossFunction } from './loss'
import { createOptimizer, type Optimizer, type OptimizerState } from './optimizer'
import {
  createLayer,
  layerForward,
  layerBackward,
  updateLayerWeights,
  serializeLayer,
  type LayerState,
} from './layer'

export type NeuralNetworkConfig = {
  // Layer sizes: [input, hidden1, hidden2, ..., output]
  layers: number[]

  // Activation function for hidden layers
  activation?: string

  // Activation for output layer (defaults to same as hidden)
  outputActivation?: string

  // Loss function
  loss?: string

  // Optimizer config
  optimizer?: string
  learningRate?: number
  momentum?: number
  beta1?: number
  beta2?: number
  decay?: number

  // Regularization
  l2?: number
  dropout?: number
  clipValue?: number
}

export type TrainingData = {
  input: Float32Array | number[]
  output: Float32Array | number[]
}

export type TrainingOptions = {
  epochs?: number
  batchSize?: number
  shuffle?: boolean
  verbose?: boolean
  validation?: number // fraction of data for validation
  earlyStop?: boolean
  patience?: number
}

export type TrainingMetrics = {
  epoch: number
  loss: number
  accuracy?: number
  validationLoss?: number
  validationAccuracy?: number
}

export class NeuralNetwork {
  private layers: LayerState[] = []
  private activations: ActivationFunction[] = []
  private outputActivation: ActivationFunction
  private lossFunction: LossFunction
  private optimizer: Optimizer
  private optimizerStates: OptimizerState[] = []
  private config: Required<
    Pick<NeuralNetworkConfig, 'layers' | 'learningRate' | 'l2' | 'dropout' | 'clipValue'>
  >

  constructor(config: NeuralNetworkConfig) {
    if (config.layers.length < 2) {
      throw new Error('Neural network must have at least input and output layers')
    }

    this.config = {
      layers: config.layers,
      learningRate: config.learningRate ?? 0.01,
      l2: config.l2 ?? 0,
      dropout: config.dropout ?? 0,
      clipValue: config.clipValue ?? 5,
    }

    // Create activation functions
    const hiddenActivation = getActivation(config.activation ?? 'relu')
    this.outputActivation = getActivation(config.outputActivation ?? config.activation ?? 'sigmoid')

    // Create layers
    for (let i = 0; i < config.layers.length - 1; i++) {
      const inputSize = config.layers[i]!
      const outputSize = config.layers[i + 1]!
      const isOutputLayer = i === config.layers.length - 2

      this.layers.push(
        createLayer({
          inputSize,
          outputSize,
          activation: isOutputLayer ? this.outputActivation : hiddenActivation,
        })
      )

      this.activations.push(isOutputLayer ? this.outputActivation : hiddenActivation)

      // Initialize optimizer state for each layer
      this.optimizerStates.push({ t: 0 })
    }

    // Create loss function
    this.lossFunction = getLoss(config.loss ?? 'mse')

    // Create optimizer
    this.optimizer = createOptimizer(config.optimizer ?? 'adam', {
      learningRate: this.config.learningRate,
      momentum: config.momentum,
      beta1: config.beta1,
      beta2: config.beta2,
      decay: config.decay,
      clipValue: this.config.clipValue,
    })
  }

  /**
   * Forward pass through entire network
   */
  run(input: Float32Array | number[]): Float32Array {
    const inputArray = input instanceof Float32Array ? input : new Float32Array(input)

    if (inputArray.length !== this.config.layers[0]) {
      throw new Error(
        `Input size mismatch: expected ${this.config.layers[0]}, got ${inputArray.length}`
      )
    }

    let activation = inputArray

    for (let i = 0; i < this.layers.length; i++) {
      activation = layerForward(this.layers[i]!, activation, this.activations[i]!)
    }

    return activation
  }

  /**
   * Predict (alias for run)
   */
  predict(input: Float32Array | number[]): Float32Array {
    return this.run(input)
  }

  /**
   * Train on single example (online learning)
   */
  trainOne(example: TrainingData): number {
    const input = example.input instanceof Float32Array ? example.input : new Float32Array(example.input)
    const target = example.output instanceof Float32Array ? example.output : new Float32Array(example.output)

    // Forward pass
    const output = this.run(input)

    // Compute loss
    let totalLoss = 0
    for (let i = 0; i < output.length; i++) {
      totalLoss += this.lossFunction.compute(output[i]!, target[i]!)
    }

    // Backward pass
    this.backward(target)

    return totalLoss / output.length
  }

  /**
   * Train on batch of examples
   */
  train(data: TrainingData[], options: TrainingOptions = {}): TrainingMetrics[] {
    const epochs = options.epochs ?? 100
    const batchSize = options.batchSize ?? 32
    const shuffle = options.shuffle ?? true
    const verbose = options.verbose ?? true
    const validationSplit = options.validation ?? 0

    // Split data into training and validation
    let trainData = data
    let validationData: TrainingData[] = []

    if (validationSplit > 0) {
      const splitIndex = Math.floor(data.length * (1 - validationSplit))
      trainData = data.slice(0, splitIndex)
      validationData = data.slice(splitIndex)
    }

    const metrics: TrainingMetrics[] = []
    let bestValidationLoss = Number.POSITIVE_INFINITY
    let patienceCounter = 0

    for (let epoch = 0; epoch < epochs; epoch++) {
      // Shuffle data
      if (shuffle) {
        for (let i = trainData.length - 1; i > 0; i--) {
          const j = Math.floor(Math.random() * (i + 1))
          const temp = trainData[i]!
          trainData[i] = trainData[j]!
          trainData[j] = temp
        }
      }

      // Train on batches
      let epochLoss = 0
      let correct = 0

      for (let i = 0; i < trainData.length; i += batchSize) {
        const batch = trainData.slice(i, Math.min(i + batchSize, trainData.length))

        for (const example of batch) {
          const loss = this.trainOne(example)
          epochLoss += loss

          // Calculate accuracy for binary classification
          if (example.output.length === 1) {
            const prediction = this.run(example.input)[0]! > 0.5 ? 1 : 0
            const actual = example.output[0]! > 0.5 ? 1 : 0
            if (prediction === actual) correct++
          }
        }
      }

      const avgLoss = epochLoss / trainData.length
      const accuracy = trainData.length > 0 ? correct / trainData.length : 0

      // Validation
      let validationLoss: number | undefined
      let validationAccuracy: number | undefined

      if (validationData.length > 0) {
        let valLoss = 0
        let valCorrect = 0

        for (const example of validationData) {
          const input = example.input instanceof Float32Array ? example.input : new Float32Array(example.input)
          const target = example.output instanceof Float32Array ? example.output : new Float32Array(example.output)
          const output = this.run(input)

          for (let i = 0; i < output.length; i++) {
            valLoss += this.lossFunction.compute(output[i]!, target[i]!)
          }

          if (example.output.length === 1) {
            const prediction = output[0]! > 0.5 ? 1 : 0
            const actual = target[0]! > 0.5 ? 1 : 0
            if (prediction === actual) valCorrect++
          }
        }

        validationLoss = valLoss / validationData.length
        validationAccuracy = validationData.length > 0 ? valCorrect / validationData.length : 0

        // Early stopping
        if (options.earlyStop) {
          if (validationLoss < bestValidationLoss) {
            bestValidationLoss = validationLoss
            patienceCounter = 0
          } else {
            patienceCounter++
            if (patienceCounter >= (options.patience ?? 10)) {
              if (verbose) {
                console.log(`Early stopping at epoch ${epoch + 1}`)
              }
              break
            }
          }
        }
      }

      metrics.push({
        epoch: epoch + 1,
        loss: avgLoss,
        accuracy,
        validationLoss,
        validationAccuracy,
      })

      if (verbose && (epoch % Math.max(1, Math.floor(epochs / 10)) === 0 || epoch === epochs - 1)) {
        let msg = `Epoch ${epoch + 1}/${epochs} - loss: ${avgLoss.toFixed(4)}`
        if (accuracy > 0) msg += ` - acc: ${(accuracy * 100).toFixed(2)}%`
        if (validationLoss !== undefined) msg += ` - val_loss: ${validationLoss.toFixed(4)}`
        if (validationAccuracy !== undefined)
          msg += ` - val_acc: ${(validationAccuracy * 100).toFixed(2)}%`
        console.log(msg)
      }
    }

    return metrics
  }

  /**
   * Backward pass through network
   */
  private backward(target: Float32Array): void {
    const output = this.layers[this.layers.length - 1]!.lastOutput

    // Compute output gradient
    let gradient = new Float32Array(output.length)
    for (let i = 0; i < output.length; i++) {
      gradient[i] = this.lossFunction.gradient(output[i]!, target[i]!)
    }

    // Backpropagate through layers
    for (let i = this.layers.length - 1; i >= 0; i--) {
      const layer = this.layers[i]!
      const activation = this.activations[i]!

      const { inputGradient, weightGradient, biasGradient } = layerBackward(
        layer,
        gradient,
        activation
      )

      // Add L2 regularization to weight gradients
      if (this.config.l2 > 0) {
        for (let j = 0; j < weightGradient.length; j++) {
          weightGradient[j] = weightGradient[j]! + this.config.l2 * layer.weights[j]!
        }
      }

      // Update weights using optimizer
      const { weights: newWeights, state: newState } = this.optimizer.update(
        layer.weights,
        weightGradient,
        this.optimizerStates[i]!
      )

      const { weights: newBiases } = this.optimizer.update(
        layer.biases,
        biasGradient,
        { t: 0 } // Simple update for biases
      )

      // Update layer
      this.layers[i] = updateLayerWeights(layer, newWeights, newBiases)
      this.optimizerStates[i] = newState

      // Continue backprop (copy to ensure compatible Float32Array type)
      gradient = new Float32Array(inputGradient)
    }
  }

  /**
   * Serialize network to JSON
   */
  toJSON(): object {
    return {
      config: this.config,
      layers: this.layers.map(serializeLayer),
    }
  }

  /**
   * Get network summary
   */
  summary(): void {
    console.log('Neural Network Summary')
    console.log('='.repeat(60))

    let totalParams = 0

    for (let i = 0; i < this.layers.length; i++) {
      const layer = this.layers[i]!
      const inputSize = this.config.layers[i]!
      const outputSize = this.config.layers[i + 1]!
      const params = layer.weights.length + layer.biases.length

      totalParams += params

      console.log(`Layer ${i + 1}: Dense(${inputSize} → ${outputSize})`)
      console.log(`  Activation: ${this.activations[i] === this.outputActivation ? 'output' : 'hidden'}`)
      console.log(`  Parameters: ${params.toLocaleString()}`)
      console.log(`  Weights: ${layer.weights.length.toLocaleString()}`)
      console.log(`  Biases: ${layer.biases.length.toLocaleString()}`)
    }

    console.log('='.repeat(60))
    console.log(`Total Parameters: ${totalParams.toLocaleString()}`)
    console.log(`Memory: ~${(totalParams * 4 / 1024).toFixed(2)} KB`)
  }
}
