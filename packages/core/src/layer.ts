/**
 * Dense (Fully Connected) Layer
 */

import type { ActivationFunction } from './activation'

export type LayerConfig = {
  inputSize: number
  outputSize: number
  activation: string | ActivationFunction
  useBias?: boolean
}

export type LayerState = {
  weights: Float32Array // shape: [inputSize, outputSize]
  biases: Float32Array // shape: [outputSize]
  lastInput: Float32Array // cached for backward pass
  lastOutput: Float32Array // cached for backward pass
  lastPreActivation: Float32Array // cached for backward pass
}

/**
 * Initialize layer weights using Xavier initialization
 */
export function createLayer(config: LayerConfig): LayerState {
  const { inputSize, outputSize, useBias = true } = config

  // Xavier initialization: scale = sqrt(2 / (inputSize + outputSize))
  const scale = Math.sqrt(2 / (inputSize + outputSize))

  const weights = new Float32Array(inputSize * outputSize)
  for (let i = 0; i < weights.length; i++) {
    weights[i] = (Math.random() * 2 - 1) * scale
  }

  const biases = useBias ? new Float32Array(outputSize) : new Float32Array(0)

  return {
    weights,
    biases,
    lastInput: new Float32Array(inputSize),
    lastOutput: new Float32Array(outputSize),
    lastPreActivation: new Float32Array(outputSize),
  }
}

/**
 * Forward pass through layer
 */
export function layerForward(
  state: LayerState,
  input: Float32Array,
  activation: ActivationFunction
): Float32Array {
  const inputSize = input.length
  const outputSize = state.lastOutput.length

  // Cache input for backward pass
  state.lastInput.set(input)

  // Compute pre-activation: output = weights^T * input + bias
  for (let j = 0; j < outputSize; j++) {
    let sum = 0
    for (let i = 0; i < inputSize; i++) {
      sum += input[i]! * state.weights[i * outputSize + j]!
    }
    if (state.biases.length > 0) {
      sum += state.biases[j]!
    }
    state.lastPreActivation[j] = sum
    state.lastOutput[j] = activation.forward(sum)
  }

  return state.lastOutput
}

/**
 * Backward pass through layer
 * Returns gradient with respect to input
 */
export function layerBackward(
  state: LayerState,
  outputGradient: Float32Array,
  activation: ActivationFunction
): {
  inputGradient: Float32Array
  weightGradient: Float32Array
  biasGradient: Float32Array
} {
  const inputSize = state.lastInput.length
  const outputSize = state.lastOutput.length

  // Compute gradient of activation
  const activationGradient = new Float32Array(outputSize)
  for (let j = 0; j < outputSize; j++) {
    activationGradient[j] = outputGradient[j]! * activation.derivative(state.lastPreActivation[j]!)
  }

  // Compute gradient with respect to weights
  const weightGradient = new Float32Array(inputSize * outputSize)
  for (let i = 0; i < inputSize; i++) {
    for (let j = 0; j < outputSize; j++) {
      weightGradient[i * outputSize + j] = state.lastInput[i]! * activationGradient[j]!
    }
  }

  // Compute gradient with respect to biases
  const biasGradient = new Float32Array(outputSize)
  if (state.biases.length > 0) {
    biasGradient.set(activationGradient)
  }

  // Compute gradient with respect to input (for backprop to previous layer)
  const inputGradient = new Float32Array(inputSize)
  for (let i = 0; i < inputSize; i++) {
    let sum = 0
    for (let j = 0; j < outputSize; j++) {
      sum += state.weights[i * outputSize + j]! * activationGradient[j]!
    }
    inputGradient[i] = sum
  }

  return { inputGradient, weightGradient, biasGradient }
}

/**
 * Update layer weights
 */
export function updateLayerWeights(
  state: LayerState,
  weightUpdate: Float32Array,
  biasUpdate: Float32Array
): LayerState {
  const newWeights = new Float32Array(state.weights.length)
  for (let i = 0; i < newWeights.length; i++) {
    newWeights[i] = weightUpdate[i]!
  }

  const newBiases = new Float32Array(state.biases.length)
  if (state.biases.length > 0) {
    for (let i = 0; i < newBiases.length; i++) {
      newBiases[i] = biasUpdate[i]!
    }
  }

  return {
    ...state,
    weights: newWeights,
    biases: newBiases,
  }
}

/**
 * Serialize layer to JSON
 */
export function serializeLayer(state: LayerState): object {
  return {
    weights: Array.from(state.weights),
    biases: Array.from(state.biases),
  }
}

/**
 * Deserialize layer from JSON
 */
export function deserializeLayer(data: {
  weights: number[]
  biases: number[]
}): Partial<LayerState> {
  return {
    weights: new Float32Array(data.weights),
    biases: new Float32Array(data.biases),
    lastInput: new Float32Array(Math.sqrt(data.weights.length)),
    lastOutput: new Float32Array(data.biases.length),
    lastPreActivation: new Float32Array(data.biases.length),
  }
}
