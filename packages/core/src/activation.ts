/**
 * Activation Functions for Neural Networks
 * Each function returns both forward pass and derivative
 */

export type ActivationFunction = {
  forward: (x: number) => number
  derivative: (x: number) => number
}

/**
 * ReLU (Rectified Linear Unit)
 * f(x) = max(0, x)
 * f'(x) = 1 if x > 0, else 0
 */
export const relu: ActivationFunction = {
  forward: (x: number) => Math.max(0, x),
  derivative: (x: number) => (x > 0 ? 1 : 0),
}

/**
 * Leaky ReLU
 * f(x) = max(0.01x, x)
 * f'(x) = 1 if x > 0, else 0.01
 */
export const leakyRelu: ActivationFunction = {
  forward: (x: number) => (x > 0 ? x : 0.01 * x),
  derivative: (x: number) => (x > 0 ? 1 : 0.01),
}

/**
 * Sigmoid
 * f(x) = 1 / (1 + e^-x)
 * f'(x) = f(x) * (1 - f(x))
 */
export const sigmoid: ActivationFunction = {
  forward: (x: number) => 1 / (1 + Math.exp(-x)),
  derivative: (x: number) => {
    const fx = 1 / (1 + Math.exp(-x))
    return fx * (1 - fx)
  },
}

/**
 * Tanh
 * f(x) = (e^x - e^-x) / (e^x + e^-x)
 * f'(x) = 1 - f(x)^2
 */
export const tanh: ActivationFunction = {
  forward: (x: number) => Math.tanh(x),
  derivative: (x: number) => {
    const fx = Math.tanh(x)
    return 1 - fx * fx
  },
}

/**
 * Linear (identity)
 * f(x) = x
 * f'(x) = 1
 */
export const linear: ActivationFunction = {
  forward: (x: number) => x,
  derivative: (_x: number) => 1,
}

/**
 * Softmax (for output layer, multi-class)
 * Applied to entire vector, not single values
 */
export function softmax(values: Float32Array): Float32Array {
  const max = Math.max(...values)
  const exps = new Float32Array(values.length)
  let sum = 0

  for (let i = 0; i < values.length; i++) {
    exps[i] = Math.exp(values[i]! - max)
    sum += exps[i]!
  }

  for (let i = 0; i < values.length; i++) {
    exps[i] = exps[i]! / sum
  }

  return exps
}

/**
 * Get activation function by name
 */
export function getActivation(name: string): ActivationFunction {
  switch (name.toLowerCase()) {
    case 'relu':
      return relu
    case 'leakyrelu':
    case 'leaky_relu':
      return leakyRelu
    case 'sigmoid':
      return sigmoid
    case 'tanh':
      return tanh
    case 'linear':
      return linear
    default:
      throw new Error(`Unknown activation function: ${name}`)
  }
}
