/**
 * Pure functions for creating tensors
 */

import type { Tensor, TensorOptions } from './types'

/**
 * Create tensor from data
 * Pure function - returns new tensor
 */
export function tensor(
  data: number[] | number[][] | Float32Array,
  options: TensorOptions = {}
): Tensor {
  const requiresGrad = options.requiresGrad ?? false

  // Flatten nested arrays and infer shape
  let flatData: number[]
  let shape: number[]

  if (data instanceof Float32Array) {
    flatData = Array.from(data)
    shape = [data.length]
  } else if (Array.isArray(data)) {
    if (Array.isArray(data[0])) {
      // 2D array
      const rows = data.length
      const cols = (data[0] as number[]).length
      flatData = (data as number[][]).flat()
      shape = [rows, cols]
    } else {
      // 1D array
      flatData = data as number[]
      shape = [flatData.length]
    }
  } else {
    throw new Error('Invalid data type')
  }

  return {
    data: new Float32Array(flatData),
    shape,
    requiresGrad,
    gradFn: options.gradFn,
  }
}

/**
 * Create tensor filled with zeros
 * Pure function
 */
export function zeros(shape: readonly number[], options: TensorOptions = {}): Tensor {
  const size = shape.reduce((a, b) => a * b, 1)
  return {
    data: new Float32Array(size),
    shape,
    requiresGrad: options.requiresGrad ?? false,
    gradFn: options.gradFn,
  }
}

/**
 * Create tensor filled with ones
 * Pure function
 */
export function ones(shape: readonly number[], options: TensorOptions = {}): Tensor {
  const size = shape.reduce((a, b) => a * b, 1)
  const data = new Float32Array(size)
  for (let i = 0; i < size; i++) {
    data[i] = 1
  }
  return {
    data,
    shape,
    requiresGrad: options.requiresGrad ?? false,
    gradFn: options.gradFn,
  }
}

/**
 * Create tensor filled with a scalar value
 * Pure function
 */
export function full(
  shape: readonly number[],
  value: number,
  options: TensorOptions = {}
): Tensor {
  const size = shape.reduce((a, b) => a * b, 1)
  const data = new Float32Array(size)
  for (let i = 0; i < size; i++) {
    data[i] = value
  }
  return {
    data,
    shape,
    requiresGrad: options.requiresGrad ?? false,
    gradFn: options.gradFn,
  }
}

/**
 * Create scalar tensor
 * Pure function
 */
export function scalar(value: number, options: TensorOptions = {}): Tensor {
  return {
    data: new Float32Array([value]),
    shape: [1],
    requiresGrad: options.requiresGrad ?? false,
    gradFn: options.gradFn,
  }
}

/**
 * Create tensor with random normal distribution (Xavier/He initialization)
 * Pure function
 */
export function randn(shape: readonly number[], options: TensorOptions = {}): Tensor {
  const size = shape.reduce((a, b) => a * b, 1)
  const data = new Float32Array(size)

  // Box-Muller transform for normal distribution
  for (let i = 0; i < size; i += 2) {
    const u1 = Math.random()
    const u2 = Math.random()

    const z0 = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2)
    const z1 = Math.sqrt(-2 * Math.log(u1)) * Math.sin(2 * Math.PI * u2)

    data[i] = z0
    if (i + 1 < size) {
      data[i + 1] = z1
    }
  }

  return {
    data,
    shape,
    requiresGrad: options.requiresGrad ?? false,
    gradFn: options.gradFn,
  }
}

/**
 * Create tensor with Xavier initialization
 * Pure function - good for sigmoid/tanh activations
 */
export function xavierNormal(shape: readonly number[], options: TensorOptions = {}): Tensor {
  if (shape.length !== 2) {
    throw new Error('Xavier initialization requires 2D tensor (weight matrix)')
  }

  const [fanIn, fanOut] = shape
  const std = Math.sqrt(2 / (fanIn! + fanOut!))

  const t = randn(shape, options)
  const data = new Float32Array(t.data.length)
  for (let i = 0; i < t.data.length; i++) {
    data[i] = t.data[i]! * std
  }

  return {
    data,
    shape,
    requiresGrad: options.requiresGrad ?? false,
    gradFn: options.gradFn,
  }
}

/**
 * Create tensor with He initialization
 * Pure function - good for ReLU activations
 */
export function heNormal(shape: readonly number[], options: TensorOptions = {}): Tensor {
  if (shape.length !== 2) {
    throw new Error('He initialization requires 2D tensor (weight matrix)')
  }

  const fanIn = shape[0]!
  const std = Math.sqrt(2 / fanIn)

  const t = randn(shape, options)
  const data = new Float32Array(t.data.length)
  for (let i = 0; i < t.data.length; i++) {
    data[i] = t.data[i]! * std
  }

  return {
    data,
    shape,
    requiresGrad: options.requiresGrad ?? false,
    gradFn: options.gradFn,
  }
}

/**
 * Create tensor with uniform distribution [0, 1)
 * Pure function
 */
export function rand(shape: readonly number[], options: TensorOptions = {}): Tensor {
  const size = shape.reduce((a, b) => a * b, 1)
  const data = new Float32Array(size)
  for (let i = 0; i < size; i++) {
    data[i] = Math.random()
  }
  return {
    data,
    shape,
    requiresGrad: options.requiresGrad ?? false,
    gradFn: options.gradFn,
  }
}

/**
 * Create tensor with uniform distribution [low, high)
 * Pure function
 */
export function uniform(
  shape: readonly number[],
  low: number,
  high: number,
  options: TensorOptions = {}
): Tensor {
  const size = shape.reduce((a, b) => a * b, 1)
  const data = new Float32Array(size)
  const range = high - low
  for (let i = 0; i < size; i++) {
    data[i] = Math.random() * range + low
  }
  return {
    data,
    shape,
    requiresGrad: options.requiresGrad ?? false,
    gradFn: options.gradFn,
  }
}
