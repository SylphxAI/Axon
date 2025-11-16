/**
 * Pure functional activation functions with autograd support
 */

import type { Tensor, GradFn } from '@neuronline/tensor'

/**
 * ReLU activation
 * Pure function with autograd
 */
export function relu(x: Tensor): Tensor {
  const data = new Float32Array(x.data.length)
  for (let i = 0; i < x.data.length; i++) {
    data[i] = Math.max(0, x.data[i]!)
  }

  const gradFn: GradFn | undefined = x.requiresGrad
    ? {
        name: 'relu',
        inputs: [x],
        backward: (grad: Tensor) => {
          const inputGrad = new Float32Array(x.data.length)
          for (let i = 0; i < x.data.length; i++) {
            inputGrad[i] = x.data[i]! > 0 ? grad.data[i]! : 0
          }
          return [{ ...x, data: inputGrad, requiresGrad: false }]
        },
      }
    : undefined

  return {
    data,
    shape: x.shape,
    requiresGrad: x.requiresGrad,
    gradFn,
  }
}

/**
 * Leaky ReLU activation
 * Pure function with autograd
 */
export function leakyRelu(x: Tensor, alpha = 0.01): Tensor {
  const data = new Float32Array(x.data.length)
  for (let i = 0; i < x.data.length; i++) {
    data[i] = x.data[i]! > 0 ? x.data[i]! : alpha * x.data[i]!
  }

  const gradFn: GradFn | undefined = x.requiresGrad
    ? {
        name: 'leakyRelu',
        inputs: [x],
        backward: (grad: Tensor) => {
          const inputGrad = new Float32Array(x.data.length)
          for (let i = 0; i < x.data.length; i++) {
            inputGrad[i] = x.data[i]! > 0 ? grad.data[i]! : alpha * grad.data[i]!
          }
          return [{ ...x, data: inputGrad, requiresGrad: false }]
        },
      }
    : undefined

  return {
    data,
    shape: x.shape,
    requiresGrad: x.requiresGrad,
    gradFn,
  }
}

/**
 * Sigmoid activation
 * Pure function with autograd
 */
export function sigmoid(x: Tensor): Tensor {
  const data = new Float32Array(x.data.length)
  for (let i = 0; i < x.data.length; i++) {
    data[i] = 1 / (1 + Math.exp(-x.data[i]!))
  }

  const gradFn: GradFn | undefined = x.requiresGrad
    ? {
        name: 'sigmoid',
        inputs: [x],
        backward: (grad: Tensor) => {
          const inputGrad = new Float32Array(x.data.length)
          for (let i = 0; i < x.data.length; i++) {
            const s = data[i]! // sigmoid(x)
            inputGrad[i] = grad.data[i]! * s * (1 - s)
          }
          return [{ ...x, data: inputGrad, requiresGrad: false }]
        },
      }
    : undefined

  return {
    data,
    shape: x.shape,
    requiresGrad: x.requiresGrad,
    gradFn,
  }
}

/**
 * Tanh activation
 * Pure function with autograd
 */
export function tanh(x: Tensor): Tensor {
  const data = new Float32Array(x.data.length)
  for (let i = 0; i < x.data.length; i++) {
    data[i] = Math.tanh(x.data[i]!)
  }

  const gradFn: GradFn | undefined = x.requiresGrad
    ? {
        name: 'tanh',
        inputs: [x],
        backward: (grad: Tensor) => {
          const inputGrad = new Float32Array(x.data.length)
          for (let i = 0; i < x.data.length; i++) {
            const t = data[i]! // tanh(x)
            inputGrad[i] = grad.data[i]! * (1 - t * t)
          }
          return [{ ...x, data: inputGrad, requiresGrad: false }]
        },
      }
    : undefined

  return {
    data,
    shape: x.shape,
    requiresGrad: x.requiresGrad,
    gradFn,
  }
}

/**
 * Softmax activation (along last dimension)
 * Pure function with autograd
 */
export function softmax(x: Tensor): Tensor {
  if (x.shape.length !== 2) {
    throw new Error('softmax currently only supports 2D tensors')
  }

  const [rows, cols] = x.shape
  const data = new Float32Array(x.data.length)

  // For each row
  for (let i = 0; i < rows!; i++) {
    // Find max for numerical stability
    let max = -Infinity
    for (let j = 0; j < cols!; j++) {
      max = Math.max(max, x.data[i * cols! + j]!)
    }

    // Compute exp and sum
    let sum = 0
    for (let j = 0; j < cols!; j++) {
      const exp = Math.exp(x.data[i * cols! + j]! - max)
      data[i * cols! + j] = exp
      sum += exp
    }

    // Normalize
    for (let j = 0; j < cols!; j++) {
      data[i * cols! + j] = data[i * cols! + j]! / sum
    }
  }

  const gradFn: GradFn | undefined = x.requiresGrad
    ? {
        name: 'softmax',
        inputs: [x],
        backward: (grad: Tensor) => {
          // Softmax gradient: y * (grad - (grad · y))
          const [rows, cols] = x.shape
          const inputGrad = new Float32Array(x.data.length)

          for (let i = 0; i < rows!; i++) {
            // Compute dot product grad · y
            let dot = 0
            for (let j = 0; j < cols!; j++) {
              dot += grad.data[i * cols! + j]! * data[i * cols! + j]!
            }

            // Compute gradient
            for (let j = 0; j < cols!; j++) {
              inputGrad[i * cols! + j] =
                data[i * cols! + j]! * (grad.data[i * cols! + j]! - dot)
            }
          }

          return [{ ...x, data: inputGrad, requiresGrad: false }]
        },
      }
    : undefined

  return {
    data,
    shape: x.shape,
    requiresGrad: x.requiresGrad,
    gradFn,
  }
}
