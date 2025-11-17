/**
 * Pure functional activation functions with autograd support
 */

import type { Tensor, GradFn } from '@sylphx/tensor'
import { acquireBuffer } from '@sylphx/tensor'

/**
 * ReLU activation
 * Pure function with autograd
 */
export function relu(x: Tensor): Tensor {
  const data = acquireBuffer(x.data.length)

  // Unroll by 8 for better performance
  let i = 0
  const len8 = x.data.length - 7
  for (; i < len8; i += 8) {
    data[i] = Math.max(0, x.data[i]!)
    data[i + 1] = Math.max(0, x.data[i + 1]!)
    data[i + 2] = Math.max(0, x.data[i + 2]!)
    data[i + 3] = Math.max(0, x.data[i + 3]!)
    data[i + 4] = Math.max(0, x.data[i + 4]!)
    data[i + 5] = Math.max(0, x.data[i + 5]!)
    data[i + 6] = Math.max(0, x.data[i + 6]!)
    data[i + 7] = Math.max(0, x.data[i + 7]!)
  }

  // Handle remainder
  for (; i < x.data.length; i++) {
    data[i] = Math.max(0, x.data[i]!)
  }

  const gradFn: GradFn | undefined = x.requiresGrad
    ? {
        name: 'relu',
        inputs: [x],
        backward: (grad: Tensor) => {
          const inputGrad = acquireBuffer(x.data.length)

          // Unroll by 8
          let i = 0
          const len8 = x.data.length - 7
          for (; i < len8; i += 8) {
            inputGrad[i] = x.data[i]! > 0 ? grad.data[i]! : 0
            inputGrad[i + 1] = x.data[i + 1]! > 0 ? grad.data[i + 1]! : 0
            inputGrad[i + 2] = x.data[i + 2]! > 0 ? grad.data[i + 2]! : 0
            inputGrad[i + 3] = x.data[i + 3]! > 0 ? grad.data[i + 3]! : 0
            inputGrad[i + 4] = x.data[i + 4]! > 0 ? grad.data[i + 4]! : 0
            inputGrad[i + 5] = x.data[i + 5]! > 0 ? grad.data[i + 5]! : 0
            inputGrad[i + 6] = x.data[i + 6]! > 0 ? grad.data[i + 6]! : 0
            inputGrad[i + 7] = x.data[i + 7]! > 0 ? grad.data[i + 7]! : 0
          }

          // Handle remainder
          for (; i < x.data.length; i++) {
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
  const data = acquireBuffer(x.data.length)

  // Unroll by 8
  let i = 0
  const len8 = x.data.length - 7
  for (; i < len8; i += 8) {
    data[i] = x.data[i]! > 0 ? x.data[i]! : alpha * x.data[i]!
    data[i + 1] = x.data[i + 1]! > 0 ? x.data[i + 1]! : alpha * x.data[i + 1]!
    data[i + 2] = x.data[i + 2]! > 0 ? x.data[i + 2]! : alpha * x.data[i + 2]!
    data[i + 3] = x.data[i + 3]! > 0 ? x.data[i + 3]! : alpha * x.data[i + 3]!
    data[i + 4] = x.data[i + 4]! > 0 ? x.data[i + 4]! : alpha * x.data[i + 4]!
    data[i + 5] = x.data[i + 5]! > 0 ? x.data[i + 5]! : alpha * x.data[i + 5]!
    data[i + 6] = x.data[i + 6]! > 0 ? x.data[i + 6]! : alpha * x.data[i + 6]!
    data[i + 7] = x.data[i + 7]! > 0 ? x.data[i + 7]! : alpha * x.data[i + 7]!
  }

  // Handle remainder
  for (; i < x.data.length; i++) {
    data[i] = x.data[i]! > 0 ? x.data[i]! : alpha * x.data[i]!
  }

  const gradFn: GradFn | undefined = x.requiresGrad
    ? {
        name: 'leakyRelu',
        inputs: [x],
        backward: (grad: Tensor) => {
          const inputGrad = acquireBuffer(x.data.length)

          // Unroll by 8
          let i = 0
          const len8 = x.data.length - 7
          for (; i < len8; i += 8) {
            inputGrad[i] = x.data[i]! > 0 ? grad.data[i]! : alpha * grad.data[i]!
            inputGrad[i + 1] = x.data[i + 1]! > 0 ? grad.data[i + 1]! : alpha * grad.data[i + 1]!
            inputGrad[i + 2] = x.data[i + 2]! > 0 ? grad.data[i + 2]! : alpha * grad.data[i + 2]!
            inputGrad[i + 3] = x.data[i + 3]! > 0 ? grad.data[i + 3]! : alpha * grad.data[i + 3]!
            inputGrad[i + 4] = x.data[i + 4]! > 0 ? grad.data[i + 4]! : alpha * grad.data[i + 4]!
            inputGrad[i + 5] = x.data[i + 5]! > 0 ? grad.data[i + 5]! : alpha * grad.data[i + 5]!
            inputGrad[i + 6] = x.data[i + 6]! > 0 ? grad.data[i + 6]! : alpha * grad.data[i + 6]!
            inputGrad[i + 7] = x.data[i + 7]! > 0 ? grad.data[i + 7]! : alpha * grad.data[i + 7]!
          }

          // Handle remainder
          for (; i < x.data.length; i++) {
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
  const data = acquireBuffer(x.data.length)

  // Unroll by 4 for transcendental functions
  let i = 0
  const len4 = x.data.length - 3
  for (; i < len4; i += 4) {
    data[i] = 1 / (1 + Math.exp(-x.data[i]!))
    data[i + 1] = 1 / (1 + Math.exp(-x.data[i + 1]!))
    data[i + 2] = 1 / (1 + Math.exp(-x.data[i + 2]!))
    data[i + 3] = 1 / (1 + Math.exp(-x.data[i + 3]!))
  }

  // Handle remainder
  for (; i < x.data.length; i++) {
    data[i] = 1 / (1 + Math.exp(-x.data[i]!))
  }

  const gradFn: GradFn | undefined = x.requiresGrad
    ? {
        name: 'sigmoid',
        inputs: [x],
        backward: (grad: Tensor) => {
          const inputGrad = acquireBuffer(x.data.length)

          // Unroll by 4
          let i = 0
          const len4 = x.data.length - 3
          for (; i < len4; i += 4) {
            const s0 = data[i]!
            const s1 = data[i + 1]!
            const s2 = data[i + 2]!
            const s3 = data[i + 3]!
            inputGrad[i] = grad.data[i]! * s0 * (1 - s0)
            inputGrad[i + 1] = grad.data[i + 1]! * s1 * (1 - s1)
            inputGrad[i + 2] = grad.data[i + 2]! * s2 * (1 - s2)
            inputGrad[i + 3] = grad.data[i + 3]! * s3 * (1 - s3)
          }

          // Handle remainder
          for (; i < x.data.length; i++) {
            const s = data[i]!
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
  const data = acquireBuffer(x.data.length)

  // Unroll by 4
  let i = 0
  const len4 = x.data.length - 3
  for (; i < len4; i += 4) {
    data[i] = Math.tanh(x.data[i]!)
    data[i + 1] = Math.tanh(x.data[i + 1]!)
    data[i + 2] = Math.tanh(x.data[i + 2]!)
    data[i + 3] = Math.tanh(x.data[i + 3]!)
  }

  // Handle remainder
  for (; i < x.data.length; i++) {
    data[i] = Math.tanh(x.data[i]!)
  }

  const gradFn: GradFn | undefined = x.requiresGrad
    ? {
        name: 'tanh',
        inputs: [x],
        backward: (grad: Tensor) => {
          const inputGrad = acquireBuffer(x.data.length)

          // Unroll by 4
          let i = 0
          const len4 = x.data.length - 3
          for (; i < len4; i += 4) {
            const t0 = data[i]!
            const t1 = data[i + 1]!
            const t2 = data[i + 2]!
            const t3 = data[i + 3]!
            inputGrad[i] = grad.data[i]! * (1 - t0 * t0)
            inputGrad[i + 1] = grad.data[i + 1]! * (1 - t1 * t1)
            inputGrad[i + 2] = grad.data[i + 2]! * (1 - t2 * t2)
            inputGrad[i + 3] = grad.data[i + 3]! * (1 - t3 * t3)
          }

          // Handle remainder
          for (; i < x.data.length; i++) {
            const t = data[i]!
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
 * Softmax activation
 * Pure function with autograd
 *
 * @param x Input tensor
 * @param dim Dimension to apply softmax (default: -1, last dimension)
 *
 * Examples:
 * - 1D [3] → [3]
 * - 2D [2,3] with dim=-1 → [2,3] (softmax along last dim)
 * - 3D [2,3,4] with dim=-1 → [2,3,4] (softmax along last dim)
 */
export function softmax(x: Tensor, dim: number = -1): Tensor {
  // Normalize dimension
  const normDim = dim < 0 ? x.shape.length + dim : dim

  if (normDim < 0 || normDim >= x.shape.length) {
    throw new Error(`softmax: dim ${dim} out of range for tensor with ${x.shape.length} dimensions`)
  }

  const data = acquireBuffer(x.data.length)

  // Calculate strides for the tensor
  const strides: number[] = []
  let stride = 1
  for (let i = x.shape.length - 1; i >= 0; i--) {
    strides[i] = stride
    stride *= x.shape[i]!
  }

  // Size of the dimension we're applying softmax on
  const dimSize = x.shape[normDim]!

  // Number of independent softmax operations
  const outerSize = x.data.length / dimSize

  // Iterate over all positions except the softmax dimension
  for (let outer = 0; outer < outerSize; outer++) {
    // Convert outer index to multi-dimensional index
    let tempOuter = outer
    const indices: number[] = []
    for (let d = x.shape.length - 1; d >= 0; d--) {
      if (d === normDim) {
        indices[d] = 0
      } else {
        const size = x.shape[d]!
        indices[d] = tempOuter % size
        tempOuter = Math.floor(tempOuter / size)
      }
    }

    // Calculate base offset
    let baseOffset = 0
    for (let d = 0; d < x.shape.length; d++) {
      if (d !== normDim) {
        baseOffset += indices[d]! * strides[d]!
      }
    }

    // Find max for numerical stability
    let max = -Infinity
    for (let i = 0; i < dimSize; i++) {
      const offset = baseOffset + i * strides[normDim]!
      max = Math.max(max, x.data[offset]!)
    }

    // Compute exp and sum
    let sum = 0
    for (let i = 0; i < dimSize; i++) {
      const offset = baseOffset + i * strides[normDim]!
      const exp = Math.exp(x.data[offset]! - max)
      data[offset] = exp
      sum += exp
    }

    // Normalize
    for (let i = 0; i < dimSize; i++) {
      const offset = baseOffset + i * strides[normDim]!
      data[offset] = data[offset]! / sum
    }
  }

  const gradFn: GradFn | undefined = x.requiresGrad
    ? {
        name: 'softmax',
        inputs: [x],
        backward: (grad: Tensor) => {
          // Softmax gradient: y * (grad - (grad · y))
          const inputGrad = acquireBuffer(x.data.length)

          for (let outer = 0; outer < outerSize; outer++) {
            // Convert outer index to multi-dimensional index
            let tempOuter = outer
            const indices: number[] = []
            for (let d = x.shape.length - 1; d >= 0; d--) {
              if (d === normDim) {
                indices[d] = 0
              } else {
                const size = x.shape[d]!
                indices[d] = tempOuter % size
                tempOuter = Math.floor(tempOuter / size)
              }
            }

            // Calculate base offset
            let baseOffset = 0
            for (let d = 0; d < x.shape.length; d++) {
              if (d !== normDim) {
                baseOffset += indices[d]! * strides[d]!
              }
            }

            // Compute dot product grad · y
            let dot = 0
            for (let i = 0; i < dimSize; i++) {
              const offset = baseOffset + i * strides[normDim]!
              dot += grad.data[offset]! * data[offset]!
            }

            // Compute gradient
            for (let i = 0; i < dimSize; i++) {
              const offset = baseOffset + i * strides[normDim]!
              inputGrad[offset] = data[offset]! * (grad.data[offset]! - dot)
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
