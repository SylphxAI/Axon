/**
 * Pure tensor operations with autograd support
 */

import type { Tensor, GradFn } from './types'
import { tensor } from './creation'

/**
 * Element-wise addition with broadcasting
 * Pure function with autograd support
 *
 * Broadcasting rules:
 * - Scalar (shape [1]) broadcasts to any shape
 * - 1D (shape [n]) broadcasts to 2D (shape [m, n]) along last dimension
 * - Same shapes: element-wise addition
 */
export function add(a: Tensor, b: Tensor): Tensor {
  const requiresGrad = a.requiresGrad || b.requiresGrad

  // Case 1: Same shape - simple element-wise
  if (a.shape.length === b.shape.length && a.data.length === b.data.length) {
    const data = new Float32Array(a.data.length)
    for (let i = 0; i < data.length; i++) {
      data[i] = a.data[i]! + b.data[i]!
    }

    const gradFn: GradFn | undefined = requiresGrad
      ? {
          name: 'add',
          inputs: [a, b],
          backward: (grad: Tensor) => [grad, grad],
        }
      : undefined

    return { data, shape: a.shape, requiresGrad, gradFn }
  }

  // Case 2: Scalar broadcasting (shape [1])
  if (b.data.length === 1) {
    const data = new Float32Array(a.data.length)
    const bValue = b.data[0]!
    for (let i = 0; i < data.length; i++) {
      data[i] = a.data[i]! + bValue
    }

    const gradFn: GradFn | undefined = requiresGrad
      ? {
          name: 'add_scalar',
          inputs: [a, b],
          backward: (grad: Tensor) => {
            // Sum all gradients for scalar
            let sum = 0
            for (let i = 0; i < grad.data.length; i++) {
              sum += grad.data[i]!
            }
            return [grad, { ...b, data: new Float32Array([sum]), requiresGrad: false }]
          },
        }
      : undefined

    return { data, shape: a.shape, requiresGrad, gradFn }
  }

  if (a.data.length === 1) {
    const data = new Float32Array(b.data.length)
    const aValue = a.data[0]!
    for (let i = 0; i < data.length; i++) {
      data[i] = aValue + b.data[i]!
    }

    const gradFn: GradFn | undefined = requiresGrad
      ? {
          name: 'add_scalar',
          inputs: [a, b],
          backward: (grad: Tensor) => {
            let sum = 0
            for (let i = 0; i < grad.data.length; i++) {
              sum += grad.data[i]!
            }
            return [{ ...a, data: new Float32Array([sum]), requiresGrad: false }, grad]
          },
        }
      : undefined

    return { data, shape: b.shape, requiresGrad, gradFn }
  }

  // Case 3: 1D broadcasts to 2D (e.g., bias + layer output)
  // a: [m, n], b: [n] -> broadcast b across rows
  if (a.shape.length === 2 && b.shape.length === 1) {
    const [rows, cols] = a.shape
    if (cols !== b.data.length) {
      throw new Error(`Cannot broadcast ${b.shape} to ${a.shape}`)
    }

    const data = new Float32Array(a.data.length)
    for (let i = 0; i < rows!; i++) {
      for (let j = 0; j < cols!; j++) {
        data[i * cols! + j] = a.data[i * cols! + j]! + b.data[j]!
      }
    }

    const gradFn: GradFn | undefined = requiresGrad
      ? {
          name: 'add_broadcast',
          inputs: [a, b],
          backward: (grad: Tensor) => {
            // Gradient for a: same shape as grad
            // Gradient for b: sum across rows
            const bGrad = new Float32Array(cols!)
            for (let i = 0; i < rows!; i++) {
              for (let j = 0; j < cols!; j++) {
                bGrad[j] = bGrad[j]! + grad.data[i * cols! + j]!
              }
            }
            return [grad, { ...b, data: bGrad, requiresGrad: false }]
          },
        }
      : undefined

    return { data, shape: a.shape, requiresGrad, gradFn }
  }

  // Case 4: 2D broadcasts to 1D (reverse)
  if (a.shape.length === 1 && b.shape.length === 2) {
    const [rows, cols] = b.shape
    if (cols !== a.data.length) {
      throw new Error(`Cannot broadcast ${a.shape} to ${b.shape}`)
    }

    const data = new Float32Array(b.data.length)
    for (let i = 0; i < rows!; i++) {
      for (let j = 0; j < cols!; j++) {
        data[i * cols! + j] = a.data[j]! + b.data[i * cols! + j]!
      }
    }

    const gradFn: GradFn | undefined = requiresGrad
      ? {
          name: 'add_broadcast',
          inputs: [a, b],
          backward: (grad: Tensor) => {
            const aGrad = new Float32Array(cols!)
            for (let i = 0; i < rows!; i++) {
              for (let j = 0; j < cols!; j++) {
                aGrad[j] = aGrad[j]! + grad.data[i * cols! + j]!
              }
            }
            return [{ ...a, data: aGrad, requiresGrad: false }, grad]
          },
        }
      : undefined

    return { data, shape: b.shape, requiresGrad, gradFn }
  }

  throw new Error(`Broadcasting not supported for shapes ${a.shape} and ${b.shape}`)
}

/**
 * Element-wise subtraction
 * Pure function with autograd support
 */
export function sub(a: Tensor, b: Tensor): Tensor {
  const requiresGrad = a.requiresGrad || b.requiresGrad
  const data = new Float32Array(a.data.length)

  for (let i = 0; i < data.length; i++) {
    data[i] = a.data[i]! - b.data[i]!
  }

  const gradFn: GradFn | undefined = requiresGrad
    ? {
        name: 'sub',
        inputs: [a, b],
        backward: (grad: Tensor) => {
          // d(a-b)/da = 1, d(a-b)/db = -1
          const negGrad = mul(grad, tensor([-1]))
          return [grad, negGrad]
        },
      }
    : undefined

  return {
    data,
    shape: a.shape,
    requiresGrad,
    gradFn,
  }
}

/**
 * Element-wise multiplication
 * Pure function with autograd support
 */
export function mul(a: Tensor, b: Tensor): Tensor {
  const requiresGrad = a.requiresGrad || b.requiresGrad
  const data = new Float32Array(Math.max(a.data.length, b.data.length))

  // Broadcasting support
  if (a.shape.length === b.shape.length && a.data.length === b.data.length) {
    for (let i = 0; i < data.length; i++) {
      data[i] = a.data[i]! * b.data[i]!
    }
  } else if (b.shape[0] === 1) {
    const bValue = b.data[0]!
    for (let i = 0; i < a.data.length; i++) {
      data[i] = a.data[i]! * bValue
    }
  } else if (a.shape[0] === 1) {
    const aValue = a.data[0]!
    for (let i = 0; i < b.data.length; i++) {
      data[i] = aValue * b.data[i]!
    }
  }

  const gradFn: GradFn | undefined = requiresGrad
    ? {
        name: 'mul',
        inputs: [a, b],
        backward: (grad: Tensor) => {
          // d(a*b)/da = b, d(a*b)/db = a
          return [mul(grad, b), mul(grad, a)]
        },
      }
    : undefined

  return {
    data,
    shape: a.shape.length >= b.shape.length ? a.shape : b.shape,
    requiresGrad,
    gradFn,
  }
}

/**
 * Matrix multiplication (2D only for now)
 * Pure function with autograd support
 */
export function matmul(a: Tensor, b: Tensor): Tensor {
  if (a.shape.length !== 2 || b.shape.length !== 2) {
    throw new Error('matmul requires 2D tensors')
  }

  const [aRows, aCols] = a.shape
  const [bRows, bCols] = b.shape

  if (aCols !== bRows) {
    throw new Error(`Cannot multiply ${a.shape} × ${b.shape}`)
  }

  const requiresGrad = a.requiresGrad || b.requiresGrad
  const data = new Float32Array(aRows! * bCols!)

  // Matrix multiplication
  for (let i = 0; i < aRows!; i++) {
    for (let j = 0; j < bCols!; j++) {
      let sum = 0
      for (let k = 0; k < aCols!; k++) {
        sum += a.data[i * aCols! + k]! * b.data[k * bCols! + j]!
      }
      data[i * bCols! + j] = sum
    }
  }

  const gradFn: GradFn | undefined = requiresGrad
    ? {
        name: 'matmul',
        inputs: [a, b],
        backward: (grad: Tensor) => {
          // d(A@B)/dA = grad @ B^T
          // d(A@B)/dB = A^T @ grad
          const bT = transpose(b)
          const aT = transpose(a)
          return [matmul(grad, bT), matmul(aT, grad)]
        },
      }
    : undefined

  return {
    data,
    shape: [aRows!, bCols!],
    requiresGrad,
    gradFn,
  }
}

/**
 * Transpose (2D only for now)
 * Pure function
 */
export function transpose(t: Tensor): Tensor {
  if (t.shape.length !== 2) {
    throw new Error('transpose requires 2D tensor')
  }

  const [rows, cols] = t.shape
  const data = new Float32Array(t.data.length)

  for (let i = 0; i < rows!; i++) {
    for (let j = 0; j < cols!; j++) {
      data[j * rows! + i] = t.data[i * cols! + j]!
    }
  }

  const gradFn: GradFn | undefined = t.requiresGrad
    ? {
        name: 'transpose',
        inputs: [t],
        backward: (grad: Tensor) => {
          // Transpose gradient flows back transposed
          return [transpose(grad)]
        },
      }
    : undefined

  return {
    data,
    shape: [cols!, rows!],
    requiresGrad: t.requiresGrad,
    gradFn,
  }
}

/**
 * Sum all elements
 * Pure function with autograd support
 */
export function sum(t: Tensor): Tensor {
  let total = 0
  for (let i = 0; i < t.data.length; i++) {
    total += t.data[i]!
  }

  const gradFn: GradFn | undefined = t.requiresGrad
    ? {
        name: 'sum',
        inputs: [t],
        backward: (grad: Tensor) => {
          // Gradient broadcasts to all elements
          const data = new Float32Array(t.data.length)
          const gradValue = grad.data[0]!
          for (let i = 0; i < data.length; i++) {
            data[i] = gradValue
          }
          return [{ ...t, data }]
        },
      }
    : undefined

  return {
    data: new Float32Array([total]),
    shape: [1],
    requiresGrad: t.requiresGrad,
    gradFn,
  }
}

/**
 * Mean of all elements
 * Pure function with autograd support
 */
export function mean(t: Tensor): Tensor {
  const s = sum(t)
  const n = t.data.length
  return mul(s, tensor([1 / n]))
}

/**
 * Reshape tensor (view, no copy)
 * Pure function
 */
export function reshape(t: Tensor, shape: readonly number[]): Tensor {
  const size = shape.reduce((a, b) => a * b, 1)
  if (size !== t.data.length) {
    throw new Error(`Cannot reshape ${t.shape} to ${shape}`)
  }

  return {
    ...t,
    shape,
  }
}

/**
 * Get scalar value from 1-element tensor
 * Pure function
 */
export function item(t: Tensor): number {
  if (t.data.length !== 1) {
    throw new Error('item() requires single-element tensor')
  }
  return t.data[0]!
}

/**
 * Convert tensor to array
 * Pure function
 */
export function toArray(t: Tensor): number[] | number[][] {
  if (t.shape.length === 1) {
    return Array.from(t.data)
  }
  if (t.shape.length === 2) {
    const [rows, cols] = t.shape
    const result: number[][] = []
    for (let i = 0; i < rows!; i++) {
      const row: number[] = []
      for (let j = 0; j < cols!; j++) {
        row.push(t.data[i * cols! + j]!)
      }
      result.push(row)
    }
    return result
  }
  throw new Error('toArray only supports 1D and 2D tensors')
}

/**
 * Clone tensor (deep copy)
 * Pure function
 */
export function clone(t: Tensor): Tensor {
  return {
    data: new Float32Array(t.data),
    shape: [...t.shape],
    requiresGrad: t.requiresGrad,
    gradFn: t.gradFn,
  }
}
