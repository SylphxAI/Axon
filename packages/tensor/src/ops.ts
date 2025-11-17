/**
 * Pure tensor operations with autograd support
 */

import type { Tensor, GradFn } from './types'
import { tensor } from './creation'
import { acquireBuffer } from './pool'

// Optional WASM acceleration (graceful degradation)
let wasmModule: any = null

// Optional WebGPU acceleration (async operations only)
let gpuModule: any = null

/**
 * Load WASM acceleration (optional)
 * Call this once at app startup to enable 2x+ faster matrix multiplication
 * Falls back to pure TypeScript if WASM is unavailable
 *
 * WASM is synchronous and integrates directly into tensor operations
 * Activates for matrices ≥1024 elements (e.g., 32x32)
 */
export async function loadAcceleration(): Promise<boolean> {
  if (wasmModule) return true

  try {
    const { loadWASM, wasm } = await import('@neuronline/wasm')
    await loadWASM()
    wasmModule = wasm
    return true
  } catch (e) {
    // WASM not available - will fall back to pure TS
    return false
  }
}

/**
 * Load WebGPU acceleration (optional)
 * WebGPU provides GPU-accelerated operations but requires async API
 *
 * NOTE: WebGPU operations are async and cannot be integrated into
 * synchronous tensor operations. Use the WebGPU API directly for
 * large batch operations where GPU acceleration is beneficial.
 *
 * Recommended threshold: ≥10,000 elements due to GPU overhead
 */
export async function loadGPUAcceleration(): Promise<boolean> {
  if (gpuModule) return true

  try {
    const webgpu = await import('@neuronline/webgpu')
    await webgpu.initWebGPU()
    gpuModule = webgpu
    return true
  } catch (e) {
    // WebGPU not available (not in browser or not supported)
    return false
  }
}

/**
 * Check if WebGPU is available and initialized
 */
export function isGPUAvailable(): boolean {
  return gpuModule !== null && gpuModule.isWebGPUInitialized()
}

/**
 * Get WebGPU module for direct async operations
 * Use this for large batch operations where GPU acceleration is beneficial
 */
export function getGPU(): any {
  if (!gpuModule) {
    throw new Error('WebGPU not loaded. Call loadGPUAcceleration() first.')
  }
  return gpuModule
}

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
    const data = acquireBuffer(a.data.length)
    const aData = a.data
    const bData = b.data
    const len = data.length

    // Unroll loop by 8 for better performance
    const len8 = len - (len % 8)
    let i = 0
    for (; i < len8; i += 8) {
      data[i] = aData[i]! + bData[i]!
      data[i + 1] = aData[i + 1]! + bData[i + 1]!
      data[i + 2] = aData[i + 2]! + bData[i + 2]!
      data[i + 3] = aData[i + 3]! + bData[i + 3]!
      data[i + 4] = aData[i + 4]! + bData[i + 4]!
      data[i + 5] = aData[i + 5]! + bData[i + 5]!
      data[i + 6] = aData[i + 6]! + bData[i + 6]!
      data[i + 7] = aData[i + 7]! + bData[i + 7]!
    }
    // Handle remainder
    for (; i < len; i++) {
      data[i] = aData[i]! + bData[i]!
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
    const data = acquireBuffer(a.data.length)
    const bValue = b.data[0]!
    const len = data.length

    // Unroll by 8 for better performance
    let i = 0
    const len8 = len - 7
    for (; i < len8; i += 8) {
      data[i] = a.data[i]! + bValue
      data[i + 1] = a.data[i + 1]! + bValue
      data[i + 2] = a.data[i + 2]! + bValue
      data[i + 3] = a.data[i + 3]! + bValue
      data[i + 4] = a.data[i + 4]! + bValue
      data[i + 5] = a.data[i + 5]! + bValue
      data[i + 6] = a.data[i + 6]! + bValue
      data[i + 7] = a.data[i + 7]! + bValue
    }

    // Handle remainder
    for (; i < len; i++) {
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
    const data = acquireBuffer(b.data.length)
    const aValue = a.data[0]!
    const len = data.length

    // Unroll by 8 for better performance
    let i = 0
    const len8 = len - 7
    for (; i < len8; i += 8) {
      data[i] = aValue + b.data[i]!
      data[i + 1] = aValue + b.data[i + 1]!
      data[i + 2] = aValue + b.data[i + 2]!
      data[i + 3] = aValue + b.data[i + 3]!
      data[i + 4] = aValue + b.data[i + 4]!
      data[i + 5] = aValue + b.data[i + 5]!
      data[i + 6] = aValue + b.data[i + 6]!
      data[i + 7] = aValue + b.data[i + 7]!
    }

    // Handle remainder
    for (; i < len; i++) {
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

    const data = acquireBuffer(a.data.length)
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
            const bGrad = acquireBuffer(cols!)
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

    const data = acquireBuffer(b.data.length)
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
            const aGrad = acquireBuffer(cols!)
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
  const data = acquireBuffer(a.data.length)
  const len = data.length

  // Unroll by 8 for better performance
  let i = 0
  const len8 = len - 7
  for (; i < len8; i += 8) {
    data[i] = a.data[i]! - b.data[i]!
    data[i + 1] = a.data[i + 1]! - b.data[i + 1]!
    data[i + 2] = a.data[i + 2]! - b.data[i + 2]!
    data[i + 3] = a.data[i + 3]! - b.data[i + 3]!
    data[i + 4] = a.data[i + 4]! - b.data[i + 4]!
    data[i + 5] = a.data[i + 5]! - b.data[i + 5]!
    data[i + 6] = a.data[i + 6]! - b.data[i + 6]!
    data[i + 7] = a.data[i + 7]! - b.data[i + 7]!
  }

  // Handle remainder
  for (; i < len; i++) {
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
  const data = acquireBuffer(Math.max(a.data.length, b.data.length))

  // Broadcasting support - highly optimized
  if (a.shape.length === b.shape.length && a.data.length === b.data.length) {
    const aData = a.data
    const bData = b.data
    const len = data.length

    // Unroll by 8 for better ILP
    const len8 = len - (len % 8)
    let i = 0
    for (; i < len8; i += 8) {
      data[i] = aData[i]! * bData[i]!
      data[i + 1] = aData[i + 1]! * bData[i + 1]!
      data[i + 2] = aData[i + 2]! * bData[i + 2]!
      data[i + 3] = aData[i + 3]! * bData[i + 3]!
      data[i + 4] = aData[i + 4]! * bData[i + 4]!
      data[i + 5] = aData[i + 5]! * bData[i + 5]!
      data[i + 6] = aData[i + 6]! * bData[i + 6]!
      data[i + 7] = aData[i + 7]! * bData[i + 7]!
    }
    for (; i < len; i++) {
      data[i] = aData[i]! * bData[i]!
    }
  } else if (b.shape[0] === 1) {
    const bValue = b.data[0]!
    const aData = a.data
    const len = a.data.length

    // Unroll by 8
    const len8 = len - (len % 8)
    let i = 0
    for (; i < len8; i += 8) {
      data[i] = aData[i]! * bValue
      data[i + 1] = aData[i + 1]! * bValue
      data[i + 2] = aData[i + 2]! * bValue
      data[i + 3] = aData[i + 3]! * bValue
      data[i + 4] = aData[i + 4]! * bValue
      data[i + 5] = aData[i + 5]! * bValue
      data[i + 6] = aData[i + 6]! * bValue
      data[i + 7] = aData[i + 7]! * bValue
    }
    for (; i < len; i++) {
      data[i] = aData[i]! * bValue
    }
  } else if (a.shape[0] === 1) {
    const aValue = a.data[0]!
    const bData = b.data
    const len = b.data.length

    // Unroll by 8
    const len8 = len - (len % 8)
    let i = 0
    for (; i < len8; i += 8) {
      data[i] = aValue * bData[i]!
      data[i + 1] = aValue * bData[i + 1]!
      data[i + 2] = aValue * bData[i + 2]!
      data[i + 3] = aValue * bData[i + 3]!
      data[i + 4] = aValue * bData[i + 4]!
      data[i + 5] = aValue * bData[i + 5]!
      data[i + 6] = aValue * bData[i + 6]!
      data[i + 7] = aValue * bData[i + 7]!
    }
    for (; i < len; i++) {
      data[i] = aValue * bData[i]!
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
  const rows = aRows!
  const cols = bCols!
  const inner = aCols!

  // Use WASM for medium/large matrices (>1024 elements = 32x32)
  // Benchmark shows 2-2.7x speedup for these sizes
  const totalElements = rows * cols
  const USE_WASM_THRESHOLD = 1024

  if (wasmModule && totalElements >= USE_WASM_THRESHOLD) {
    // WASM-accelerated path (2x+ faster)
    const data = wasmModule.matmul(a.data, b.data, rows, inner, cols)

    const gradFn: GradFn | undefined = requiresGrad
      ? {
          name: 'matmul',
          inputs: [a, b],
          backward: (grad: Tensor) => {
            const bT = transpose(b)
            const aT = transpose(a)
            return [matmul(grad, bT), matmul(aT, grad)]
          },
        }
      : undefined

    return {
      data,
      shape: [rows, cols],
      requiresGrad,
      gradFn,
    }
  }

  // Pure TypeScript path (for small matrices or when WASM unavailable)
  const data = acquireBuffer(rows * cols)
  const aData = a.data
  const bData = b.data

  // Tile size for cache optimization (L1 cache ~32KB)
  const TILE = 32

  // Blocked/tiled matrix multiplication for cache efficiency
  for (let ii = 0; ii < rows; ii += TILE) {
    const iMax = Math.min(ii + TILE, rows)

    for (let jj = 0; jj < cols; jj += TILE) {
      const jMax = Math.min(jj + TILE, cols)

      for (let kk = 0; kk < inner; kk += TILE) {
        const kMax = Math.min(kk + TILE, inner)

        // Compute tile
        for (let i = ii; i < iMax; i++) {
          const aRowOffset = i * inner
          const outRowOffset = i * cols

          for (let j = jj; j < jMax; j++) {
            let sum = data[outRowOffset + j]!

            // Unroll by 4 for better ILP
            let k = kk
            const kMax4 = kMax - 3

            for (; k < kMax4; k += 4) {
              const aIdx = aRowOffset + k
              sum +=
                aData[aIdx]! * bData[k * cols + j]! +
                aData[aIdx + 1]! * bData[(k + 1) * cols + j]! +
                aData[aIdx + 2]! * bData[(k + 2) * cols + j]! +
                aData[aIdx + 3]! * bData[(k + 3) * cols + j]!
            }

            // Handle remainder
            for (; k < kMax; k++) {
              sum += aData[aRowOffset + k]! * bData[k * cols + j]!
            }

            data[outRowOffset + j] = sum
          }
        }
      }
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
  const data = acquireBuffer(t.data.length)

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
  const len = t.data.length

  // Unroll by 8 for better performance
  let i = 0
  const len8 = len - 7
  for (; i < len8; i += 8) {
    total += t.data[i]!
    total += t.data[i + 1]!
    total += t.data[i + 2]!
    total += t.data[i + 3]!
    total += t.data[i + 4]!
    total += t.data[i + 5]!
    total += t.data[i + 6]!
    total += t.data[i + 7]!
  }

  // Handle remainder
  for (; i < len; i++) {
    total += t.data[i]!
  }

  const gradFn: GradFn | undefined = t.requiresGrad
    ? {
        name: 'sum',
        inputs: [t],
        backward: (grad: Tensor) => {
          // Gradient broadcasts to all elements
          const data = acquireBuffer(t.data.length)
          const gradValue = grad.data[0]!
          const dataLen = data.length

          // Unroll by 8
          let j = 0
          const len8 = dataLen - 7
          for (; j < len8; j += 8) {
            data[j] = gradValue
            data[j + 1] = gradValue
            data[j + 2] = gradValue
            data[j + 3] = gradValue
            data[j + 4] = gradValue
            data[j + 5] = gradValue
            data[j + 6] = gradValue
            data[j + 7] = gradValue
          }

          // Handle remainder
          for (; j < dataLen; j++) {
            data[j] = gradValue
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
