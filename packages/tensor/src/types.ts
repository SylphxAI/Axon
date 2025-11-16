/**
 * Pure Functional Tensor Library
 * Core types and data structures
 */

/**
 * Gradient function for autograd
 * Pure function that computes gradients for backward pass
 */
export type GradFn = {
  readonly name: string
  readonly backward: (grad: Tensor) => readonly Tensor[]
  readonly inputs: readonly Tensor[]
}

/**
 * Tensor: Immutable n-dimensional array
 * Core data structure for all operations
 */
export type Tensor = {
  readonly data: Float32Array
  readonly shape: readonly number[]
  readonly requiresGrad: boolean
  readonly grad?: Tensor
  readonly gradFn?: GradFn
}

/**
 * Tensor creation options
 */
export type TensorOptions = {
  readonly requiresGrad?: boolean
  readonly gradFn?: GradFn
}

/**
 * Device type (for future WASM/GPU support)
 */
export type Device = 'cpu' | 'wasm' | 'webgpu'

/**
 * Tensor metadata
 */
export type TensorMeta = {
  readonly device: Device
  readonly dtype: 'float32' // Future: support more dtypes
}
