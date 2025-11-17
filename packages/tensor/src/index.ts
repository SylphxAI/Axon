/**
 * @sylphx/tensor
 * Pure functional tensor library with autograd
 */

// Types
export type { Tensor, TensorOptions, GradFn, Device, TensorMeta } from './types'

// Creation
export {
  tensor,
  zeros,
  ones,
  full,
  scalar,
  randn,
  xavierNormal,
  heNormal,
  rand,
  uniform,
} from './creation'

// Operations
export {
  add,
  sub,
  mul,
  matmul,
  transpose,
  sum,
  mean,
  reshape,
  item,
  toArray,
  clone,
  loadAcceleration,
  loadGPUAcceleration,
  isGPUAvailable,
  getGPU,
} from './ops'

// Autograd
export { backward, zeroGrad, detach, requiresGrad } from './autograd'

// Memory management
export {
  acquireBuffer,
  releaseBuffer,
  clearPool,
  poolStats,
  setPoolingEnabled,
  withScope,
} from './pool'
