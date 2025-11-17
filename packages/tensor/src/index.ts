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
  div,
  sqrt,
  square,
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

// Auto-load WASM acceleration (graceful degradation)
// Provides 2-2.7x speedup for matrix operations
// Falls back silently to pure TypeScript if unavailable
import { loadAcceleration } from './ops'
await loadAcceleration().catch(() => {
  // WASM not available - silently fall back to pure TS
  // No action needed, library works fine without it
})
