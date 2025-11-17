# Glossary

## Tensor
**Definition**: Multi-dimensional array with gradient tracking
**Usage**: `packages/tensor/src/types.ts`
**Context**: Core data structure. Contains `data: Float32Array`, `shape: number[]`, `requiresGrad: boolean`, optional `gradFn`

## Autograd
**Definition**: Automatic differentiation via computational graph
**Usage**: `packages/tensor/src/ops.ts`, `packages/tensor/src/backward.ts`
**Context**: Each operation creates a `gradFn` that knows how to compute gradients. `backward()` traverses graph in reverse.

## GradFn
**Definition**: Gradient function attached to tensors
**Usage**: `packages/tensor/src/types.ts`
**Context**: Contains `name`, `inputs`, and `backward(grad) => inputGrads[]`. Enables autograd.

## Layer State
**Definition**: Immutable parameters for a neural network layer
**Usage**: All `packages/nn/src/*.ts` files
**Context**: Pure functional pattern. Layer state is separate from layer operations. Example: `LinearState = { weight, bias }`

## im2col
**Definition**: Image-to-column transformation for efficient convolution
**Usage**: `packages/nn/src/conv2d.ts`
**Context**: Converts convolution into matrix multiplication. Extracts image patches into columns. Standard optimization technique.

## Tiling (Blocking)
**Definition**: Breaking matrix multiplication into cache-sized blocks
**Usage**: `packages/tensor/src/ops.ts` matmul
**Context**: 32x32 tiles fit in L1 cache (~32KB). Dramatically improves cache hit rate.

## Loop Unrolling
**Definition**: Expanding loop iterations to reduce overhead and enable ILP
**Usage**: `packages/tensor/src/ops.ts` matmul (4x), mul/add (8x)
**Context**: Reduces branch prediction overhead, enables instruction-level parallelism, better CPU pipeline utilization.

## Optimizer State
**Definition**: Immutable state for optimizer (momentum, running averages)
**Usage**: `packages/optim/src/*.ts`
**Context**: Optimizers maintain state (e.g., Adam's m and v). State is separate from model parameters.

## Episodes/sec (eps/sec)
**Definition**: Training episodes per second in RL benchmark
**Usage**: `PERFORMANCE.md`, benchmark scripts
**Context**: Primary performance metric. Measured on 2048 DQN training (100 episodes).

## Workspace Protocol
**Definition**: Bun/pnpm workspace dependency syntax `workspace:*`
**Usage**: `package.json` dependencies in monorepo
**Context**: Links local packages without publishing. Enables atomic cross-package changes.
