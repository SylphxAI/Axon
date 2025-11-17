# Architecture

## System Overview

NeuronLine uses a **pure functional monorepo architecture** with explicit state management. All operations return new tensors; state updates are explicit.

Data flows: **Input → Tensor → Layer (pure function) → Output Tensor → Loss → Backward → Gradients → Optimizer → Updated State**

Key decision: **Immutability over performance** for predictability and debugging. Performance recovered through algorithmic optimization (tiling, unrolling).

## Key Components

<!-- VERIFY: packages/tensor/ -->
- **@neuronline/tensor** (`packages/tensor/`): Core tensor operations, autograd engine, computational graph
<!-- VERIFY: packages/functional/ -->
- **@neuronline/functional** (`packages/functional/`): Activation functions, loss functions (stateless)
<!-- VERIFY: packages/nn/ -->
- **@neuronline/nn** (`packages/nn/`): Neural network layers (Linear, Conv2D, LSTM, BatchNorm, Dropout)
<!-- VERIFY: packages/optim/ -->
- **@neuronline/optim** (`packages/optim/`): Optimizers with state (SGD, Adam, RMSprop, AdaGrad)
<!-- VERIFY: packages/data/ -->
- **@neuronline/data** (`packages/data/`): Data loaders, batching, normalization
<!-- VERIFY: packages/wasm/ -->
- **@neuronline/wasm** (`packages/wasm/`): AssemblyScript WASM operations (optional acceleration)
<!-- VERIFY: packages/webgpu/ -->
- **@neuronline/webgpu** (`packages/webgpu/`): GPU compute shaders (optional acceleration)

## Design Patterns

### Pattern: Pure Functional State

**Why**: Eliminates entire classes of bugs (mutation, shared state). Makes testing and debugging trivial.

**Where**: All tensor operations, layer forward passes, optimizer updates

**Implementation**:
```typescript
// Layer returns new state, doesn't mutate
const { output, state: newState } = layer.forward(input, oldState)

// Optimizer returns new model, doesn't mutate
const newModel = optimizer.step(oldModel, optimizerState)
```

**Trade-off**:
- **Gained**: Predictability, testability, time-travel debugging, easy serialization
- **Lost**: Raw performance (offset by algorithmic optimization)

### Pattern: Computational Graph for Autograd

**Why**: Automatic differentiation without explicit backward passes. PyTorch-like.

**Where**: `packages/tensor/src/ops.ts` - every operation builds graph

**Implementation**:
```typescript
const result = {
  data: newData,
  shape,
  requiresGrad,
  gradFn: {
    name: 'operation',
    inputs: [a, b],
    backward: (grad) => [gradA, gradB]
  }
}
```

**Trade-off**:
- **Gained**: Automatic gradients, composability, user doesn't write backward passes
- **Lost**: Memory overhead for graph, slightly slower than hand-coded backward

### Pattern: Tiled Matrix Multiplication

**Why**: CPU L1 cache is ~32KB. Naive matmul has poor cache locality.

**Where**: `packages/tensor/src/ops.ts` matmul function

**Implementation**: 32x32 tile blocking with 4x inner loop unrolling

**Trade-off**:
- **Gained**: +22% performance, consistent perf across matrix sizes
- **Lost**: Code complexity, harder to understand

### Pattern: Monorepo with Workspace Dependencies

**Why**: Share code without publishing, tree-shakeable, clear boundaries

**Where**: Root `package.json` with `workspace:*` protocol

**Trade-off**:
- **Gained**: DX, atomic changes, shared types
- **Lost**: Slower CI (build all), complex publish process

## Boundaries

**Package boundaries**:
- `tensor`: No dependencies on other packages (foundation)
- `functional`: Depends only on `tensor`
- `nn`: Depends on `tensor`, `functional`
- `optim`: Depends on `tensor`
- `data`: Depends on `tensor`
- `wasm`, `webgpu`: Optional, depend on `tensor`

**Acceleration boundaries**:
- WASM: For operations >10K elements (overhead too high for small tensors)
- WebGPU: For operations >100K elements (GPU transfer overhead)
- CPU (default): For operations <10K elements

## Key Decisions

See `decisions/` directory for ADRs on:
- Pure functional architecture (ADR-001)
- Tiled matmul vs naive (ADR-002)
- WASM/WebGPU as optional packages (ADR-003)
