# 005. Loop Unrolling for Instruction-Level Parallelism

**Status:** ✅ Accepted
**Date:** 2024-11-17

## Context

Element-wise tensor operations (add, mul, activation functions) dominate forward/backward pass compute time. Modern CPUs have instruction-level parallelism (ILP) that allows multiple independent operations to execute simultaneously, but simple loops don't expose this parallelism to the compiler/CPU.

## Decision

Apply systematic loop unrolling across all hot paths:
- **8x unrolling** for simple arithmetic operations (add, sub, mul, div, square, max)
- **4x unrolling** for transcendental functions (sqrt, exp, log, tanh, sigmoid)
- Explicit remainder loops to handle non-multiples

## Rationale

**Why 8x for arithmetic:**
- Most modern CPUs can execute 4-8 arithmetic operations per cycle
- 8x unrolling exposes enough ILP without excessive code size
- Simple operations have low latency, can pipeline many in parallel

**Why 4x for transcendental:**
- Math.* functions have higher latency (10-20 cycles)
- Less benefit from aggressive unrolling
- Balances ILP with code size and register pressure

**Consistency:**
- Same pattern applied across all packages (tensor, functional, optim, nn)
- Makes code predictable and maintainable
- Easy to identify hot paths (look for unrolled loops)

## Consequences

**Positive:**
- Better CPU utilization through exposed ILP
- Fewer loop overhead instructions (condition checks, increments)
- Consistent performance across different data sizes
- Complements memory pooling (less GC + better compute = faster overall)

**Negative:**
- Increased code size (~3x for unrolled sections)
- More complex to read and maintain
- Requires explicit remainder handling
- Manually written (could be macro/codegen but kept simple)

## Implementation

<!-- VERIFY: packages/tensor/src/ops.ts -->
Example from `packages/tensor/src/ops.ts`:
```typescript
// Unroll by 8 for better performance
let i = 0
const len8 = len - 7
for (; i < len8; i += 8) {
  data[i] = aData[i]! + bData[i]!
  data[i + 1] = aData[i + 1]! + bData[i + 1]!
  // ... 6 more lines
}

// Handle remainder
for (; i < len; i++) {
  data[i] = aData[i]! + bData[i]!
}
```

Applied to:
- `packages/tensor/src/ops.ts`: add, sub, sum, autograd
- `packages/functional/src/activation.ts`: all activations
- `packages/optim/src/*.ts`: Adam, RMSprop, AdaGrad helpers
- `packages/nn/src/*.ts`: Conv2D, BatchNorm, Dropout

## References

- Complements: ADR-004 (Memory Pooling)
- Related: ADR-002 (Tiled Matmul - different optimization, same goal)
