# 002. Tiled Matrix Multiplication

**Status:** ✅ Accepted
**Date:** 2024-11-17

## Context

Matrix multiplication is the bottleneck in neural networks (80%+ of compute in forward/backward pass).

Naive triple-loop has poor cache locality. Large matrices thrash L1 cache.

## Decision

Implement **32x32 tiled (blocked) matrix multiplication** with 4x inner loop unrolling.

## Rationale

- L1 cache is ~32KB on modern CPUs
- 32x32 tile of f32 = 4KB, fits comfortably in L1
- Tiling improves cache hit rate from ~40% to ~95%
- 4x unrolling improves ILP (instruction-level parallelism)
- Standard optimization in BLAS libraries (ATLAS, OpenBLAS)

## Consequences

**Positive**:
- +22% performance vs naive implementation
- Consistent performance across matrix sizes
- Scalable to large matrices without degradation

**Negative**:
- Code complexity (3 nested tile loops + 3 nested compute loops)
- Harder to understand and maintain
- Slightly slower for very small matrices (overhead of tiling)

**Performance Data**:
- Baseline (naive): 3.35 eps/sec
- Tiled + unrolled: 4.10 eps/sec (+22.4%)

## Alternatives Considered

1. **Naive triple loop**: Simple but slow, poor cache usage
2. **Strassen algorithm**: O(n^2.8) vs O(n^3) but high overhead for small matrices
3. **WASM SIMD**: Requires WASM, not universally supported
4. **Transposed B matrix**: Improves access pattern but requires extra copy

## References

<!-- VERIFY: packages/tensor/src/ops.ts -->
Implementation: `packages/tensor/src/ops.ts` lines 306-357
<!-- VERIFY: PERFORMANCE.md -->
Performance: `PERFORMANCE.md` v0.1.2 benchmarks
