# 003. WASM/WebGPU as Optional Packages

**Status:** ✅ Accepted
**Date:** 2024-11-17

## Context

Need hardware acceleration for large models, but not all environments support WASM/WebGPU.

Options:
1. Bundle WASM/WebGPU in core (bloats bundle, fails in unsupported envs)
2. Separate optional packages (user imports if needed)
3. Dynamic detection and loading (complex, unreliable)

## Decision

Create **separate optional packages** (`@neuronline/wasm`, `@neuronline/webgpu`) that users explicitly import.

## Rationale

- **Bundle size**: Core stays <20KB, acceleration adds ~5KB each
- **Environment compatibility**: Core works everywhere, acceleration opt-in
- **Progressive enhancement**: Fast by default, faster with acceleration
- **Clear dependencies**: User knows what they're importing
- **Tree-shakeable**: Bundlers can eliminate unused acceleration

## Consequences

**Positive**:
- Core library works in all environments (even IE11 with polyfills)
- Zero overhead if acceleration not used
- Users control bundle size
- Easy to test (can test with/without acceleration)

**Negative**:
- Users must explicitly import acceleration
- Potential API duplication (need wrapper for accelerated ops)
- Documentation must explain when to use acceleration

**Usage Pattern**:
```typescript
// Core (always works)
import * as T from '@neuronline/tensor'
const result = T.matmul(a, b)

// With WASM (opt-in)
import { loadWASM } from '@neuronline/wasm'
await loadWASM()
// Now tensor ops use WASM when beneficial

// With WebGPU (opt-in, browser only)
import { initWebGPU, matmulGPU } from '@neuronline/webgpu'
await initWebGPU()
const result = await matmulGPU(a, b, m, k, n)
```

## References

<!-- VERIFY: packages/wasm/ -->
WASM: `packages/wasm/` (AssemblyScript implementation)
<!-- VERIFY: packages/webgpu/ -->
WebGPU: `packages/webgpu/` (Compute shaders)
<!-- VERIFY: packages/tensor/ -->
Core: `packages/tensor/` (No dependency on wasm/webgpu)
