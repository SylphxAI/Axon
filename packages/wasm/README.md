# @neuronline/wasm

WebAssembly-accelerated tensor operations for NeuronLine.

## Status: Work in Progress

The WASM loader is functional and works across all JavaScript environments (Node.js, browsers, Deno, Bun), but the actual tensor operations require AssemblyScript loader integration for proper memory management.

**Current State:**
- ✅ Universal WASM loader (inline base64, works everywhere)
- ✅ 1.4KB compiled WASM module
- ✅ Environment detection (browser/Node.js/Bun/Deno)
- ⏳ Memory management interface (requires AssemblyScript loader)
- ⏳ Tensor operation wrappers

**What Works:**
```typescript
import { loadWASM, isWASMLoaded } from '@neuronline/wasm'

// Load WASM module
const wasm = await loadWASM()
console.log(isWASMLoaded()) // true

// Module has all functions defined
console.log(wasm.add)    // [Function]
console.log(wasm.mul)    // [Function]
console.log(wasm.matmul) // [Function]
```

**What Needs Work:**
- AssemblyScript loader integration for memory management
- Proper pointer-based interface for Float32Arrays
- High-level wrapper functions for tensor operations

## Design

The WASM module is embedded inline as base64 (1.4KB) to avoid file I/O and work in all environments:

```typescript
// Inline WASM - works in browsers without file loading
const WASM_BASE64 = '...' // 1.4KB encoded module

// Universal base64 decoder
function decodeBase64(base64: string): Uint8Array {
  if (typeof atob !== 'undefined') {
    // Browser
    return decodeWithAtob(base64)
  }
  if (typeof Buffer !== 'undefined') {
    // Node.js/Bun/Deno
    return new Uint8Array(Buffer.from(base64, 'base64'))
  }
  throw new Error('No base64 decoder available')
}
```

## Future Work

1. **AssemblyScript Loader Integration**
   - Import `@assemblyscript/loader` for proper memory management
   - Create wrapper functions that handle memory allocation
   - Map JavaScript arrays to WASM memory

2. **High-Level API**
   ```typescript
   import { matmulWASM } from '@neuronline/wasm'

   const a = tensor([[1, 2], [3, 4]])
   const b = tensor([[5, 6], [7, 8]])
   const c = matmulWASM(a, b) // Uses WASM acceleration
   ```

3. **Performance Benchmarks**
   - Compare pure TypeScript vs WASM for different matrix sizes
   - Determine crossover point where WASM becomes faster
   - Document when to use WASM acceleration

## Installation

```bash
npm install @neuronline/wasm
```

## Usage (Current)

```typescript
import { loadWASM, getWASM, isWASMLoaded } from '@neuronline/wasm'

// Load the WASM module
await loadWASM()

// Check if loaded
if (isWASMLoaded()) {
  const wasm = getWASM()
  console.log('WASM ready:', wasm)
}
```

## Bundle Size

- Source: ~2KB (TypeScript)
- Built: ~3KB (with inline WASM)
- WASM: 1.4KB (base64 encoded)

## License

MIT
