/**
 * @sylphx/wasm
 * WASM-accelerated tensor operations (universal loader)
 * Works in Node.js, browsers, Deno, and Bun
 */

interface WASMExports {
  matmul(aPtr: number, bPtr: number, cPtr: number, m: number, k: number, n: number): void
  add(aPtr: number, bPtr: number, cPtr: number, len: number): void
  mul(aPtr: number, bPtr: number, cPtr: number, len: number): void
  relu(inputPtr: number, outputPtr: number, len: number): void
  sigmoid(inputPtr: number, outputPtr: number, len: number): void
  tanh(inputPtr: number, outputPtr: number, len: number): void
  memory: WebAssembly.Memory
}

// Inline base64-encoded WASM module (1.4KB)
const WASM_BASE64 = 'AGFzbQEAAAABFwNgA39/fwBgBH9/f38AYAZ/f39/f38AAwcGAgEBAAAABQMBAAAHNwcGbWF0bXVsAAADYWRkAAEDbXVsAAIEcmVsdQADB3NpZ21vaWQABAR0YW5oAAUGbWVtb3J5AgAK/QkGjAICCn8BfQNAIAMgC0oEQEEAIQcDQCAFIAdKBEBBACEIA0AgBCAISgRAIAtBIGoiBiADIAMgBkobIQwgB0EgaiIGIAUgBSAGShshDSAIQSBqIgYgBCAEIAZKGyEOIAshCQNAIAkgDEgEQCAHIQoDQCAKIA1IBEAgBSAJbCAKakECdCIPIAJqKgIAIRAgCCEGA0AgBiAOSARAIBAgACAEIAlsIAZqQQJ0aioCACABIAUgBmwgCmpBAnRqKgIAlJIhECAGQQFqIQYMAQsLIAIgD2ogEDgCACAKQQFqIQoMAQsLIAlBAWohCQwBCwsgCEEgaiEIDAELCyAHQSBqIQcMAQsLIAtBIGohCwwBCwsL2QIBA38gAyADQQhvayEFA0AgBCAFSARAIARBAnQiBiACaiAAIAZqKgIAIAEgBmoqAgCSOAIAIARBAWpBAnQiBiACaiAAIAZqKgIAIAEgBmoqAgCSOAIAIARBAmpBAnQiBiACaiAAIAZqKgIAIAEgBmoqAgCSOAIAIARBA2pBAnQiBiACaiAAIAZqKgIAIAEgBmoqAgCSOAIAIARBBGpBAnQiBiACaiAAIAZqKgIAIAEgBmoqAgCSOAIAIARBBWpBAnQiBiACaiAAIAZqKgIAIAEgBmoqAgCSOAIAIARBBmpBAnQiBiACaiAAIAZqKgIAIAEgBmoqAgCSOAIAIARBB2pBAnQiBiACaiAAIAZqKgIAIAEgBmoqAgCSOAIAIARBCGohBAwBCwsDQCADIARKBEAgBEECdCIFIAJqIAAgBWoqAgAgASAFaioCAJI4AgAgBEEBaiEEDAELCwvZAgEDfyADIANBCG9rIQUDQCAEIAVIBEAgBEECdCIGIAJqIAAgBmoqAgAgASAGaioCAJQ4AgAgBEEBakECdCIGIAJqIAAgBmoqAgAgASAGaioCAJQ4AgAgBEECakECdCIGIAJqIAAgBmoqAgAgASAGaioCAJQ4AgAgBEEDakECdCIGIAJqIAAgBmoqAgAgASAGaioCAJQ4AgAgBEEEakECdCIGIAJqIAAgBmoqAgAgASAGaioCAJQ4AgAgBEEFakECdCIGIAJqIAAgBmoqAgAgASAGaioCAJQ4AgAgBEEGakECdCIGIAJqIAAgBmoqAgAgASAGaioCAJQ4AgAgBEEHakECdCIGIAJqIAAgBmoqAgAgASAGaioCAJQ4AgAgBEEIaiEEDAELCwNAIAMgBEoEQCAEQQJ0IgUgAmogACAFaioCACABIAVqKgIAlDgCACAEQQFqIQQMAQsLC0ECAn8BfQNAIAIgA0oEQCADQQJ0IgQgAGoqAgAhBSABIARqQwAAAAAgBSAFQwAAAABdGzgCACADQQFqIQMMAQsLC1QCAX0CfwNAIAIgBEoEQCAAIARBAnQiBWoqAgAhAyABIAVqIANDAAAAP5QgA4wgAyADQwAAAABdG0MAAIA/kpVDAAAAP5I4AgAgBEEBaiEEDAELCwufAQIBfQJ/A0AgAiAESgRAIAEgBEECdCIFagJ9Q+4UrEYgACAFaioCAEMAAABAlCIDQwAAIEFeDQAaQ85rPjggA0MAACDBXQ0AGiADQwAAgDuUQwAAgD+SIgMgA5QiAyADlCIDIAOUIgMgA5QiAyADlCIDIAOUIgMgA5QiAyADlAsiA0MAAIC/kiADQwAAgD+SlTgCACAEQQFqIQQMAQsLCw=='

let wasmInstance: WASMExports | null = null

/**
 * Decode base64 to Uint8Array (works in all environments)
 */
function decodeBase64(base64: string): Uint8Array {
  // Browser
  if (typeof atob !== 'undefined') {
    const binary = atob(base64)
    const bytes = new Uint8Array(binary.length)
    for (let i = 0; i < binary.length; i++) {
      bytes[i] = binary.charCodeAt(i)
    }
    return bytes
  }

  // Node.js / Bun / Deno
  if (typeof Buffer !== 'undefined') {
    return new Uint8Array(Buffer.from(base64, 'base64'))
  }

  throw new Error('No base64 decoder available')
}

/**
 * Load WASM module
 * Must be called before using WASM operations
 * Works in Node.js, browsers, Deno, and Bun
 */
export async function loadWASM(): Promise<WASMExports> {
  if (wasmInstance) {
    return wasmInstance
  }

  // Decode inline WASM module
  const wasmBytes = decodeBase64(WASM_BASE64)

  // Provide required environment imports
  const imports = {
    env: {
      abort: () => {
        throw new Error('WASM aborted')
      },
    },
  }

  const wasmModule = await WebAssembly.instantiate(wasmBytes, imports)

  wasmInstance = wasmModule.instance.exports as unknown as WASMExports
  return wasmInstance
}

/**
 * Get loaded WASM instance
 * Throws if WASM not loaded
 */
export function getWASM(): WASMExports {
  if (!wasmInstance) {
    throw new Error('WASM not loaded. Call loadWASM() first.')
  }
  return wasmInstance
}

/**
 * Check if WASM is loaded
 */
export function isWASMLoaded(): boolean {
  return wasmInstance !== null
}

/**
 * Simple memory allocator for WASM
 * Uses fixed offsets in WASM linear memory
 */
class WASMMemoryAllocator {
  private offset = 1024 // Start after WASM runtime data
  private memory: WebAssembly.Memory

  constructor(memory: WebAssembly.Memory) {
    this.memory = memory
  }

  /**
   * Allocate space for a Float32Array
   * Returns byte offset in WASM memory
   */
  alloc(length: number): number {
    const byteLength = length * 4 // Float32 = 4 bytes
    const ptr = this.offset
    this.offset += byteLength

    // Ensure memory is large enough
    const needed = Math.ceil(this.offset / 65536) // 64KB pages
    const current = this.memory.buffer.byteLength / 65536
    if (needed > current) {
      this.memory.grow(needed - current)
    }

    return ptr
  }

  /**
   * Write Float32Array to WASM memory
   */
  write(ptr: number, data: Float32Array): void {
    const view = new Float32Array(this.memory.buffer, ptr, data.length)
    view.set(data)
  }

  /**
   * Read Float32Array from WASM memory
   */
  read(ptr: number, length: number): Float32Array {
    const view = new Float32Array(this.memory.buffer, ptr, length)
    return new Float32Array(view) // Copy to new array
  }

  /**
   * Reset allocator (simple bump allocator)
   */
  reset(): void {
    this.offset = 1024
  }
}

/**
 * High-level WASM operations with automatic memory management
 */
export const wasm = {
  /**
   * Element-wise addition: c = a + b
   */
  add(a: Float32Array, b: Float32Array): Float32Array {
    const instance = getWASM()
    const allocator = new WASMMemoryAllocator(instance.memory)

    const aPtr = allocator.alloc(a.length)
    const bPtr = allocator.alloc(b.length)
    const cPtr = allocator.alloc(a.length)

    allocator.write(aPtr, a)
    allocator.write(bPtr, b)

    instance.add(aPtr, bPtr, cPtr, a.length)

    return allocator.read(cPtr, a.length)
  },

  /**
   * Element-wise multiplication: c = a * b
   */
  mul(a: Float32Array, b: Float32Array): Float32Array {
    const instance = getWASM()
    const allocator = new WASMMemoryAllocator(instance.memory)

    const aPtr = allocator.alloc(a.length)
    const bPtr = allocator.alloc(b.length)
    const cPtr = allocator.alloc(a.length)

    allocator.write(aPtr, a)
    allocator.write(bPtr, b)

    instance.mul(aPtr, bPtr, cPtr, a.length)

    return allocator.read(cPtr, a.length)
  },

  /**
   * Matrix multiplication: C = A @ B
   */
  matmul(a: Float32Array, b: Float32Array, m: number, k: number, n: number): Float32Array {
    const instance = getWASM()
    const allocator = new WASMMemoryAllocator(instance.memory)

    const aPtr = allocator.alloc(m * k)
    const bPtr = allocator.alloc(k * n)
    const cPtr = allocator.alloc(m * n)

    allocator.write(aPtr, a)
    allocator.write(bPtr, b)

    // Initialize output to zero
    const zeros = new Float32Array(m * n)
    allocator.write(cPtr, zeros)

    instance.matmul(aPtr, bPtr, cPtr, m, k, n)

    return allocator.read(cPtr, m * n)
  },

  /**
   * ReLU activation
   */
  relu(input: Float32Array): Float32Array {
    const instance = getWASM()
    const allocator = new WASMMemoryAllocator(instance.memory)

    const inputPtr = allocator.alloc(input.length)
    const outputPtr = allocator.alloc(input.length)

    allocator.write(inputPtr, input)

    instance.relu(inputPtr, outputPtr, input.length)

    return allocator.read(outputPtr, input.length)
  },

  /**
   * Sigmoid activation
   */
  sigmoid(input: Float32Array): Float32Array {
    const instance = getWASM()
    const allocator = new WASMMemoryAllocator(instance.memory)

    const inputPtr = allocator.alloc(input.length)
    const outputPtr = allocator.alloc(input.length)

    allocator.write(inputPtr, input)

    instance.sigmoid(inputPtr, outputPtr, input.length)

    return allocator.read(outputPtr, input.length)
  },

  /**
   * Tanh activation
   */
  tanh(input: Float32Array): Float32Array {
    const instance = getWASM()
    const allocator = new WASMMemoryAllocator(instance.memory)

    const inputPtr = allocator.alloc(input.length)
    const outputPtr = allocator.alloc(input.length)

    allocator.write(inputPtr, input)

    instance.tanh(inputPtr, outputPtr, input.length)

    return allocator.read(outputPtr, input.length)
  },
}
