/**
 * @neuronline/wasm
 * WASM-accelerated tensor operations (universal loader)
 * Works in Node.js, browsers, Deno, and Bun
 */

interface WASMExports {
  matmul(
    a: Float32Array,
    b: Float32Array,
    c: Float32Array,
    m: number,
    k: number,
    n: number
  ): void
  add(a: Float32Array, b: Float32Array, c: Float32Array, len: number): void
  mul(a: Float32Array, b: Float32Array, c: Float32Array, len: number): void
  relu(input: Float32Array, output: Float32Array, len: number): void
  sigmoid(input: Float32Array, output: Float32Array, len: number): void
  tanh(input: Float32Array, output: Float32Array, len: number): void
  memory: WebAssembly.Memory
}

// Inline base64-encoded WASM module (1.4KB)
const WASM_BASE64 = 'AGFzbQEAAAABIwVgBH9/f38AYAN/f38AYAJ/fwF9YAN/f30AYAZ/f39/f38AAg0BA2VudgVhYm9ydAAAAwkIAgMEAAABAQEFAwEAAQc3BwZtYXRtdWwAAwNhZGQABANtdWwABQRyZWx1AAYHc2lnbW9pZAAHBHRhbmgACAZtZW1vcnkCAAwBBAqFCQgtACABIAAoAghBAnZPBEBBoAhB4AhBmApBwAAQAAALIAAoAgQgAUECdGoqAgALLwAgASAAKAIIQQJ2TwRAQaAIQeAIQaMKQcAAEAAACyAAKAIEIAFBAnRqIAI4AgAL/wECCX8BfQNAIAMgC0oEQEEAIQkDQCAFIAlKBEBBACEKA0AgBCAKSgRAIAtBIGoiBiADIAMgBkobIQwgCUEgaiIGIAUgBSAGShshDSAKQSBqIgYgBCAEIAZKGyEOIAshBwNAIAcgDEgEQCAJIQgDQCAIIA1IBEAgAiAFIAdsIAhqEAEhDyAKIQYDQCAGIA5IBEAgDyAAIAQgB2wgBmoQASABIAUgBmwgCGoQAZSSIQ8gBkEBaiEGDAELCyACIAUgB2wgCGogDxACIAhBAWohCAwBCwsgB0EBaiEHDAELCyAKQSBqIQoMAQsLIAlBIGohCQwBCwsgC0EgaiELDAELCwuEAgEDfyADIANBCG9rIQUDQCAEIAVIBEAgAiAEIAAgBBABIAEgBBABkhACIAIgBEEBaiIGIAAgBhABIAEgBhABkhACIAIgBEECaiIGIAAgBhABIAEgBhABkhACIAIgBEEDaiIGIAAgBhABIAEgBhABkhACIAIgBEEEaiIGIAAgBhABIAEgBhABkhACIAIgBEEFaiIGIAAgBhABIAEgBhABkhACIAIgBEEGaiIGIAAgBhABIAEgBhABkhACIAIgBEEHaiIGIAAgBhABIAEgBhABkhACIARBCGohBAwBCwsDQCADIARKBEAgAiAEIAAgBBABIAEgBBABkhACIARBAWohBAwBCwsLhAIBA38gAyADQQhvayEFA0AgBCAFSARAIAIgBCAAIAQQASABIAQQAZQQAiACIARBAWoiBiAAIAYQASABIAYQAZQQAiACIARBAmoiBiAAIAYQASABIAYQAZQQAiACIARBA2oiBiAAIAYQASABIAYQAZQQAiACIARBBGoiBiAAIAYQASABIAYQAZQQAiACIARBBWoiBiAAIAYQASABIAYQAZQQAiACIARBBmoiBiAAIAYQASABIAYQAZQQAiACIARBB2oiBiAAIAYQASABIAYQAZQQAiAEQQhqIQQMAQsLA0AgAyAESgRAIAIgBCAAIAQQASABIAQQAZQQAiAEQQFqIQQMAQsLCzYCAX8BfQNAIAIgA0oEQCABIANDAAAAACAAIAMQASIEIARDAAAAAF0bEAIgA0EBaiEDDAELCwtJAgF/AX0DQCACIANKBEAgASADIAAgAxABIgRDAAAAP5QgBIwgBCAEQwAAAABdG0MAAIA/kpVDAAAAP5IQAiADQQFqIQMMAQsLC5YBAgF9AX8DQCACIARKBEAgASAEAn1D7hSsRiAAIAQQAUMAAABAlCIDQwAAIEFeDQAaQ85rPjggA0MAACDBXQ0AGiADQwAAgDuUQwAAgD+SIgMgA5QiAyADlCIDIAOUIgMgA5QiAyADlCIDIAOUIgMgA5QiAyADlAsiA0MAAIC/kiADQwAAgD+SlRACIARBAWohBAwBCwsLC3EEAEGMCAsBPABBmAgLKwIAAAAkAAAASQBuAGQAZQB4ACAAbwB1AHQAIABvAGYAIAByAGEAbgBnAGUAQcwICwE8AEHYCAsrAgAAACQAAAB+AGwAaQBiAC8AdAB5AHAAZQBkAGEAcgByAGEAeQAuAHQAcw=='

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
