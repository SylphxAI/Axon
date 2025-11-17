/**
 * @neuronline/wasm
 * WASM-accelerated tensor operations (loader)
 */

import { readFileSync } from 'fs'
import { join } from 'path'

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

let wasmInstance: WASMExports | null = null

/**
 * Load WASM module
 * Must be called before using WASM operations
 */
export async function loadWASM(): Promise<WASMExports> {
  if (wasmInstance) {
    return wasmInstance
  }

  const wasmPath = join(__dirname, '../build/neuronline.wasm')
  const wasmBuffer = readFileSync(wasmPath)

  // Provide required environment imports
  const imports = {
    env: {
      abort: () => {
        throw new Error('WASM aborted')
      },
    },
  }

  const wasmModule = await WebAssembly.instantiate(wasmBuffer, imports)

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
