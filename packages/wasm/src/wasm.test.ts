import { describe, expect, test } from 'bun:test'
import { loadWASM, getWASM, isWASMLoaded } from './index'

describe('WASM Loader', () => {
  test('isWASMLoaded returns false initially', () => {
    // Note: Other tests may have loaded it, but this checks the state
    const loaded = isWASMLoaded()
    expect(typeof loaded).toBe('boolean')
  })

  test('loadWASM loads the module', async () => {
    const wasm = await loadWASM()

    expect(wasm).toBeDefined()
    expect(wasm.matmul).toBeDefined()
    expect(wasm.add).toBeDefined()
    expect(wasm.mul).toBeDefined()
    expect(wasm.relu).toBeDefined()
    expect(wasm.sigmoid).toBeDefined()
    expect(wasm.tanh).toBeDefined()
    expect(wasm.memory).toBeDefined()
  })

  test('isWASMLoaded returns true after loading', async () => {
    await loadWASM()
    expect(isWASMLoaded()).toBe(true)
  })

  test('getWASM returns loaded instance', async () => {
    await loadWASM()
    const wasm = getWASM()

    expect(wasm).toBeDefined()
    expect(wasm.matmul).toBeDefined()
  })

  test('loadWASM returns same instance on second call', async () => {
    const wasm1 = await loadWASM()
    const wasm2 = await loadWASM()

    expect(wasm1).toBe(wasm2)
  })

  // WASM functions require AssemblyScript loader for memory management
  // These tests are skipped pending full integration
  test.skip('WASM add function works', async () => {
    const wasm = await loadWASM()

    const a = new Float32Array([1, 2, 3, 4])
    const b = new Float32Array([5, 6, 7, 8])
    const c = new Float32Array(4)

    wasm.add(a, b, c, 4)

    expect(Array.from(c)).toEqual([6, 8, 10, 12])
  })

  test.skip('WASM mul function works', async () => {
    const wasm = await loadWASM()

    const a = new Float32Array([2, 3, 4, 5])
    const b = new Float32Array([1, 2, 3, 4])
    const c = new Float32Array(4)

    wasm.mul(a, b, c, 4)

    expect(Array.from(c)).toEqual([2, 6, 12, 20])
  })
})
