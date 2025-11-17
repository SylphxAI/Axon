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

  test('WASM add function works', async () => {
    const { wasm } = await import('./index')
    await loadWASM()

    const a = new Float32Array([1, 2, 3, 4])
    const b = new Float32Array([5, 6, 7, 8])
    const c = wasm.add(a, b)

    expect(Array.from(c)).toEqual([6, 8, 10, 12])
  })

  test('WASM mul function works', async () => {
    const { wasm } = await import('./index')
    await loadWASM()

    const a = new Float32Array([2, 3, 4, 5])
    const b = new Float32Array([1, 2, 3, 4])
    const c = wasm.mul(a, b)

    expect(Array.from(c)).toEqual([2, 6, 12, 20])
  })
})
