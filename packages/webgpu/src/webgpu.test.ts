/**
 * WebGPU tests
 * Note: These tests require WebGPU support and will be skipped in Node.js/CI environments
 */

import { describe, test, expect, beforeAll } from 'bun:test'
import { initWebGPU, isWebGPUSupported, matmulGPU, addGPU, reluGPU } from './index'

const hasWebGPU = isWebGPUSupported()

// Skip all tests if WebGPU is not supported
const describeWebGPU = hasWebGPU ? describe : describe.skip

describeWebGPU('WebGPU Operations', () => {
  beforeAll(async () => {
    if (hasWebGPU) {
      await initWebGPU()
    }
  })

  test('addGPU performs element-wise addition', async () => {
    const a = new Float32Array([1, 2, 3, 4])
    const b = new Float32Array([5, 6, 7, 8])

    const result = await addGPU(a, b, 4)

    expect(Array.from(result)).toEqual([6, 8, 10, 12])
  })

  test('addGPU works with larger arrays', async () => {
    const size = 1000
    const a = new Float32Array(size).fill(1)
    const b = new Float32Array(size).fill(2)

    const result = await addGPU(a, b, size)

    expect(result.length).toBe(size)
    expect(result[0]).toBe(3)
    expect(result[size - 1]).toBe(3)
  })

  test('reluGPU applies ReLU activation', async () => {
    const input = new Float32Array([-2, -1, 0, 1, 2])

    const result = await reluGPU(input, 5)

    expect(Array.from(result)).toEqual([0, 0, 0, 1, 2])
  })

  test('reluGPU works with larger arrays', async () => {
    const size = 1000
    const input = new Float32Array(size)
    for (let i = 0; i < size; i++) {
      input[i] = i - 500 // Range from -500 to 499
    }

    const result = await reluGPU(input, size)

    // First 500 should be 0 (negative inputs)
    expect(result[0]).toBe(0)
    expect(result[499]).toBe(0)
    // Last 500 should be positive
    expect(result[500]).toBe(0)
    expect(result[999]).toBe(499)
  })

  test('matmulGPU performs matrix multiplication', async () => {
    // A: 2x3 matrix
    const a = new Float32Array([
      1, 2, 3,
      4, 5, 6,
    ])

    // B: 3x2 matrix
    const b = new Float32Array([
      7, 8,
      9, 10,
      11, 12,
    ])

    // Expected C: 2x2 matrix
    // C[0,0] = 1*7 + 2*9 + 3*11 = 7 + 18 + 33 = 58
    // C[0,1] = 1*8 + 2*10 + 3*12 = 8 + 20 + 36 = 64
    // C[1,0] = 4*7 + 5*9 + 6*11 = 28 + 45 + 66 = 139
    // C[1,1] = 4*8 + 5*10 + 6*12 = 32 + 50 + 72 = 154

    const result = await matmulGPU(a, b, 2, 3, 2)

    expect(result.length).toBe(4)
    expect(result[0]).toBe(58)
    expect(result[1]).toBe(64)
    expect(result[2]).toBe(139)
    expect(result[3]).toBe(154)
  })

  test('matmulGPU works with square matrices', async () => {
    // 3x3 identity matrix
    const identity = new Float32Array([
      1, 0, 0,
      0, 1, 0,
      0, 0, 1,
    ])

    // 3x3 test matrix
    const test = new Float32Array([
      1, 2, 3,
      4, 5, 6,
      7, 8, 9,
    ])

    const result = await matmulGPU(identity, test, 3, 3, 3)

    // Identity * Test should equal Test
    expect(Array.from(result)).toEqual(Array.from(test))
  })

  test('matmulGPU handles larger matrices', async () => {
    const m = 32
    const k = 32
    const n = 32

    // Create simple matrices for testing
    const a = new Float32Array(m * k).fill(1)
    const b = new Float32Array(k * n).fill(2)

    const result = await matmulGPU(a, b, m, k, n)

    expect(result.length).toBe(m * n)
    // Each element should be sum of 32 multiplications: 1 * 2 * 32 = 64
    expect(result[0]).toBe(64)
    expect(result[m * n - 1]).toBe(64)
  })
})

// Info test that always runs
test('WebGPU support detection', () => {
  const supported = isWebGPUSupported()
  console.log(`WebGPU supported: ${supported}`)

  if (!supported) {
    console.log('WebGPU tests will be skipped (not supported in this environment)')
  }

  expect(typeof supported).toBe('boolean')
})
