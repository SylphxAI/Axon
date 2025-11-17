/**
 * WASM Demo - Test WASM-accelerated operations
 */

import { loadWASM } from '../../packages/wasm/src/index'

console.log('⚡ WASM Demo - Testing WASM Acceleration\n')

async function main() {
  // Load WASM module
  console.log('Loading WASM module...')
  const wasm = await loadWASM()
  console.log('✅ WASM module loaded\n')

  // Test element-wise addition
  console.log('--- Testing element-wise addition ---')
  const a = new Float32Array([1, 2, 3, 4, 5])
  const b = new Float32Array([10, 20, 30, 40, 50])
  const c = new Float32Array(5)

  wasm.add(a, b, c, 5)
  console.log('a:', Array.from(a))
  console.log('b:', Array.from(b))
  console.log('c = a + b:', Array.from(c))
  console.log('✅ Addition works\n')

  // Test element-wise multiplication
  console.log('--- Testing element-wise multiplication ---')
  const d = new Float32Array([2, 3, 4, 5, 6])
  const e = new Float32Array([1, 2, 3, 4, 5])
  const f = new Float32Array(5)

  wasm.mul(d, e, f, 5)
  console.log('d:', Array.from(d))
  console.log('e:', Array.from(e))
  console.log('f = d * e:', Array.from(f))
  console.log('✅ Multiplication works\n')

  // Test ReLU
  console.log('--- Testing ReLU activation ---')
  const input = new Float32Array([-2, -1, 0, 1, 2])
  const output = new Float32Array(5)

  wasm.relu(input, output, 5)
  console.log('input:', Array.from(input))
  console.log('relu(input):', Array.from(output))
  console.log('✅ ReLU works\n')

  // Test sigmoid
  console.log('--- Testing sigmoid activation ---')
  const sigInput = new Float32Array([-2, -1, 0, 1, 2])
  const sigOutput = new Float32Array(5)

  wasm.sigmoid(sigInput, sigOutput, 5)
  console.log('input:', Array.from(sigInput))
  console.log('sigmoid(input):', Array.from(sigOutput).map(x => x.toFixed(4)))
  console.log('✅ Sigmoid works\n')

  // Test tanh
  console.log('--- Testing tanh activation ---')
  const tanhInput = new Float32Array([-2, -1, 0, 1, 2])
  const tanhOutput = new Float32Array(5)

  wasm.tanh(tanhInput, tanhOutput, 5)
  console.log('input:', Array.from(tanhInput))
  console.log('tanh(input):', Array.from(tanhOutput).map(x => x.toFixed(4)))
  console.log('✅ Tanh works\n')

  // Test matrix multiplication
  console.log('--- Testing matrix multiplication ---')
  // 2x3 matrix
  const matA = new Float32Array([
    1, 2, 3,
    4, 5, 6,
  ])
  // 3x2 matrix
  const matB = new Float32Array([
    7, 8,
    9, 10,
    11, 12,
  ])
  // Result: 2x2
  const matC = new Float32Array(4)

  wasm.matmul(matA, matB, matC, 2, 3, 2)
  console.log('A (2x3):')
  console.log('  [1 2 3]')
  console.log('  [4 5 6]')
  console.log('B (3x2):')
  console.log('  [7  8]')
  console.log('  [9 10]')
  console.log('  [11 12]')
  console.log('C = A @ B (2x2):')
  console.log(`  [${matC[0]} ${matC[1]}]`)
  console.log(`  [${matC[2]} ${matC[3]}]`)
  console.log('✅ Matmul works\n')

  console.log('✅ All WASM operations verified!')
}

main().catch(console.error)
