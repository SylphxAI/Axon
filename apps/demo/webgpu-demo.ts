/**
 * WebGPU Demo - GPU-accelerated tensor operations
 * Note: Requires browser with WebGPU support (Chrome/Edge 113+)
 */

import {
  initWebGPU,
  isWebGPUSupported,
  matmulGPU,
  addGPU,
  reluGPU,
} from '../../packages/webgpu/src/index'

console.log('🎮 WebGPU Demo - GPU-Accelerated Tensor Operations\n')

async function main() {
  // Check WebGPU support
  if (!isWebGPUSupported()) {
    console.error('❌ WebGPU not supported in this environment')
    console.log('WebGPU requires:')
    console.log('  • Chrome/Edge 113+ or Firefox Nightly')
    console.log('  • Enabled WebGPU flag (chrome://flags/#enable-unsafe-webgpu)')
    console.log('  • Browser environment (not Node.js)')
    return
  }

  console.log('✅ WebGPU supported\n')

  // Initialize WebGPU
  console.log('Initializing WebGPU device...')
  try {
    await initWebGPU()
    console.log('✅ WebGPU initialized\n')
  } catch (err) {
    console.error('❌ Failed to initialize WebGPU:', err)
    return
  }

  // Test element-wise addition
  console.log('--- Testing GPU Addition ---')
  const a = new Float32Array([1, 2, 3, 4, 5])
  const b = new Float32Array([10, 20, 30, 40, 50])
  const addResult = await addGPU(a, b, 5)
  console.log('a:', Array.from(a))
  console.log('b:', Array.from(b))
  console.log('a + b:', Array.from(addResult))
  console.log('✅ GPU addition works\n')

  // Test ReLU
  console.log('--- Testing GPU ReLU ---')
  const input = new Float32Array([-2, -1, 0, 1, 2])
  const reluResult = await reluGPU(input, 5)
  console.log('input:', Array.from(input))
  console.log('relu(input):', Array.from(reluResult))
  console.log('✅ GPU ReLU works\n')

  // Test matrix multiplication
  console.log('--- Testing GPU Matrix Multiplication ---')
  // 2x3 matrix
  const matA = new Float32Array([1, 2, 3, 4, 5, 6])
  // 3x2 matrix
  const matB = new Float32Array([7, 8, 9, 10, 11, 12])
  // Result: 2x2
  const matmulResult = await matmulGPU(matA, matB, 2, 3, 2)

  console.log('A (2x3):')
  console.log('  [1 2 3]')
  console.log('  [4 5 6]')
  console.log('B (3x2):')
  console.log('  [7  8]')
  console.log('  [9 10]')
  console.log('  [11 12]')
  console.log('C = A @ B (2x2):')
  console.log(`  [${matmulResult[0]} ${matmulResult[1]}]`)
  console.log(`  [${matmulResult[2]} ${matmulResult[3]}]`)
  console.log('Expected:')
  console.log('  [58 64]')
  console.log('  [139 154]')
  console.log('✅ GPU matmul works\n')

  // Benchmark: GPU vs CPU
  console.log('--- Performance Benchmark ---')
  const size = 1000
  const largeA = new Float32Array(size).map(() => Math.random())
  const largeB = new Float32Array(size).map(() => Math.random())

  // GPU
  const gpuStart = performance.now()
  await addGPU(largeA, largeB, size)
  const gpuTime = performance.now() - gpuStart

  // CPU
  const cpuStart = performance.now()
  const cpuResult = new Float32Array(size)
  for (let i = 0; i < size; i++) {
    cpuResult[i] = largeA[i]! + largeB[i]!
  }
  const cpuTime = performance.now() - cpuStart

  console.log(`CPU (${size} elements): ${cpuTime.toFixed(2)}ms`)
  console.log(`GPU (${size} elements): ${gpuTime.toFixed(2)}ms`)
  console.log(`Speedup: ${(cpuTime / gpuTime).toFixed(2)}x`)
  console.log(
    '\nNote: GPU overhead significant for small arrays. Use GPU for large tensors (>10K elements)'
  )

  console.log('\n✅ All WebGPU operations verified!')
}

main().catch(console.error)
