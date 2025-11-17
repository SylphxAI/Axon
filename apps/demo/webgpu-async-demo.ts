/**
 * WebGPU Async Operations Demo
 * Shows how to use WebGPU for large batch operations
 *
 * NOTE: This demo requires a browser environment with WebGPU support
 * Run in browser console or use a browser-based test runner
 */

import * as T from '@neuronline/tensor'

async function main() {
  console.log('🎮 WebGPU Async Operations Demo\n')

  // Load accelerations
  console.log('Loading acceleration modules...')
  const wasmLoaded = await T.loadAcceleration()
  const gpuLoaded = await T.loadGPUAcceleration()

  console.log(`WASM: ${wasmLoaded ? '✅ ENABLED' : '❌ DISABLED'}`)
  console.log(`WebGPU: ${gpuLoaded ? '✅ ENABLED' : '❌ DISABLED'}\n`)

  if (!gpuLoaded) {
    console.log('⚠️  WebGPU not available. This demo requires WebGPU support.')
    console.log('Please run in a browser with WebGPU enabled (Chrome, Edge, etc.)\n')
    return
  }

  // Example 1: Sync WASM operations (integrated into tensor API)
  console.log('📊 Example 1: Synchronous WASM Operations\n')
  console.log('WASM integrates directly into tensor operations - no code changes needed!')

  const a = T.randn([32, 64])
  const b = T.randn([64, 128])

  const start1 = performance.now()
  const c = T.matmul(a, b) // Automatically uses WASM if matrices >= 1024 elements
  const end1 = performance.now()

  console.log(`  Matrix multiplication: ${a.shape} @ ${b.shape} = ${c.shape}`)
  console.log(`  Time: ${(end1 - start1).toFixed(4)}ms`)
  console.log(
    `  Acceleration: ${c.shape[0]! * c.shape[1]! >= 1024 ? 'WASM ⚡' : 'TypeScript'}\n`
  )

  // Example 2: Async WebGPU operations (requires explicit GPU API)
  console.log('📊 Example 2: Asynchronous WebGPU Operations\n')
  console.log('WebGPU requires using the GPU API directly for async operations:')

  const gpu = T.getGPU()
  const m = 128
  const k = 256
  const n = 512

  // Prepare data
  const aData = new Float32Array(m * k)
  const bData = new Float32Array(k * n)
  for (let i = 0; i < aData.length; i++) aData[i] = Math.random()
  for (let i = 0; i < bData.length; i++) bData[i] = Math.random()

  const start2 = performance.now()
  const result = await gpu.matmulGPU(aData, bData, m, k, n)
  const end2 = performance.now()

  console.log(`  Matrix multiplication: [${m}, ${k}] @ [${k}, ${n}] = [${m}, ${n}]`)
  console.log(`  Time: ${(end2 - start2).toFixed(4)}ms`)
  console.log(`  Elements: ${result.length}`)
  console.log(`  Acceleration: WebGPU 🚀\n`)

  // Example 3: When to use each
  console.log('💡 When to Use Each Acceleration Method:\n')

  console.log('Use WASM (Automatic):')
  console.log('  ✅ Synchronous operations')
  console.log('  ✅ Medium matrices (1K-100K elements)')
  console.log('  ✅ Batch size 32-64')
  console.log('  ✅ No code changes needed')
  console.log('  ✅ Works in Node, Bun, Deno, Browser')

  console.log('\nUse WebGPU (Manual):')
  console.log('  ✅ Very large matrices (>100K elements)')
  console.log('  ✅ Large batch sizes (>128)')
  console.log('  ✅ Multiple parallel operations')
  console.log('  ⚠️  Requires async/await')
  console.log('  ⚠️  Browser only (Chrome, Edge, etc.)')
  console.log('  ⚠️  Explicit GPU API usage needed')

  console.log('\nUse TypeScript (Default):')
  console.log('  ✅ Small matrices (<1K elements)')
  console.log('  ✅ Single-example inference')
  console.log('  ✅ Always available')

  // Example 4: Hybrid approach - WASM for training, WebGPU for inference
  console.log('\n🔄 Example 3: Hybrid Approach\n')
  console.log('Use WASM for training (sync), WebGPU for large batch inference (async):')

  // Training with WASM (synchronous)
  console.log('\nTraining phase (WASM):')
  const trainBatch = T.randn([32, 16]) // Batch of 32 examples
  const weights = T.randn([16, 64])

  const trainStart = performance.now()
  const trainOutput = T.matmul(trainBatch, weights) // Uses WASM automatically
  const trainEnd = performance.now()

  console.log(`  Forward pass: [32, 16] @ [16, 64] = [32, 64]`)
  console.log(`  Time: ${(trainEnd - trainStart).toFixed(4)}ms (WASM ⚡)`)

  // Inference with WebGPU (asynchronous, larger batch)
  console.log('\nInference phase (WebGPU):')
  const infBatchData = new Float32Array(256 * 16) // Batch of 256 examples
  const weightsData = new Float32Array(16 * 64)
  for (let i = 0; i < infBatchData.length; i++) infBatchData[i] = Math.random()
  for (let i = 0; i < weightsData.length; i++) weightsData[i] = Math.random()

  const infStart = performance.now()
  const infOutput = await gpu.matmulGPU(infBatchData, weightsData, 256, 16, 64)
  const infEnd = performance.now()

  console.log(`  Forward pass: [256, 16] @ [16, 64] = [256, 64]`)
  console.log(`  Time: ${(infEnd - infStart).toFixed(4)}ms (WebGPU 🚀)`)

  console.log('\n✨ Demo complete!')
}

// Only run if we're in the right environment
if (typeof navigator !== 'undefined' && 'gpu' in navigator) {
  main().catch(console.error)
} else {
  console.log('⚠️  WebGPU not available in this environment')
  console.log('Please run this demo in a browser with WebGPU support')
}
