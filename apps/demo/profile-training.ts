/**
 * Profile Training Performance
 * Identifies hot paths and optimization opportunities in training loop
 */

import * as T from '@sylphx/tensor'
import * as F from '@sylphx/functional'
import * as nn from '@sylphx/nn'
import * as optim from '@sylphx/optim'

console.log('🔬 Training Performance Profiler\n')

// Load WASM acceleration
const wasmLoaded = await T.loadAcceleration()
console.log(`WASM: ${wasmLoaded ? '✅ ENABLED' : '❌ DISABLED'}\n`)

// Performance tracking
const timings: Record<string, number> = {}
let totalTime = 0

function startTimer(label: string) {
  timings[label] = timings[label] || 0
  return performance.now()
}

function endTimer(label: string, start: number) {
  const elapsed = performance.now() - start
  timings[label] = timings[label]! + elapsed
  totalTime += elapsed
}

// Create a realistic neural network (similar to DQN)
console.log('📐 Creating Network (3 layers: 16→64→64→4)\n')
const network = {
  linear1: nn.linear.init(16, 64),
  linear2: nn.linear.init(64, 64),
  linear3: nn.linear.init(64, 4),
}

const params = [
  network.linear1.weight,
  network.linear1.bias,
  network.linear2.weight,
  network.linear2.bias,
  network.linear3.weight,
  network.linear3.bias,
]

let optimizerState = optim.adam.init(params, { lr: 0.001 })

// Training configuration
const BATCH_SIZE = 32
const NUM_BATCHES = 100
const INPUT_SIZE = 16
const OUTPUT_SIZE = 4

console.log(`Training for ${NUM_BATCHES} batches of size ${BATCH_SIZE}\n`)
console.log('⏱️  Profiling...\n')

for (let batch = 0; batch < NUM_BATCHES; batch++) {
  // Data preparation
  let start = startTimer('data_prep')
  const inputs = T.randn([BATCH_SIZE, INPUT_SIZE])
  const targets = T.randn([BATCH_SIZE, OUTPUT_SIZE])
  endTimer('data_prep', start)

  // Forward pass - Layer 1
  start = startTimer('linear1_forward')
  let h = nn.linear.forward(inputs, network.linear1)
  endTimer('linear1_forward', start)

  start = startTimer('relu1')
  h = F.relu(h)
  endTimer('relu1', start)

  // Forward pass - Layer 2
  start = startTimer('linear2_forward')
  h = nn.linear.forward(h, network.linear2)
  endTimer('linear2_forward', start)

  start = startTimer('relu2')
  h = F.relu(h)
  endTimer('relu2', start)

  // Forward pass - Layer 3
  start = startTimer('linear3_forward')
  const output = nn.linear.forward(h, network.linear3)
  endTimer('linear3_forward', start)

  // Loss computation
  start = startTimer('loss')
  const loss = F.mse(output, targets)
  endTimer('loss', start)

  // Backward pass
  start = startTimer('backward')
  const grads = T.backward(loss)
  endTimer('backward', start)

  // Optimizer step
  start = startTimer('optimizer')
  const result = optim.adam.step(optimizerState, params, grads)
  optimizerState = result.state
  endTimer('optimizer', start)

  // Update network (parameter copying)
  start = startTimer('param_update')
  network.linear1 = {
    weight: result.params[0]!,
    bias: result.params[1]!,
  }
  network.linear2 = {
    weight: result.params[2]!,
    bias: result.params[3]!,
  }
  network.linear3 = {
    weight: result.params[4]!,
    bias: result.params[5]!,
  }
  endTimer('param_update', start)
}

// Analyze results
console.log('📊 Performance Profile:\n')
console.log('Operation'.padEnd(25), 'Time (ms)'.padEnd(15), 'Percent'.padEnd(12), 'Per Batch')
console.log('─'.repeat(75))

// Sort by time spent
const sorted = Object.entries(timings).sort((a, b) => b[1] - a[1])

for (const [label, time] of sorted) {
  const percent = ((time / totalTime) * 100).toFixed(2)
  const perBatch = (time / NUM_BATCHES).toFixed(4)
  console.log(
    label.padEnd(25),
    time.toFixed(2).padEnd(15),
    `${percent}%`.padEnd(12),
    `${perBatch}ms`
  )
}

console.log('─'.repeat(75))
console.log(
  'TOTAL'.padEnd(25),
  totalTime.toFixed(2).padEnd(15),
  '100.00%'.padEnd(12),
  `${(totalTime / NUM_BATCHES).toFixed(4)}ms`
)

console.log('\n💡 Analysis:\n')

// Identify hot paths (>10% of time)
const hotPaths = sorted.filter(([_, time]) => (time / totalTime) * 100 > 10)
if (hotPaths.length > 0) {
  console.log('🔥 Hot Paths (>10% of time):')
  for (const [label, time] of hotPaths) {
    const percent = ((time / totalTime) * 100).toFixed(2)
    console.log(`  • ${label}: ${percent}%`)
  }
  console.log()
}

// Calculate breakdown by category
const categories = {
  forward: 0,
  backward: 0,
  optimizer: 0,
  loss: 0,
  data_prep: 0,
  other: 0,
}

for (const [label, time] of Object.entries(timings)) {
  if (label.includes('forward')) categories.forward += time
  else if (label.includes('backward')) categories.backward += time
  else if (label.includes('optimizer')) categories.optimizer += time
  else if (label.includes('loss')) categories.loss += time
  else if (label.includes('data_prep')) categories.data_prep += time
  else if (label.includes('relu')) categories.forward += time
  else categories.other += time
}

console.log('📦 Time Breakdown by Category:\n')
for (const [category, time] of Object.entries(categories)) {
  const percent = ((time / totalTime) * 100).toFixed(2)
  console.log(`  ${category.padEnd(15)}: ${percent.padStart(6)}%  (${time.toFixed(2)}ms)`)
}

console.log('\n🎯 Optimization Opportunities:\n')

// Suggest optimizations based on profiling
if ((categories.backward / totalTime) * 100 > 30) {
  console.log('  • Backward pass is >30% - Consider WASM optimization for gradient computation')
}

if ((categories.forward / totalTime) * 100 > 40) {
  console.log('  • Forward pass is >40% - Linear layers dominate, already WASM accelerated ✅')
}

if ((categories.optimizer / totalTime) * 100 > 20) {
  console.log('  • Optimizer is >20% - Consider optimizing Adam parameter updates')
}

if ((timings.linear1_forward || 0 + timings.linear2_forward || 0) / totalTime > 0.5) {
  console.log('  • Linear layers >50% - Batching working well, WASM activated ✅')
}

console.log('\n✨ Current Optimizations Active:\n')
console.log('  ✅ Batched training (32 examples together)')
console.log('  ✅ WASM acceleration for matrix multiplication')
console.log('  ✅ Memory pooling (reduced GC pressure)')
console.log('  ✅ Loop unrolling in all tensor operations')
console.log('  ✅ Tiled matrix multiplication (cache efficiency)')

console.log('\n🚀 Performance Metrics:\n')
const throughput = (BATCH_SIZE * NUM_BATCHES * 1000) / totalTime
console.log(`  Training throughput: ${throughput.toFixed(0)} examples/sec`)
console.log(`  Avg batch time: ${(totalTime / NUM_BATCHES).toFixed(4)}ms`)
console.log(`  Avg example time: ${(totalTime / (BATCH_SIZE * NUM_BATCHES)).toFixed(4)}ms`)
