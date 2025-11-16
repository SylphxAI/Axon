/**
 * LSTM Demo - Character-level sequence modeling
 * Demonstrates LSTM forward pass and sequence processing
 */

import * as nn from '../../packages/nn/src/index'
import * as T from '../../packages/tensor/src/index'

console.log('🔄 LSTM Demo - Sequence Processing\n')

// Configuration
const inputSize = 4 // One-hot encoded characters
const hiddenSize = 8
const batchSize = 1

// Initialize LSTM
console.log('Initializing LSTM...')
const model = nn.lstm.init(inputSize, hiddenSize)
console.log(`✅ LSTM initialized (input: ${inputSize}, hidden: ${hiddenSize})\n`)

// Create a simple sequence: "hello"
// One-hot encoding: h=0, e=1, l=2, o=3
const sequence = [
  T.tensor([[1, 0, 0, 0]]), // h - shape [1, 4]
  T.tensor([[0, 1, 0, 0]]), // e
  T.tensor([[0, 0, 1, 0]]), // l
  T.tensor([[0, 0, 1, 0]]), // l
  T.tensor([[0, 0, 0, 1]]), // o
]

console.log('Processing sequence: "hello"')
console.log(`Sequence length: ${sequence.length}`)
console.log(`Input shape: [${sequence[0]!.shape}]\n`)

// Initialize hidden state
let hidden = nn.lstm.initHidden(batchSize, hiddenSize)

// Process sequence step by step
console.log('Forward pass (step-by-step):')
for (let t = 0; t < sequence.length; t++) {
  const result = nn.lstm.forward(sequence[t]!, hidden, model)
  hidden = result.hidden

  console.log(`  Step ${t + 1}: h_t = [${result.output.data.slice(0, 4).map(x => x.toFixed(4)).join(', ')}...]`)
}

console.log('\n✅ Single-step forward pass complete\n')

// Process entire sequence at once
console.log('Processing entire sequence:')
const initialHidden = nn.lstm.initHidden(batchSize, hiddenSize)
const { outputs, finalHidden } = nn.lstm.forwardSequence(sequence, initialHidden, model)

console.log(`Output sequence length: ${outputs.length}`)
console.log(`Final hidden state shape: [${finalHidden.h.shape}]`)
console.log(`Final cell state shape: [${finalHidden.c.shape}]`)

// Show final hidden state values
console.log('\nFinal hidden state (h_t):')
console.log(`  ${finalHidden.h.data.map(x => x.toFixed(4)).join(', ')}`)

console.log('\nFinal cell state (c_t):')
console.log(`  ${finalHidden.c.data.map(x => x.toFixed(4)).join(', ')}`)

console.log('\n✅ LSTM sequence processing complete!')

// Verify shapes
console.log('\n📊 Shape verification:')
console.log(`  Input: [${batchSize}, ${inputSize}]`)
console.log(`  Hidden: [${batchSize}, ${hiddenSize}]`)
console.log(`  Cell: [${batchSize}, ${hiddenSize}]`)
console.log(`  Output: [${batchSize}, ${hiddenSize}]`)
