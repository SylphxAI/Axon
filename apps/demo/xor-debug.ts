import { NeuralNetwork } from '@sylphx/neuronline-core'

console.log('🔍 Debugging XOR Learning\n')

const nn = new NeuralNetwork({
  layers: [2, 8, 1],
  activation: 'tanh',
  outputActivation: 'sigmoid',
  loss: 'binary-crossentropy',
  optimizer: 'adam',
  learningRate: 0.3,
})

const data = [
  { input: [0, 0], output: [0] },
  { input: [0, 1], output: [1] },
  { input: [1, 0], output: [1] },
  { input: [1, 1], output: [0] },
]

console.log('Training XOR...')
const metrics = nn.train(data, {
  epochs: 2000,
  verbose: true,
})

console.log('\n✅ Final Predictions:')
console.log(`  0 XOR 0 = ${nn.run([0, 0])[0]!.toFixed(6)} (expected: 0.0)`)
console.log(`  0 XOR 1 = ${nn.run([0, 1])[0]!.toFixed(6)} (expected: 1.0)`)
console.log(`  1 XOR 0 = ${nn.run([1, 0])[0]!.toFixed(6)} (expected: 1.0)`)
console.log(`  1 XOR 1 = ${nn.run([1, 1])[0]!.toFixed(6)} (expected: 0.0)`)

console.log(`\n📊 Loss: ${metrics[0]!.loss.toFixed(6)} → ${metrics[metrics.length - 1]!.loss.toFixed(6)}`)
console.log(`📊 Accuracy: ${((metrics[metrics.length - 1]!.accuracy ?? 0) * 100).toFixed(2)}%`)
