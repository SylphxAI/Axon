/**
 * XOR Problem - Pure Functional API
 * Demonstrates the pure functional API
 */

import { tensor } from '@sylphx/tensor'
import { Sequential, Linear, Tanh } from '@sylphx/nn'
import { Adam } from '@sylphx/optim'
import { mse } from '@sylphx/functional'
import { getParams, trainStep } from '@sylphx/train'

console.log('🧠 XOR Problem - Pure Functional API\n')

// Training data - XOR truth table
const x = tensor([[0, 0], [0, 1], [1, 0], [1, 1]], { requiresGrad: true })
const y = tensor([[0], [1], [1], [0]], { requiresGrad: true })

// Model - pure function composition
const model = Sequential(
  Linear(2, 8),
  Tanh(),
  Linear(8, 4),
  Tanh(),
  Linear(4, 1),
  Tanh()
)

// Initialize states
console.log('Initializing model and optimizer...')
let modelState = model.init()
const optimizer = Adam({ lr: 0.05 })
let optState = optimizer.init(getParams(modelState))

console.log(`Model has ${getParams(modelState).length} trainable parameters\n`)

// Training loop - pure functional
console.log('Training...')
const epochs = 3000

for (let epoch = 0; epoch < epochs; epoch++) {
  const result = trainStep({
    model,
    modelState,
    optimizer,
    optState,
    input: x,
    target: y,
    lossFn: mse
  })

  modelState = result.modelState
  optState = result.optState

  if (epoch % 500 === 0 || epoch === epochs - 1) {
    console.log(`Epoch ${epoch}, Loss: ${result.loss.toFixed(6)}`)
  }
}

// Test predictions
console.log('\n📊 Test Results:')
console.log('─'.repeat(40))

const testCases = [
  [0, 0],
  [0, 1],
  [1, 0],
  [1, 1]
]

for (const testInput of testCases) {
  const input = tensor([testInput])
  const output = model.forward(input, modelState)
  const pred = output.data[0]!
  const expected = testInput[0]! ^ testInput[1]!  // XOR

  console.log(
    `XOR(${testInput[0]}, ${testInput[1]}) = ${pred.toFixed(4)} ` +
    `(expected: ${expected}, ` +
    `${Math.abs(pred - expected) < 0.1 ? '✅' : '❌'})`
  )
}

console.log('\n✨ Done!')
