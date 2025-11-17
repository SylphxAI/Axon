#!/usr/bin/env bun
/**
 * Test new pure functional API
 */

import { tensor } from './packages/tensor/src/index'
import { Sequential, Linear, Tanh } from './packages/nn/src/index'
import { Adam } from './packages/optim/src/index'
import { mse } from './packages/functional/src/index'
import { getParams, trainStep } from './packages/train/src/index'

console.log('🧠 Testing Pure Functional API\n')

// XOR data
const x = tensor([[0, 0], [0, 1], [1, 0], [1, 1]], { requiresGrad: true })
const y = tensor([[0], [1], [1], [0]], { requiresGrad: true })

// Model
const model = Sequential(
  Linear(2, 4),
  Tanh(),
  Linear(4, 1),
  Tanh()
)

// Init
let modelState = model.init()
const optimizer = Adam({ lr: 0.1 })
let optState = optimizer.init(getParams(modelState))

console.log(`Parameters: ${getParams(modelState).length}`)

// Train
console.log('Training...')
for (let epoch = 0; epoch < 1000; epoch++) {
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

  if (epoch % 200 === 0) {
    console.log(`Epoch ${epoch}, Loss: ${result.loss.toFixed(6)}`)
  }
}

// Test
console.log('\n📊 Results:')
for (const [i, j] of [[0, 0], [0, 1], [1, 0], [1, 1]]) {
  const input = tensor([[i, j]])
  const output = model.forward(input, modelState)
  const pred = output.data[0]!
  const expected = i ^ j
  console.log(`XOR(${i}, ${j}) = ${pred.toFixed(4)} (expected: ${expected})`)
}

console.log('\n✨ Pure functional API works!')
