/**
 * XOR Example - Pure Functional PyTorch-like API
 * Demonstrates the new architecture
 */

import * as T from '../../packages/tensor/src/index'
import * as F from '../../packages/functional/src/index'
import * as nn from '../../packages/nn/src/index'

console.log('🧠 XOR Learning - Pure Functional API\n')

// Data
const xData = T.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], { requiresGrad: false })
const yData = T.tensor([[0], [1], [1], [0]], { requiresGrad: false })

// Model state (immutable)
type ModelState = {
  linear1: nn.LinearState
  linear2: nn.LinearState
}

// Pure function: initialize model
const initModel = (): ModelState => ({
  linear1: nn.linear.init(2, 8),
  linear2: nn.linear.init(8, 1),
})

// Pure function: forward pass
const forward = (x: T.Tensor, model: ModelState): T.Tensor => {
  let h = nn.linear.forward(x, model.linear1)
  h = F.tanh(h)
  h = nn.linear.forward(h, model.linear2)
  return F.sigmoid(h)
}

// Pure function: training step
const trainStep = (
  model: ModelState,
  x: T.Tensor,
  y: T.Tensor,
  lr: number
): { model: ModelState; loss: number } => {
  // Forward
  const pred = forward(x, model)

  // Loss
  const loss = F.binaryCrossEntropy(pred, y)

  // Backward
  const grads = T.backward(loss)

  // Manual SGD update (pure)
  const updateParam = (param: T.Tensor, grad: T.Tensor | undefined): T.Tensor => {
    if (!grad) return param
    return T.sub(param, T.mul(grad, T.scalar(lr)))
  }

  // Update weights (returns new model)
  const newModel: ModelState = {
    linear1: {
      weight: updateParam(model.linear1.weight, grads.get(model.linear1.weight)),
      bias: updateParam(model.linear1.bias, grads.get(model.linear1.bias)),
    },
    linear2: {
      weight: updateParam(model.linear2.weight, grads.get(model.linear2.weight)),
      bias: updateParam(model.linear2.bias, grads.get(model.linear2.bias)),
    },
  }

  return { model: newModel, loss: T.item(loss) }
}

// Train (pure functional loop)
console.log('Training...')
let model = initModel()

for (let epoch = 0; epoch < 2000; epoch++) {
  const result = trainStep(model, xData, yData, 0.5)
  model = result.model

  if (epoch % 200 === 0) {
    console.log(`Epoch ${epoch}: Loss = ${result.loss.toFixed(6)}`)
  }
}

// Test
console.log('\n✅ Final Predictions:')
const predictions = forward(xData, model)
const predArray = T.toArray(predictions) as number[][]

console.log(`  0 XOR 0 = ${predArray[0]![0]!.toFixed(4)} (expected: 0.0000)`)
console.log(`  0 XOR 1 = ${predArray[1]![0]!.toFixed(4)} (expected: 1.0000)`)
console.log(`  1 XOR 0 = ${predArray[2]![0]!.toFixed(4)} (expected: 1.0000)`)
console.log(`  1 XOR 1 = ${predArray[3]![0]!.toFixed(4)} (expected: 0.0000)`)

// Calculate accuracy
let correct = 0
for (let i = 0; i < 4; i++) {
  const pred = predArray[i]![0]! > 0.5 ? 1 : 0
  const actual = T.item(T.tensor([yData.data[i]!]))
  if (pred === actual) correct++
}
console.log(`\n📊 Accuracy: ${(correct / 4 * 100).toFixed(2)}%`)

console.log('\n✨ Pure Functional Architecture:')
console.log('  • No classes, only pure functions')
console.log('  • Immutable state (model never mutated)')
console.log('  • Autograd works!')
console.log('  • PyTorch-like API')
