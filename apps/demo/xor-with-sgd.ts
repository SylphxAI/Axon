/**
 * XOR example using SGD optimizer
 */

import * as T from '../../packages/tensor/src/index'
import * as F from '../../packages/functional/src/index'
import * as nn from '../../packages/nn/src/index'
import * as optim from '../../packages/optim/src/index'

console.log('🧠 XOR with SGD Optimizer\n')

// Data
const xData = T.tensor(
  [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
  ],
  { requiresGrad: false }
)

const yData = T.tensor([[0], [1], [1], [0]], { requiresGrad: false })

// Model
type ModelState = {
  linear1: nn.LinearState
  linear2: nn.LinearState
}

let model: ModelState = {
  linear1: nn.linear.init(2, 8),
  linear2: nn.linear.init(8, 1),
}

// Collect all parameters
const getParams = (m: ModelState): T.Tensor[] => [
  m.linear1.weight,
  m.linear1.bias,
  m.linear2.weight,
  m.linear2.bias,
]

// Initialize optimizer (SGD with lr=0.1, like the manual version)
let optimizer = optim.sgd.init(getParams(model), { lr: 0.1 })

// Forward pass
const forward = (x: T.Tensor, m: ModelState): T.Tensor => {
  let h = nn.linear.forward(x, m.linear1)
  h = F.tanh(h)
  h = nn.linear.forward(h, m.linear2)
  return F.sigmoid(h)
}

// Training step
const trainStep = (
  m: ModelState,
  opt: optim.OptimizerState,
  x: T.Tensor,
  y: T.Tensor
): { model: ModelState; optimizer: optim.OptimizerState; loss: number } => {
  // Forward + loss
  const pred = forward(x, m)
  const loss = F.binaryCrossEntropy(pred, y)
  const grads = T.backward(loss)

  // Optimizer step
  const result = optim.sgd.step(opt, getParams(m), grads)

  // Rebuild model with new parameters
  const newModel: ModelState = {
    linear1: {
      weight: result.params[0]!,
      bias: result.params[1]!,
    },
    linear2: {
      weight: result.params[2]!,
      bias: result.params[3]!,
    },
  }

  return {
    model: newModel,
    optimizer: result.state,
    loss: T.item(loss),
  }
}

// Train
console.log('Training with SGD (lr=0.1)...\n')
for (let epoch = 0; epoch <= 2000; epoch++) {
  const result = trainStep(model, optimizer, xData, yData)
  model = result.model
  optimizer = result.optimizer

  if (epoch % 200 === 0) {
    console.log(`Epoch ${epoch}: Loss = ${result.loss.toFixed(6)}`)
  }
}

// Test
console.log('\n✅ Final Predictions:')
const finalPred = forward(xData, model)
const preds = T.toArray(finalPred) as number[][]
const targets = T.toArray(yData) as number[][]

for (let i = 0; i < preds.length; i++) {
  const x = T.toArray(xData)[i] as number[]
  const pred = preds[i]![0]!
  const target = targets[i]![0]!
  console.log(
    `  ${x[0]} XOR ${x[1]} = ${pred.toFixed(4)} (expected: ${target.toFixed(4)})`
  )
}

// Accuracy
const correct = preds.reduce((acc, pred, i) => {
  const predicted = pred[0]! > 0.5 ? 1 : 0
  const target = targets[i]![0]!
  return acc + (predicted === target ? 1 : 0)
}, 0)

console.log(`\n📊 Accuracy: ${((correct / preds.length) * 100).toFixed(2)}%`)
console.log('\n✨ Using SGD optimizer with pure functional API!')
