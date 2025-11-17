/**
 * XOR training demo using train() function
 */

import { tensor } from '@sylphx/tensor'
import { Sequential, Linear, Tanh } from '@sylphx/nn'
import { Adam } from '@sylphx/optim'
import { mse } from '@sylphx/functional'
import { getParams, train } from '@sylphx/train'

// Data
const x = tensor([[0, 0], [0, 1], [1, 0], [1, 1]], { requiresGrad: true })
const y = tensor([[0], [1], [1], [0]], { requiresGrad: true })

// Model
const model = Sequential(
  Linear(2, 8),
  Tanh(),
  Linear(8, 1)
)

// Init
let modelState = model.init()
const optimizer = Adam({ lr: 0.05 })
let optState = optimizer.init(getParams(modelState))

// Train for 3000 epochs
const result = train({
  model,
  modelState,
  optimizer,
  optState,
  input: x,
  target: y,
  lossFn: mse,
  epochs: 3000,
  onEpoch: (epoch, loss) => {
    if (epoch % 500 === 0) {
      console.log(`Epoch ${epoch}, Loss: ${loss}`)
    }
  }
})

modelState = result.modelState
optState = result.optState

console.log('\nFinal loss:', result.losses[result.losses.length - 1])

// Test
console.log('\nPredictions:')
const testCases = [
  [0, 0],
  [0, 1],
  [1, 0],
  [1, 1]
]

for (const input of testCases) {
  const pred = model.forward(tensor([input]), modelState)
  console.log(`${input} => ${pred.data[0]?.toFixed(4)}`)
}
