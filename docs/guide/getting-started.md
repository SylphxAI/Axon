# Getting Started

## Installation

::: code-group

```bash [bun]
bun add @sylphx/tensor @sylphx/nn @sylphx/functional @sylphx/optim
```

```bash [npm]
npm install @sylphx/tensor @sylphx/nn @sylphx/functional @sylphx/optim
```

```bash [pnpm]
pnpm add @sylphx/tensor @sylphx/nn @sylphx/functional @sylphx/optim
```

:::

## Your First Neural Network

Let's solve the XOR problem - a classic non-linear classification task.

```typescript
import { tensor } from '@sylphx/tensor'
import * as nn from '@sylphx/nn'
import * as F from '@sylphx/functional'
import { adam } from '@sylphx/optim'

// Training data - XOR truth table
const x = tensor([[0,0], [0,1], [1,0], [1,1]], { requiresGrad: true })
const y = tensor([[0], [1], [1], [0]], { requiresGrad: true })

// Initialize model (2 → 8 → 1)
let model = {
  linear1: nn.linear.init(2, 8),
  linear2: nn.linear.init(8, 1),
}

let optimizer = adam.init(model, { lr: 0.01 })

// Training loop
for (let epoch = 0; epoch < 2000; epoch++) {
  // Forward pass
  const h = F.tanh(nn.linear.forward(x, model.linear1))
  const out = F.sigmoid(nn.linear.forward(h, model.linear2))
  
  // Compute loss
  const loss = F.binaryCrossEntropy(out, y)

  // Backward pass + update
  T.backward(loss)
  ;({ model, optimizer } = adam.step(model, optimizer))
  
  if (epoch % 500 === 0) {
    console.log(`Epoch ${epoch}, Loss: ${loss.data[0]}`)
  }
}

// Test predictions
const test = (input: number[]) => {
  const x = tensor([input])
  const h = F.tanh(nn.linear.forward(x, model.linear1))
  const out = F.sigmoid(nn.linear.forward(h, model.linear2))
  return out.data[0]
}

console.log('XOR(0,0):', test([0, 0]))  // → 0.00
console.log('XOR(0,1):', test([0, 1]))  // → 1.00
console.log('XOR(1,0):', test([1, 0]))  // → 1.00
console.log('XOR(1,1):', test([1, 1]))  // → 0.00
```

## Next Steps

- Learn about [Tensors](/guide/tensors)
- Understand [Autograd](/guide/autograd)
- Explore [Neural Networks](/guide/neural-networks)
- Check out [Performance Tips](/guide/performance)
