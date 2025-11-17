---
layout: home

hero:
  name: "Axon"
  text: "Pure Functional Neural Networks"
  tagline: Fast • Functional • Universal
  actions:
    - theme: brand
      text: Get Started
      link: /guide/getting-started
    - theme: alt
      text: View on GitHub
      link: https://github.com/SylphxAI/Axon

features:
  - icon: 🔥
    title: Pure Functional
    details: Immutable tensors and operations. No side effects, predictable behavior.
  
  - icon: 🚀
    title: Blazingly Fast
    details: 155x faster than baseline with WASM acceleration, memory pooling, and batched training.
  
  - icon: 📦
    title: Modular
    details: Import only what you need. Tree-shakeable modules. ~20KB gzipped.
  
  - icon: 🎯
    title: PyTorch-like API
    details: Familiar to ML practitioners with clear separation of layers, ops, and training.
  
  - icon: 🧠
    title: Complete Features
    details: Linear, Conv2D, LSTM, GRU, BatchNorm, Dropout. SGD, Adam, RMSprop, AdaGrad.
  
  - icon: 🌐
    title: Universal
    details: Works everywhere - Browser, Node.js, Deno, Bun, Edge runtimes.
---

## Quick Example

```typescript
import { tensor } from '@sylphx/tensor'
import * as nn from '@sylphx/nn'
import * as F from '@sylphx/functional'
import { adam } from '@sylphx/optim'

// Training data
const x = tensor([[0,0], [0,1], [1,0], [1,1]], { requiresGrad: true })
const y = tensor([[0], [1], [1], [0]], { requiresGrad: true })

// Initialize model
let model = {
  linear1: nn.linear.init(2, 8),
  linear2: nn.linear.init(8, 1),
}

let optimizer = adam.init(model, { lr: 0.01 })

// Training loop
for (let epoch = 0; epoch < 2000; epoch++) {
  const h = F.tanh(nn.linear.forward(x, model.linear1))
  const out = F.sigmoid(nn.linear.forward(h, model.linear2))
  const loss = F.binaryCrossEntropy(out, y)

  T.backward(loss)
  ;({ model, optimizer } = adam.step(model, optimizer))
}
```

## Performance

- **155x faster** than baseline (3.35 → 530 episodes/sec)
- **99.7% reduction** in training time per step
- **30x less memory** usage
- **29,819 examples/sec** throughput
