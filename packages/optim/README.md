# @neuronline/optim

Pure functional optimizers for NeuronLine.

## Features

- ✅ **Pure Functional**: Immutable optimizer state
- ✅ **Multiple Algorithms**: SGD, SGD with momentum, Adam
- ✅ **Type Safe**: Full TypeScript support
- ✅ **Tiny Bundle**: ~19KB with all optimizers

## Installation

```bash
npm install @neuronline/optim @neuronline/tensor
```

## Quick Start

```typescript
import * as T from '@neuronline/tensor'
import { SGD } from '@neuronline/optim'

// Create parameters
const params = [
  T.randn([100, 10], { requiresGrad: true }),
  T.randn([10], { requiresGrad: true })
]

// Initialize optimizer
const optState = SGD.init(params, { lr: 0.01 })

// Training loop
for (let step = 0; step < 1000; step++) {
  // Forward pass + loss computation
  const loss = computeLoss(params)

  // Backward pass
  const grads = T.backward(loss)

  // Update parameters
  const result = SGD.step(optState, params, grads)

  // Use updated params for next iteration
  params = result.params
}
```

## Optimizers

### SGD (Stochastic Gradient Descent)

Basic gradient descent with optional momentum.

```typescript
import { SGD } from '@neuronline/optim'

// Without momentum
const optState = SGD.init(params, { lr: 0.01 })

// With momentum (0.9)
const optState = SGD.init(params, { lr: 0.01, momentum: 0.9 })

// Update step
const result = SGD.step(optState, params, grads)
```

**Hyperparameters:**
- `lr` (learning rate): Step size for parameter updates (default: 0.01)
- `momentum`: Momentum factor for accelerated convergence (default: 0.0)

**Update rule:**
```
v_t = momentum * v_{t-1} + grad
param = param - lr * v_t
```

### Adam (Adaptive Moment Estimation)

Adaptive learning rates with momentum and RMSprop.

```typescript
import { Adam } from '@neuronline/optim'

// Initialize with default hyperparameters
const optState = Adam.init(params, {
  lr: 0.001,
  beta1: 0.9,    // First moment decay
  beta2: 0.999,  // Second moment decay
  eps: 1e-8      // Numerical stability
})

// Update step
const result = Adam.step(optState, params, grads)
```

**Hyperparameters:**
- `lr`: Learning rate (default: 0.001)
- `beta1`: Exponential decay rate for first moment (default: 0.9)
- `beta2`: Exponential decay rate for second moment (default: 0.999)
- `eps`: Small constant for numerical stability (default: 1e-8)

**Update rule:**
```
m_t = beta1 * m_{t-1} + (1 - beta1) * grad
v_t = beta2 * v_{t-1} + (1 - beta2) * grad^2
m_hat = m_t / (1 - beta1^t)
v_hat = v_t / (1 - beta2^t)
param = param - lr * m_hat / (sqrt(v_hat) + eps)
```

## API

All optimizers follow the same interface:

### `init(params, config)`

Initialize optimizer state for a list of parameters.

```typescript
const optState = SGD.init(params, { lr: 0.01 })
```

### `step(state, params, grads)`

Perform single optimization step.

```typescript
const result = SGD.step(optState, params, grads)
// result.state: Updated optimizer state
// result.params: Updated parameters
```

### `zeroGrad(params)`

Zero out gradients (convenience function).

```typescript
import { zeroGrad } from '@neuronline/optim'

zeroGrad(params)
```

## Complete Training Example

```typescript
import * as T from '@neuronline/tensor'
import * as NN from '@neuronline/nn'
import * as F from '@neuronline/functional'
import { Adam } from '@neuronline/optim'

// Create model
const model = {
  fc1: NN.Linear.create(784, 128),
  fc2: NN.Linear.create(128, 10)
}

// Collect all parameters
const params = [
  model.fc1.weight,
  model.fc1.bias,
  model.fc2.weight,
  model.fc2.bias
]

// Initialize Adam optimizer
let optState = Adam.init(params, { lr: 0.001 })

// Training loop
for (let epoch = 0; epoch < 10; epoch++) {
  for (const batch of dataLoader) {
    // Forward pass
    let x = batch.input
    x = NN.Linear.forward(x, model.fc1)
    x = F.relu(x)
    x = NN.Linear.forward(x, model.fc2)

    // Compute loss
    const loss = F.crossEntropy(x, batch.target)

    // Backward pass
    const grads = T.backward(loss)

    // Optimize
    const result = Adam.step(optState, params, grads)
    optState = result.state

    // Update model with new parameters
    ;[model.fc1.weight, model.fc1.bias, model.fc2.weight, model.fc2.bias] = result.params

    console.log(`Loss: ${T.toArray(loss)}`)
  }
}
```

## Learning Rate Scheduling

Manually adjust learning rate:

```typescript
let optState = SGD.init(params, { lr: 0.1 })

for (let epoch = 0; epoch < 100; epoch++) {
  // Decay learning rate every 30 epochs
  if (epoch % 30 === 0) {
    optState = SGD.init(params, { lr: 0.1 * Math.pow(0.1, Math.floor(epoch / 30)) })
  }

  // Training step
  const result = SGD.step(optState, params, grads)
  optState = result.state
  params = result.params
}
```

## Optimizer Comparison

| Optimizer | Best For | Convergence | Memory |
|-----------|----------|-------------|--------|
| SGD | Simple problems, fine-tuning | Slower | Low |
| SGD + Momentum | Most problems | Medium | Medium |
| Adam | Deep networks, sparse gradients | Faster | High |

## Immutable State

All optimizers maintain immutable state:

```typescript
type SGDState = {
  config: { lr: number; momentum: number }
  velocity: T.Tensor[] | null
  step: number
}

type AdamState = {
  config: { lr: number; beta1: number; beta2: number; eps: number }
  m: T.Tensor[]  // First moment
  v: T.Tensor[]  // Second moment
  step: number
}
```

State is updated functionally - each call to `step()` returns new state and parameters without modifying inputs.

## Bundle Size

- SGD: 6KB
- Adam: 12KB
- **Total**: ~19KB (minified)

## Compatibility

- ✅ Node.js 18+
- ✅ Browsers (modern)
- ✅ Bun
- ✅ Deno

## License

MIT

## Related Packages

- `@neuronline/tensor` - Tensor operations and autograd
- `@neuronline/nn` - Neural network layers
- `@neuronline/functional` - Activation and loss functions
