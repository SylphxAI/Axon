# @neuronline/functional

Pure functional activation and loss functions for NeuronLine.

## Features

- ✅ **Pure Functional**: No side effects, immutable tensors
- ✅ **Autograd Support**: All functions differentiable
- ✅ **Complete Set**: Activations, losses, and utilities
- ✅ **Type Safe**: Full TypeScript support
- ✅ **Tiny Bundle**: ~14KB with all functions

## Installation

```bash
npm install @neuronline/functional @neuronline/tensor
```

## Quick Start

```typescript
import * as T from '@neuronline/tensor'
import * as F from '@neuronline/functional'

// Activations
const x = T.tensor([-2, -1, 0, 1, 2])
const activated = F.relu(x)        // [0, 0, 0, 1, 2]
const sigmoid = F.sigmoid(x)       // [0.12, 0.27, 0.5, 0.73, 0.88]
const tanh = F.tanh(x)             // [-0.96, -0.76, 0, 0.76, 0.96]

// Loss functions
const logits = T.randn([10, 5])
const target = T.randint([10], 0, 5)
const loss = F.crossEntropy(logits, target)
```

## Activation Functions

### ReLU (Rectified Linear Unit)

```typescript
const y = F.relu(x)  // max(0, x)
```

Most common activation for hidden layers. Zero for negative inputs.

### Leaky ReLU

```typescript
const y = F.leakyRelu(x, 0.01)  // max(0.01 * x, x)
```

Variant of ReLU that allows small gradient for negative inputs.

### Sigmoid

```typescript
const y = F.sigmoid(x)  // 1 / (1 + exp(-x))
```

Squashes inputs to (0, 1) range. Used for binary classification.

### Tanh (Hyperbolic Tangent)

```typescript
const y = F.tanh(x)  // (exp(x) - exp(-x)) / (exp(x) + exp(-x))
```

Squashes inputs to (-1, 1) range. Zero-centered alternative to sigmoid.

### Softmax

```typescript
const probs = F.softmax(logits)  // exp(x_i) / sum(exp(x))
```

Converts logits to probability distribution. Used for multi-class classification.

## Loss Functions

### Cross Entropy Loss

```typescript
const loss = F.crossEntropy(logits, target)
```

Classification loss combining softmax and negative log-likelihood.

**Arguments:**
- `logits`: Raw scores from model `[batchSize, numClasses]`
- `target`: Class indices `[batchSize]` with values in `[0, numClasses)`

**Returns:** Scalar loss averaged over batch

**Example:**
```typescript
const logits = T.tensor([[2.0, 1.0, 0.1], [0.5, 2.5, 0.2]])
const target = T.tensor([0, 1])  // First sample is class 0, second is class 1
const loss = F.crossEntropy(logits, target)
```

### Mean Squared Error (MSE)

```typescript
const loss = F.mse(predictions, targets)
```

Regression loss measuring squared difference.

**Arguments:**
- `predictions`: Model outputs `[batchSize, ...]`
- `targets`: Ground truth `[batchSize, ...]`

**Returns:** Scalar loss

**Example:**
```typescript
const pred = T.tensor([1.5, 2.3, 3.1])
const target = T.tensor([1.0, 2.0, 3.0])
const loss = F.mse(pred, target)  // mean((pred - target)^2)
```

## Complete Training Example

```typescript
import * as T from '@neuronline/tensor'
import * as NN from '@neuronline/nn'
import * as F from '@neuronline/functional'
import { SGD } from '@neuronline/optim'

// Model
const model = {
  fc1: NN.Linear.create(784, 128),
  fc2: NN.Linear.create(128, 10)
}

const params = [model.fc1.weight, model.fc1.bias, model.fc2.weight, model.fc2.bias]
let optState = SGD.init(params, { lr: 0.01 })

// Training loop
for (let epoch = 0; epoch < 10; epoch++) {
  // Forward pass
  let x = T.randn([32, 784], { requiresGrad: true })

  // Hidden layer with ReLU
  x = NN.Linear.forward(x, model.fc1)
  x = F.relu(x)

  // Output layer
  x = NN.Linear.forward(x, model.fc2)

  // Softmax for probabilities (optional, crossEntropy applies it internally)
  const probs = F.softmax(x)

  // Loss
  const target = T.randint([32], 0, 10)
  const loss = F.crossEntropy(x, target)

  // Backward
  const grads = T.backward(loss)

  // Optimize
  const result = SGD.step(optState, params, grads)
  optState = result.state

  console.log(`Epoch ${epoch}, Loss: ${T.toArray(loss)[0]}`)
}
```

## Autograd Support

All functions support automatic differentiation:

```typescript
// ReLU gradient
const x = T.tensor([-1, 0, 1], { requiresGrad: true })
const y = F.relu(x)
const loss = T.sum(y)
const grads = T.backward(loss)
console.log(grads.get(x))  // [0, 0, 1] - gradient is 0 for x < 0, 1 for x > 0

// Cross entropy gradient
const logits = T.tensor([[2.0, 1.0]], { requiresGrad: true })
const target = T.tensor([0])
const loss = F.crossEntropy(logits, target)
const grads = T.backward(loss)
console.log(grads.get(logits))  // Gradient w.r.t. logits
```

## Function Reference

### Activations

| Function | Formula | Range | Use Case |
|----------|---------|-------|----------|
| `relu(x)` | max(0, x) | [0, ∞) | Hidden layers (default) |
| `leakyRelu(x, alpha)` | max(alpha*x, x) | (-∞, ∞) | Hidden layers (dying ReLU fix) |
| `sigmoid(x)` | 1/(1+e^-x) | (0, 1) | Binary classification |
| `tanh(x)` | (e^x - e^-x)/(e^x + e^-x) | (-1, 1) | Hidden layers (zero-centered) |
| `softmax(x)` | e^xi / Σe^xj | (0, 1), Σ=1 | Multi-class classification |

### Losses

| Function | Use Case | Output Type |
|----------|----------|-------------|
| `crossEntropy(logits, target)` | Classification | Scalar |
| `mse(pred, target)` | Regression | Scalar |

## Implementation Details

All functions are implemented using primitive tensor operations from `@neuronline/tensor`:

```typescript
// ReLU implementation
export function relu(t: T.Tensor): T.Tensor {
  return T.map(t, (x) => Math.max(0, x))
}

// Sigmoid implementation
export function sigmoid(t: T.Tensor): T.Tensor {
  return T.map(t, (x) => 1 / (1 + Math.exp(-x)))
}
```

Gradients are automatically computed by the autograd engine in `@neuronline/tensor`.

## Bundle Size

- Activations: 8KB
- Losses: 6KB
- **Total**: ~14KB (minified)

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
- `@neuronline/optim` - Optimizers (SGD, Adam, etc.)
