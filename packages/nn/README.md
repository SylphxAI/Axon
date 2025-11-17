# @neuronline/nn

Pure functional neural network layers for NeuronLine.

## Features

- ✅ **Pure Functional**: Immutable state, explicit transformations
- ✅ **Modular Layers**: Linear, Conv2D, BatchNorm, Dropout, Pooling
- ✅ **Sequential API**: Compose layers with simple function chaining
- ✅ **Type Safe**: Full TypeScript support with strict types
- ✅ **Tiny Bundle**: ~31KB with all layers

## Installation

```bash
npm install @neuronline/nn @neuronline/tensor @neuronline/functional
```

## Quick Start

```typescript
import * as T from '@neuronline/tensor'
import * as NN from '@neuronline/nn'
import * as F from '@neuronline/functional'

// Define network architecture
const layers = [
  NN.Linear.create(784, 128),
  NN.Linear.create(128, 10)
]

// Forward pass
let x = T.randn([32, 784]) // batch of 32 images
for (const layer of layers) {
  x = NN.Linear.forward(x, layer)
  x = F.relu(x)
}

// Output: [32, 10] logits
```

## Available Layers

### Linear (Fully Connected)

Dense layer with weight matrix and bias vector.

```typescript
import { Linear } from '@neuronline/nn'

// Create layer
const layer = Linear.create(inputDim, outputDim)

// Forward pass
const output = Linear.forward(input, layer) // output = input @ W^T + b

// Backward pass (autograd)
const grads = T.backward(loss)
```

### Conv2D

2D convolutional layer with configurable kernel, stride, and padding.

```typescript
import { Conv2D } from '@neuronline/nn'

// Create 3x3 conv, 3 in channels, 16 out channels
const layer = Conv2D.create(3, 16, 3)

// Forward pass: [B, H, W, C] -> [B, H', W', C']
const output = Conv2D.forward(input, layer)
```

### MaxPool2D / AvgPool2D

Spatial downsampling with max or average pooling.

```typescript
import { MaxPool2D, AvgPool2D } from '@neuronline/nn'

// 2x2 pooling
const pooled = MaxPool2D.forward(input, { kernelSize: 2 })
const avgPooled = AvgPool2D.forward(input, { kernelSize: 2 })
```

### BatchNorm

Batch normalization for stable training.

```typescript
import { BatchNorm } from '@neuronline/nn'

// Create layer
const layer = BatchNorm.create(numFeatures)

// Forward pass
const normalized = BatchNorm.forward(input, layer)
```

### Dropout

Regularization by randomly dropping units during training.

```typescript
import { Dropout } from '@neuronline/nn'

// 50% dropout
const dropped = Dropout.forward(input, { dropRate: 0.5, training: true })

// Inference (no dropout)
const output = Dropout.forward(input, { dropRate: 0.5, training: false })
```

## Sequential API

Build networks by composing layers:

```typescript
import * as NN from '@neuronline/nn'
import * as F from '@neuronline/functional'

function forward(x: T.Tensor, model: Model): T.Tensor {
  // Conv block
  x = NN.Conv2D.forward(x, model.conv1)
  x = F.relu(x)
  x = NN.MaxPool2D.forward(x, { kernelSize: 2 })

  // Flatten
  const batchSize = x.shape[0]!
  x = T.reshape(x, [batchSize, -1])

  // Classifier
  x = NN.Linear.forward(x, model.fc1)
  x = F.relu(x)
  x = NN.Linear.forward(x, model.fc2)

  return x
}
```

## Training Example

```typescript
import * as T from '@neuronline/tensor'
import * as NN from '@neuronline/nn'
import * as F from '@neuronline/functional'
import { SGD } from '@neuronline/optim'

// Create model
const model = {
  fc1: NN.Linear.create(784, 128),
  fc2: NN.Linear.create(128, 10)
}

// Optimizer
const params = [model.fc1.weight, model.fc1.bias, model.fc2.weight, model.fc2.bias]
const optState = SGD.init(params, { lr: 0.01 })

// Training loop
for (let epoch = 0; epoch < 10; epoch++) {
  // Forward
  let x = T.randn([32, 784], { requiresGrad: true })
  x = NN.Linear.forward(x, model.fc1)
  x = F.relu(x)
  x = NN.Linear.forward(x, model.fc2)

  const target = T.randint([32], 0, 10)
  const loss = F.crossEntropy(x, target)

  // Backward
  const grads = T.backward(loss)

  // Update
  const updated = SGD.step(optState, params, grads)

  console.log(`Epoch ${epoch}, Loss: ${T.toArray(loss)}`)
}
```

## Layer State

All layers maintain immutable state:

```typescript
type LinearState = {
  weight: T.Tensor  // [outDim, inDim]
  bias: T.Tensor    // [outDim]
}

type Conv2DState = {
  weight: T.Tensor  // [outChannels, kernelH, kernelW, inChannels]
  bias: T.Tensor    // [outChannels]
}
```

Parameters are regular tensors that can be optimized using any optimizer from `@neuronline/optim`.

## Bundle Size

- Linear: 3KB
- Conv2D: 8KB
- BatchNorm: 4KB
- Dropout: 2KB
- Pooling: 4KB
- **Total**: ~31KB (minified)

## Compatibility

- ✅ Node.js 18+
- ✅ Browsers (modern)
- ✅ Bun
- ✅ Deno

## License

MIT

## Related Packages

- `@neuronline/tensor` - Tensor operations and autograd
- `@neuronline/optim` - Optimizers (SGD, Adam, etc.)
- `@neuronline/functional` - Activation and loss functions
- `@neuronline/wasm` - WebAssembly acceleration (optional)
