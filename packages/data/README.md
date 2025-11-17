# @neuronline/data

Data loaders and utilities for NeuronLine.

## Features

- ✅ **Pure Functional**: Immutable data transformations
- ✅ **Memory Efficient**: Streaming and batching support
- ✅ **Type Safe**: Full TypeScript support
- ✅ **Flexible**: Custom datasets and transformations
- ✅ **Tiny Bundle**: ~5KB

## Installation

```bash
npm install @neuronline/data @neuronline/tensor
```

## Quick Start

```typescript
import { DataLoader, ArrayDataset } from '@neuronline/data'
import * as T from '@neuronline/tensor'

// Create dataset from arrays
const inputs = Array.from({ length: 1000 }, () => Math.random())
const labels = Array.from({ length: 1000 }, () => Math.floor(Math.random() * 10))

const dataset = new ArrayDataset(inputs, labels)

// Create data loader with batching
const loader = new DataLoader(dataset, {
  batchSize: 32,
  shuffle: true
})

// Iterate over batches
for (const batch of loader) {
  const x = T.tensor(batch.input)
  const y = T.tensor(batch.label)

  // Train on batch...
}
```

## DataLoader

Creates batches from a dataset with shuffling support.

```typescript
const loader = new DataLoader(dataset, {
  batchSize: 32,     // Samples per batch
  shuffle: true,     // Shuffle data each epoch
  dropLast: false    // Drop incomplete final batch
})
```

### Iteration

```typescript
// For-of loop
for (const batch of loader) {
  console.log(batch.input.length)  // batchSize
}

// Manual iteration
const iterator = loader[Symbol.iterator]()
const firstBatch = iterator.next().value
```

### Properties

```typescript
loader.length       // Number of batches per epoch
loader.datasetSize  // Total number of samples
```

## Datasets

### ArrayDataset

In-memory dataset from JavaScript arrays.

```typescript
import { ArrayDataset } from '@neuronline/data'

const inputs = [[1, 2], [3, 4], [5, 6]]
const labels = [0, 1, 0]

const dataset = new ArrayDataset(inputs, labels)

console.log(dataset.length)     // 3
console.log(dataset.get(0))     // { input: [1, 2], label: 0 }
```

### Custom Datasets

Implement the `Dataset` interface:

```typescript
interface Dataset<T> {
  length: number
  get(index: number): T
}
```

**Example:**

```typescript
class ImageDataset implements Dataset<{ image: number[][], label: number }> {
  constructor(private imagePaths: string[], private labels: number[]) {}

  get length(): number {
    return this.imagePaths.length
  }

  get(index: number) {
    const image = loadImage(this.imagePaths[index]!)  // Your loading logic
    const label = this.labels[index]!
    return { image, label }
  }
}

const dataset = new ImageDataset(paths, labels)
const loader = new DataLoader(dataset, { batchSize: 16 })
```

## Complete Training Example

```typescript
import * as T from '@neuronline/tensor'
import * as NN from '@neuronline/nn'
import * as F from '@neuronline/functional'
import { SGD } from '@neuronline/optim'
import { DataLoader, ArrayDataset } from '@neuronline/data'

// Prepare data
const trainInputs = Array.from({ length: 10000 }, () =>
  Array.from({ length: 784 }, () => Math.random())
)
const trainLabels = Array.from({ length: 10000 }, () =>
  Math.floor(Math.random() * 10)
)

const dataset = new ArrayDataset(trainInputs, trainLabels)
const loader = new DataLoader(dataset, {
  batchSize: 32,
  shuffle: true
})

// Model
const model = {
  fc1: NN.Linear.create(784, 128),
  fc2: NN.Linear.create(128, 10)
}

const params = [model.fc1.weight, model.fc1.bias, model.fc2.weight, model.fc2.bias]
let optState = SGD.init(params, { lr: 0.01 })

// Training loop
for (let epoch = 0; epoch < 10; epoch++) {
  let epochLoss = 0
  let numBatches = 0

  for (const batch of loader) {
    // Convert to tensors
    let x = T.tensor(batch.input, { requiresGrad: true })
    const y = T.tensor(batch.label)

    // Forward
    x = NN.Linear.forward(x, model.fc1)
    x = F.relu(x)
    x = NN.Linear.forward(x, model.fc2)

    // Loss
    const loss = F.crossEntropy(x, y)
    epochLoss += T.toArray(loss)[0]!
    numBatches++

    // Backward
    const grads = T.backward(loss)

    // Optimize
    const result = SGD.step(optState, params, grads)
    optState = result.state
    ;[model.fc1.weight, model.fc1.bias, model.fc2.weight, model.fc2.bias] = result.params
  }

  console.log(`Epoch ${epoch}, Loss: ${epochLoss / numBatches}`)
}
```

## Data Transformations

Apply transformations in custom datasets:

```typescript
class TransformedDataset implements Dataset<{ input: number[], label: number }> {
  constructor(
    private baseDataset: Dataset<{ input: number[], label: number }>,
    private transform: (x: number[]) => number[]
  ) {}

  get length() {
    return this.baseDataset.length
  }

  get(index: number) {
    const sample = this.baseDataset.get(index)
    return {
      input: this.transform(sample.input),
      label: sample.label
    }
  }
}

// Normalize inputs to [0, 1]
const normalized = new TransformedDataset(dataset, (x) =>
  x.map(v => v / 255.0)
)
```

## Shuffling

DataLoader shuffles indices, not data:

```typescript
// Shuffle each epoch
const loader = new DataLoader(dataset, {
  batchSize: 32,
  shuffle: true
})

for (let epoch = 0; epoch < 10; epoch++) {
  // New shuffle order each epoch
  for (const batch of loader) {
    // ...
  }
}
```

## Memory Efficiency

DataLoader loads batches on-demand:

```typescript
// Only loads 32 samples at a time, not entire dataset
const loader = new DataLoader(largeDataset, { batchSize: 32 })

for (const batch of loader) {
  // Process batch and release memory
  T.withScope(() => {
    const x = T.tensor(batch.input)
    // ...
  })
}
```

## API Reference

### DataLoader

```typescript
class DataLoader<T> {
  constructor(dataset: Dataset<T>, options: {
    batchSize: number
    shuffle?: boolean
    dropLast?: boolean
  })

  [Symbol.iterator](): Iterator<Batch<T>>

  readonly length: number
  readonly datasetSize: number
}
```

### ArrayDataset

```typescript
class ArrayDataset<I, L> implements Dataset<{ input: I, label: L }> {
  constructor(inputs: I[], labels: L[])

  get(index: number): { input: I, label: L }

  readonly length: number
}
```

### Dataset Interface

```typescript
interface Dataset<T> {
  length: number
  get(index: number): T
}
```

## Bundle Size

- DataLoader: 3KB
- ArrayDataset: 2KB
- **Total**: ~5KB (minified)

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
- `@neuronline/functional` - Activation and loss functions
