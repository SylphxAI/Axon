/**
 * Demo: Model save/load functionality
 */

import * as T from '../../packages/tensor/src/index'
import * as nn from '../../packages/nn/src/index'
import * as F from '../../packages/functional/src/index'

console.log('💾 Model Save/Load Demo\n')

// Create a simple model
const model = {
  linear1: nn.linear.init(4, 8),
  linear2: nn.linear.init(8, 2),
}

console.log('Original model created')

// Serialize model
const serialized: nn.SerializedModel = {
  version: '0.1.0',
  timestamp: new Date().toISOString(),
  layers: [
    {
      type: 'linear',
      name: 'linear1',
      params: [
        nn.serializeTensor(model.linear1.weight, 'weight'),
        nn.serializeTensor(model.linear1.bias, 'bias'),
      ],
      config: { inFeatures: 4, outFeatures: 8 },
    },
    {
      type: 'linear',
      name: 'linear2',
      params: [
        nn.serializeTensor(model.linear2.weight, 'weight'),
        nn.serializeTensor(model.linear2.bias, 'bias'),
      ],
      config: { inFeatures: 8, outFeatures: 2 },
    },
  ],
  metadata: {
    description: 'Test model for save/load',
    architecture: '4 -> 8 -> 2',
  },
}

// Save to JSON
const json = nn.saveModel(serialized)
console.log('\n✅ Model serialized to JSON')

// Print summary
console.log('\n' + nn.getModelSummary(serialized))

// Save to file
const modelPath = '/tmp/test-model.json'
await nn.saveModelToFile(serialized, modelPath)
console.log(`\n✅ Model saved to ${modelPath}`)

// Load from file
const loaded = await nn.loadModelFromFile(modelPath)
console.log('\n✅ Model loaded from file')

// Deserialize parameters
const loadedModel = {
  linear1: {
    weight: nn.deserializeTensor(loaded.layers[0]!.params[0]!),
    bias: nn.deserializeTensor(loaded.layers[0]!.params[1]!),
  },
  linear2: {
    weight: nn.deserializeTensor(loaded.layers[1]!.params[0]!),
    bias: nn.deserializeTensor(loaded.layers[1]!.params[1]!),
  },
}

// Test forward pass with both models
const testInput = T.tensor([[1, 0.5, -0.5, 0.2]])

const forward = (x: T.Tensor, m: typeof model): T.Tensor => {
  let h = nn.linear.forward(x, m.linear1)
  h = F.relu(h)
  h = nn.linear.forward(h, m.linear2)
  return F.sigmoid(h)
}

const originalOutput = forward(testInput, model)
const loadedOutput = forward(testInput, loadedModel)

console.log('\n🧪 Verification:')
console.log('Original output:', T.toArray(originalOutput))
console.log('Loaded output:  ', T.toArray(loadedOutput))

// Check if outputs match
const original = T.toArray(originalOutput)[0] as number[]
const loadedArr = T.toArray(loadedOutput)[0] as number[]

const match = original.every((v, i) => Math.abs(v - loadedArr[i]!) < 1e-6)

console.log('\n' + (match ? '✅ Outputs match!' : '❌ Outputs differ!'))

console.log('\n💾 Save/Load functionality working!')
