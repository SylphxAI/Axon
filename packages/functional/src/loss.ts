/**
 * Pure functional loss functions with autograd support
 */

import type { Tensor } from '@neuronline/tensor'
import { mean, sub, mul } from '@neuronline/tensor'

/**
 * Mean Squared Error loss
 * Pure function with autograd
 */
export function mse(predicted: Tensor, target: Tensor): Tensor {
  const diff = sub(predicted, target)
  const squared = mul(diff, diff)
  return mean(squared)
}

/**
 * Binary Cross-Entropy loss
 * Pure function with autograd
 */
export function binaryCrossEntropy(predicted: Tensor, target: Tensor): Tensor {
  const epsilon = 1e-15

  // Clip predictions to avoid log(0)
  const clippedData = new Float32Array(predicted.data.length)
  for (let i = 0; i < predicted.data.length; i++) {
    clippedData[i] = Math.max(epsilon, Math.min(1 - epsilon, predicted.data[i]!))
  }

  const clipped: Tensor = {
    ...predicted,
    data: clippedData,
  }

  // -[y * log(p) + (1 - y) * log(1 - p)]
  const lossData = new Float32Array(predicted.data.length)
  for (let i = 0; i < predicted.data.length; i++) {
    const p = clipped.data[i]!
    const y = target.data[i]!
    lossData[i] = -(y * Math.log(p) + (1 - y) * Math.log(1 - p))
  }

  const gradFn = predicted.requiresGrad
    ? {
        name: 'binaryCrossEntropy',
        inputs: [predicted, target],
        backward: (grad: Tensor) => {
          // dL/dp = -y/p + (1-y)/(1-p)
          const predGrad = new Float32Array(predicted.data.length)
          const gradValue = grad.data[0]! / predicted.data.length // Mean scaling

          for (let i = 0; i < predicted.data.length; i++) {
            const p = clipped.data[i]!
            const y = target.data[i]!
            predGrad[i] = gradValue * (-y / p + (1 - y) / (1 - p))
          }

          return [
            { ...predicted, data: predGrad, requiresGrad: false },
            { ...target, data: new Float32Array(target.data.length), requiresGrad: false },
          ]
        },
      }
    : undefined

  const loss: Tensor = {
    data: lossData,
    shape: predicted.shape,
    requiresGrad: predicted.requiresGrad,
    gradFn,
  }

  return mean(loss)
}

/**
 * Categorical Cross-Entropy loss (with softmax)
 * Pure function with autograd
 */
export function crossEntropy(predicted: Tensor, target: Tensor): Tensor {
  if (predicted.shape.length !== 2 || target.shape.length !== 2) {
    throw new Error('crossEntropy requires 2D tensors')
  }

  const epsilon = 1e-15
  const [batchSize, numClasses] = predicted.shape

  // Clip predictions
  const clippedData = new Float32Array(predicted.data.length)
  for (let i = 0; i < predicted.data.length; i++) {
    clippedData[i] = Math.max(epsilon, Math.min(1 - epsilon, predicted.data[i]!))
  }

  // Compute loss: -sum(y * log(p))
  const lossData = new Float32Array(batchSize!)
  for (let i = 0; i < batchSize!; i++) {
    let loss = 0
    for (let j = 0; j < numClasses!; j++) {
      const idx = i * numClasses! + j
      const p = clippedData[idx]!
      const y = target.data[idx]!
      loss -= y * Math.log(p)
    }
    lossData[i] = loss
  }

  const gradFn = predicted.requiresGrad
    ? {
        name: 'crossEntropy',
        inputs: [predicted, target],
        backward: (grad: Tensor) => {
          // dL/dp = -y/p
          const predGrad = new Float32Array(predicted.data.length)
          const gradValue = grad.data[0]! / batchSize! // Mean scaling

          for (let i = 0; i < predicted.data.length; i++) {
            const p = clippedData[i]!
            const y = target.data[i]!
            predGrad[i] = gradValue * (-y / p)
          }

          return [
            { ...predicted, data: predGrad, requiresGrad: false },
            { ...target, data: new Float32Array(target.data.length), requiresGrad: false },
          ]
        },
      }
    : undefined

  const loss: Tensor = {
    data: lossData,
    shape: [batchSize!],
    requiresGrad: predicted.requiresGrad,
    gradFn,
  }

  return mean(loss)
}

/**
 * Huber loss (smooth L1 loss, robust to outliers)
 * Pure function with autograd
 */
export function huber(predicted: Tensor, target: Tensor, delta = 1.0): Tensor {
  const diff = sub(predicted, target)

  const lossData = new Float32Array(diff.data.length)
  for (let i = 0; i < diff.data.length; i++) {
    const absDiff = Math.abs(diff.data[i]!)
    if (absDiff <= delta) {
      lossData[i] = 0.5 * diff.data[i]! * diff.data[i]!
    } else {
      lossData[i] = delta * (absDiff - 0.5 * delta)
    }
  }

  const gradFn = predicted.requiresGrad
    ? {
        name: 'huber',
        inputs: [predicted, target],
        backward: (grad: Tensor) => {
          const predGrad = new Float32Array(predicted.data.length)
          const gradValue = grad.data[0]! / predicted.data.length

          for (let i = 0; i < diff.data.length; i++) {
            const d = diff.data[i]!
            if (Math.abs(d) <= delta) {
              predGrad[i] = gradValue * d
            } else {
              predGrad[i] = gradValue * delta * Math.sign(d)
            }
          }

          return [
            { ...predicted, data: predGrad, requiresGrad: false },
            { ...target, data: new Float32Array(target.data.length), requiresGrad: false },
          ]
        },
      }
    : undefined

  const loss: Tensor = {
    data: lossData,
    shape: predicted.shape,
    requiresGrad: predicted.requiresGrad,
    gradFn,
  }

  return mean(loss)
}
