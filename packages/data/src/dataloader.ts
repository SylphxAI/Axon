/**
 * DataLoader - Pure functional data loading and batching
 */

import type { Tensor } from '@sylphx/tensor'
import { tensor } from '@sylphx/tensor'

/**
 * Dataset interface
 */
export type Dataset<T> = {
  readonly length: number
  getItem: (index: number) => T
}

/**
 * DataLoader configuration
 */
export type DataLoaderConfig = {
  readonly batchSize: number
  readonly shuffle?: boolean
  readonly dropLast?: boolean
}

/**
 * Batch of data
 */
export type Batch = {
  readonly data: Tensor
  readonly labels: Tensor
}

/**
 * Create a simple dataset from arrays
 */
export function createDataset(
  data: number[][],
  labels: number[][]
): Dataset<{ data: number[]; label: number[] }> {
  return {
    length: data.length,
    getItem: (index: number) => ({
      data: data[index]!,
      label: labels[index]!,
    }),
  }
}

/**
 * Shuffle array indices
 */
function shuffleIndices(length: number): number[] {
  const indices = Array.from({ length }, (_, i) => i)

  // Fisher-Yates shuffle
  for (let i = indices.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1))
    ;[indices[i], indices[j]] = [indices[j]!, indices[i]!]
  }

  return indices
}

/**
 * Create batches from dataset
 */
export function* createBatches(
  dataset: Dataset<{ data: number[]; label: number[] }>,
  config: DataLoaderConfig
): Generator<Batch> {
  const { batchSize, shuffle = false, dropLast = false } = config

  // Get indices
  let indices = Array.from({ length: dataset.length }, (_, i) => i)
  if (shuffle) {
    indices = shuffleIndices(dataset.length)
  }

  // Generate batches
  for (let i = 0; i < indices.length; i += batchSize) {
    const batchIndices = indices.slice(i, i + batchSize)

    // Skip incomplete batch if dropLast
    if (dropLast && batchIndices.length < batchSize) {
      continue
    }

    // Collect batch data
    const batchData: number[][] = []
    const batchLabels: number[][] = []

    for (const idx of batchIndices) {
      const item = dataset.getItem(idx)
      batchData.push(item.data)
      batchLabels.push(item.label)
    }

    yield {
      data: tensor(batchData, { requiresGrad: false }),
      labels: tensor(batchLabels, { requiresGrad: false }),
    }
  }
}

/**
 * Split dataset into train/validation/test sets
 */
export function splitDataset<T>(
  dataset: Dataset<T>,
  splits: { train: number; val?: number; test?: number }
): {
  train: Dataset<T>
  val?: Dataset<T>
  test?: Dataset<T>
} {
  const total = splits.train + (splits.val || 0) + (splits.test || 0)
  if (Math.abs(total - 1.0) > 1e-6) {
    throw new Error('Splits must sum to 1.0')
  }

  const trainSize = Math.floor(dataset.length * splits.train)
  const valSize = splits.val ? Math.floor(dataset.length * splits.val) : 0

  return {
    train: {
      length: trainSize,
      getItem: (index: number) => dataset.getItem(index),
    },
    val: splits.val
      ? {
          length: valSize,
          getItem: (index: number) => dataset.getItem(trainSize + index),
        }
      : undefined,
    test: splits.test
      ? {
          length: dataset.length - trainSize - valSize,
          getItem: (index: number) =>
            dataset.getItem(trainSize + valSize + index),
        }
      : undefined,
  }
}

/**
 * Normalize dataset features
 */
export function normalizeData(
  data: number[][]
): { normalized: number[][]; mean: number[]; std: number[] } {
  const numFeatures = data[0]!.length
  const mean = new Array(numFeatures).fill(0)
  const std = new Array(numFeatures).fill(0)

  // Calculate mean
  for (const sample of data) {
    for (let i = 0; i < numFeatures; i++) {
      mean[i] += sample[i]! / data.length
    }
  }

  // Calculate std
  for (const sample of data) {
    for (let i = 0; i < numFeatures; i++) {
      std[i] += Math.pow(sample[i]! - mean[i]!, 2) / data.length
    }
  }

  for (let i = 0; i < numFeatures; i++) {
    std[i] = Math.sqrt(std[i]!) + 1e-8 // Add epsilon for stability
  }

  // Normalize
  const normalized = data.map((sample) =>
    sample.map((value, i) => (value - mean[i]!) / std[i]!)
  )

  return { normalized, mean, std }
}
