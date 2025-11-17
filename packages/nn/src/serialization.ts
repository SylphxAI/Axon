/**
 * Model serialization and deserialization
 * Pure functional save/load for neural networks
 */

import type { Tensor } from '@sylphx/tensor'
import { tensor } from '@sylphx/tensor'

/**
 * Serializable model state
 */
export type SerializedModel = {
  version: string
  timestamp: string
  layers: SerializedLayer[]
  metadata?: Record<string, unknown>
}

/**
 * Serialized layer
 */
export type SerializedLayer = {
  type: string
  name?: string
  params: SerializedParam[]
  config?: Record<string, unknown>
}

/**
 * Serialized parameter
 */
export type SerializedParam = {
  name: string
  shape: number[]
  data: number[]
  requiresGrad: boolean
}

/**
 * Serialize tensor to plain object
 */
export function serializeTensor(t: Tensor, name: string): SerializedParam {
  return {
    name,
    shape: [...t.shape],
    data: Array.from(t.data),
    requiresGrad: t.requiresGrad,
  }
}

/**
 * Deserialize tensor from plain object
 */
export function deserializeTensor(param: SerializedParam): Tensor {
  return tensor(
    param.shape.length === 1 ? param.data : reshape2D(param.data, param.shape),
    { requiresGrad: param.requiresGrad }
  )
}

/**
 * Reshape 1D array to 2D
 */
function reshape2D(data: number[], shape: number[]): number[][] {
  if (shape.length !== 2) {
    throw new Error('Only 2D reshaping supported')
  }

  const [rows, cols] = shape
  const result: number[][] = []

  for (let i = 0; i < rows!; i++) {
    result.push(data.slice(i * cols!, (i + 1) * cols!))
  }

  return result
}

/**
 * Save model to JSON string
 */
export function saveModel(model: SerializedModel): string {
  return JSON.stringify(model, null, 2)
}

/**
 * Load model from JSON string
 */
export function loadModel(json: string): SerializedModel {
  const model = JSON.parse(json) as SerializedModel

  // Validate version
  if (!model.version) {
    throw new Error('Invalid model: missing version')
  }

  // Validate structure
  if (!Array.isArray(model.layers)) {
    throw new Error('Invalid model: missing layers')
  }

  return model
}

/**
 * Save model to file (Node.js/Bun)
 */
export async function saveModelToFile(
  model: SerializedModel,
  filepath: string
): Promise<void> {
  const fs = await import('fs/promises')
  const json = saveModel(model)
  await fs.writeFile(filepath, json, 'utf-8')
}

/**
 * Load model from file (Node.js/Bun)
 */
export async function loadModelFromFile(filepath: string): Promise<SerializedModel> {
  const fs = await import('fs/promises')
  const json = await fs.readFile(filepath, 'utf-8')
  return loadModel(json)
}

/**
 * Get model summary
 */
export function getModelSummary(model: SerializedModel): string {
  let summary = `Model (${model.version})\n`
  summary += `Timestamp: ${model.timestamp}\n`
  summary += `Layers: ${model.layers.length}\n\n`

  let totalParams = 0

  for (const layer of model.layers) {
    const layerParams = layer.params.reduce(
      (sum, p) => sum + p.data.length,
      0
    )
    totalParams += layerParams

    summary += `${layer.type}${layer.name ? ` (${layer.name})` : ''}\n`
    summary += `  Parameters: ${layerParams.toLocaleString()}\n`

    for (const param of layer.params) {
      summary += `    ${param.name}: [${param.shape.join(', ')}]\n`
    }
  }

  summary += `\nTotal Parameters: ${totalParams.toLocaleString()}`

  return summary
}
