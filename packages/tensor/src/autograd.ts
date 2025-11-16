/**
 * Automatic differentiation (autograd)
 * Pure functional implementation
 */

import type { Tensor } from './types'
import { ones, zeros } from './creation'

/**
 * Build topological ordering of computation graph
 * Pure function - returns sorted array
 */
function topologicalSort(tensor: Tensor): Tensor[] {
  const visited = new Set<Tensor>()
  const sorted: Tensor[] = []

  function visit(t: Tensor): void {
    if (visited.has(t)) return
    visited.add(t)

    if (t.gradFn) {
      for (const input of t.gradFn.inputs) {
        visit(input)
      }
    }

    sorted.push(t)
  }

  visit(tensor)
  return sorted
}

/**
 * Compute gradients via backpropagation
 * Pure function - returns map of tensors to gradients
 */
export function backward(tensor: Tensor): Map<Tensor, Tensor> {
  if (!tensor.requiresGrad) {
    throw new Error('backward() called on tensor that does not require grad')
  }

  // Gradient map
  const grads = new Map<Tensor, Tensor>()

  // Start with gradient of 1 for the output
  grads.set(tensor, ones(tensor.shape))

  // Get topological order
  const topo = topologicalSort(tensor)

  // Backward pass (reverse topological order)
  for (let i = topo.length - 1; i >= 0; i--) {
    const node = topo[i]!

    if (!node.gradFn) continue

    const grad = grads.get(node)
    if (!grad) continue

    // Compute gradients for inputs
    const inputGrads = node.gradFn.backward(grad)

    // Accumulate gradients
    for (let j = 0; j < node.gradFn.inputs.length; j++) {
      const input = node.gradFn.inputs[j]!
      const inputGrad = inputGrads[j]!

      if (input.requiresGrad) {
        const existingGrad = grads.get(input)
        if (existingGrad) {
          // Sum gradients (for nodes with multiple consumers)
          const summedData = new Float32Array(existingGrad.data.length)
          for (let k = 0; k < summedData.length; k++) {
            summedData[k] = existingGrad.data[k]! + inputGrad.data[k]!
          }
          grads.set(input, {
            ...existingGrad,
            data: summedData,
          })
        } else {
          grads.set(input, inputGrad)
        }
      }
    }
  }

  return grads
}

/**
 * Zero out gradients
 * Pure function - returns new map
 */
export function zeroGrad(grads: Map<Tensor, Tensor>): Map<Tensor, Tensor> {
  const newGrads = new Map<Tensor, Tensor>()

  for (const [tensor, grad] of grads) {
    newGrads.set(tensor, zeros(grad.shape))
  }

  return newGrads
}

/**
 * Detach tensor from computation graph
 * Pure function - returns new tensor without gradFn
 */
export function detach(t: Tensor): Tensor {
  return {
    data: t.data,
    shape: t.shape,
    requiresGrad: false,
  }
}

/**
 * Enable gradient tracking
 * Pure function - returns new tensor with requiresGrad=true
 */
export function requiresGrad(t: Tensor, requires = true): Tensor {
  return {
    ...t,
    requiresGrad: requires,
  }
}
