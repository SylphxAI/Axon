/**
 * Training utilities for Axon
 * Pure functional helpers for training neural networks
 */

import type { Tensor } from '@sylphx/tensor'
import { backward } from '@sylphx/tensor'

/**
 * Extract all trainable parameters from model state
 * Recursively walks the state tree and collects all Tensors with requiresGrad
 */
export function getParams(modelState: any): Tensor[] {
  const params: Tensor[] = []

  const collect = (obj: any) => {
    if (!obj) return

    // Check if it's a Tensor
    if (obj.data && obj.shape && obj.requiresGrad) {
      params.push(obj)
      return
    }

    // Recursively check object properties
    if (typeof obj === 'object') {
      if (Array.isArray(obj)) {
        for (const item of obj) {
          collect(item)
        }
      } else {
        for (const key in obj) {
          collect(obj[key])
        }
      }
    }
  }

  collect(modelState)
  return params
}

/**
 * Update model state with new parameters
 * Recursively walks the state tree and replaces Tensors
 */
export function setParams(modelState: any, newParams: Tensor[]): any {
  let paramIndex = 0

  const update = (obj: any): any => {
    if (!obj) return obj

    // Check if it's a Tensor
    if (obj.data && obj.shape && obj.requiresGrad) {
      return newParams[paramIndex++]
    }

    // Recursively update object properties
    if (typeof obj === 'object') {
      if (Array.isArray(obj)) {
        return obj.map(item => update(item))
      } else {
        const newObj: any = {}
        for (const key in obj) {
          newObj[key] = update(obj[key])
        }
        return newObj
      }
    }

    return obj
  }

  return update(modelState)
}

/**
 * Zero out gradients for all parameters
 */
export function zeroGrad(params: Tensor[]): Tensor[] {
  return params.map(p => ({ ...p, grad: undefined }))
}

/**
 * Training step - pure function
 * Performs one forward pass, backward pass, and optimizer step
 */
export function trainStep<M, O>(config: {
  model: { forward: (input: Tensor, state: M) => Tensor }
  modelState: M
  optimizer: { step: (params: Tensor[], grads: Tensor[], state: O) => { params: Tensor[], state: O } }
  optState: O
  input: Tensor
  target: Tensor
  lossFn: (pred: Tensor, target: Tensor) => Tensor
}): {
  modelState: M
  optState: O
  loss: number
} {
  const { model, modelState, optimizer, optState, input, target, lossFn } = config

  // Forward pass
  const output = model.forward(input, modelState)

  // Compute loss
  const loss = lossFn(output, target)

  // Backward pass
  const gradsMap = backward(loss)

  // Get parameters and gradients
  const params = getParams(modelState)
  const grads = params.map(p => {
    if (!p.grad) {
      throw new Error(`Parameter has no gradient after backward pass. requiresGrad=${p.requiresGrad}`)
    }
    return p.grad
  })

  // Optimizer step
  const { params: newParams, state: newOptState } = optimizer.step(params, grads, optState)

  // Update model state
  const newModelState = setParams(modelState, newParams)

  // CRITICAL: Clear ALL gradients and gradFn to prevent memory leak
  // Clear gradients from all tensors in the computation graph
  for (const [tensor, _] of gradsMap) {
    // @ts-ignore - mutating to clear gradient
    tensor.grad = undefined
    // @ts-ignore - mutating to clear computation graph
    tensor.gradFn = undefined
  }

  // Also clear gradients on new parameters
  for (const p of newParams) {
    // @ts-ignore - mutating for compatibility
    p.grad = undefined
  }

  return {
    modelState: newModelState,
    optState: newOptState,
    loss: loss.data[0]!
  }
}

/**
 * Train for multiple epochs
 * Callback receives epoch number and loss, returns false to stop early
 */
export function train<M, O>(config: {
  model: { forward: (input: Tensor, state: M) => Tensor }
  modelState: M
  optimizer: { step: (params: Tensor[], grads: Tensor[], state: O) => { params: Tensor[], state: O } }
  optState: O
  input: Tensor
  target: Tensor
  lossFn: (pred: Tensor, target: Tensor) => Tensor
  epochs: number
  onEpoch?: (epoch: number, loss: number) => boolean | void
}): {
  modelState: M
  optState: O
  losses: number[]
} {
  const { model, optimizer, input, target, lossFn, epochs, onEpoch } = config
  let { modelState, optState } = config

  const losses: number[] = []

  for (let epoch = 0; epoch < epochs; epoch++) {
    const result = trainStep({
      model,
      modelState,
      optimizer,
      optState,
      input,
      target,
      lossFn
    })

    modelState = result.modelState
    optState = result.optState
    losses.push(result.loss)

    // Callback
    if (onEpoch) {
      const shouldContinue = onEpoch(epoch, result.loss)
      if (shouldContinue === false) break
    }
  }

  return { modelState, optState, losses }
}
