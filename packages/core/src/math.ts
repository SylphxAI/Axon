import type { Vector } from './types'

export function dot(a: Vector, b: Vector): number {
  let sum = 0
  const len = Math.min(a.length, b.length)
  for (let i = 0; i < len; i++) {
    sum += a[i]! * b[i]!
  }
  return sum
}

export function add(a: Vector, b: Vector, scale = 1): Vector {
  const result = new Float32Array(a.length)
  for (let i = 0; i < a.length; i++) {
    result[i] = a[i]! + scale * b[i]!
  }
  return result
}

export function scale(v: Vector, scalar: number): Vector {
  const result = new Float32Array(v.length)
  for (let i = 0; i < v.length; i++) {
    result[i] = v[i]! * scalar
  }
  return result
}

export function norm(v: Vector): number {
  return Math.sqrt(dot(v, v))
}

export function clip(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value))
}

export function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x))
}

export function sigmoidGradient(x: number): number {
  const s = sigmoid(x)
  return s * (1 - s)
}

export function relu(x: number): number {
  return Math.max(0, x)
}

export function reluGradient(x: number): number {
  return x > 0 ? 1 : 0
}

export function tanh(x: number): number {
  return Math.tanh(x)
}

export function tanhGradient(x: number): number {
  const t = Math.tanh(x)
  return 1 - t * t
}

export function softmax(logits: Vector): Vector {
  const maxLogit = Math.max(...logits)
  const exps = new Float32Array(logits.length)
  let sum = 0

  for (let i = 0; i < logits.length; i++) {
    exps[i] = Math.exp(logits[i]! - maxLogit)
    sum += exps[i]!
  }

  for (let i = 0; i < exps.length; i++) {
    exps[i] = exps[i]! / sum
  }

  return exps
}

export function crossEntropyLoss(predicted: number, actual: number): number {
  const epsilon = 1e-15
  const p = clip(predicted, epsilon, 1 - epsilon)
  return -(actual * Math.log(p) + (1 - actual) * Math.log(1 - p))
}

export function mseLoss(predicted: number, actual: number): number {
  const diff = predicted - actual
  return diff * diff
}
