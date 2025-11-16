import { describe, expect, test } from 'bun:test'
import { clip, crossEntropyLoss, dot, mseLoss, norm, sigmoid, softmax } from './math'

describe('Math utilities', () => {
  test('dot product', () => {
    const a = new Float32Array([1, 2, 3])
    const b = new Float32Array([4, 5, 6])
    expect(dot(a, b)).toBe(32)
  })

  test('norm calculation', () => {
    const v = new Float32Array([3, 4])
    expect(norm(v)).toBe(5)
  })

  test('clip values', () => {
    expect(clip(5, 0, 10)).toBe(5)
    expect(clip(-5, 0, 10)).toBe(0)
    expect(clip(15, 0, 10)).toBe(10)
  })

  test('sigmoid function', () => {
    expect(sigmoid(0)).toBeCloseTo(0.5)
    expect(sigmoid(10)).toBeCloseTo(1)
    expect(sigmoid(-10)).toBeCloseTo(0)
  })

  test('softmax function', () => {
    const logits = new Float32Array([1, 2, 3])
    const probs = softmax(logits)
    const sum = probs.reduce((a, b) => a + b, 0)
    expect(sum).toBeCloseTo(1)
    expect(probs[2]).toBeGreaterThan(probs[1]!)
    expect(probs[1]).toBeGreaterThan(probs[0]!)
  })

  test('cross entropy loss', () => {
    expect(crossEntropyLoss(0.5, 1)).toBeCloseTo(Math.LN2, 2)
    expect(crossEntropyLoss(0.9, 1)).toBeLessThan(crossEntropyLoss(0.5, 1))
  })

  test('MSE loss', () => {
    expect(mseLoss(1, 1)).toBe(0)
    expect(mseLoss(2, 1)).toBe(1)
    expect(mseLoss(0, 1)).toBe(1)
  })
})
