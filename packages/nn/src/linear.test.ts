import { describe, expect, test } from 'bun:test'
import * as Linear from './linear'
import * as T from '@neuronline/tensor'

describe('Linear Layer', () => {
  test('initialization creates correct shapes', () => {
    const state = Linear.init(10, 5)

    expect(state.weight.shape).toEqual([5, 10])
    expect(state.bias.shape).toEqual([5])
    expect(state.weight.requiresGrad).toBe(true)
    expect(state.bias.requiresGrad).toBe(true)
  })

  test('forward pass produces correct output shape', () => {
    const state = Linear.init(10, 5)
    const input = T.randn([3, 10]) // batch of 3

    const output = Linear.forward(input, state)

    expect(output.shape).toEqual([3, 5])
  })

  test('forward pass computation', () => {
    // Manual weights for predictable output
    const weight = T.tensor([[1, 2], [3, 4]]) // [2, 2]
    const bias = T.tensor([10, 20]) // [2]
    const state = { weight, bias }

    const input = T.tensor([[1, 1]]) // [1, 2]
    const output = Linear.forward(input, state)

    // output = input @ weight^T + bias
    // weight^T = [[1, 3], [2, 4]]
    // [[1, 1]] @ [[1, 3], [2, 4]] = [[3, 7]]
    // [[3, 7]] + [[10, 20]] = [[13, 27]]
    expect(T.toArray(output)).toEqual([[13, 27]])
  })

  test('parameters returns weight and bias', () => {
    const state = Linear.init(5, 3)
    const params = Linear.parameters(state)

    expect(params).toHaveLength(2)
    expect(params[0]).toBe(state.weight)
    expect(params[1]).toBe(state.bias)
  })

  test('updateWeights creates new state', () => {
    const state = Linear.init(5, 3)
    const newWeight = T.randn([3, 5])
    const newBias = T.randn([3])

    const newState = Linear.updateWeights(state, newWeight, newBias)

    expect(newState.weight).toBe(newWeight)
    expect(newState.bias).toBe(newBias)
    expect(newState).not.toBe(state) // Immutable
  })

  test('gradient flows through layer', () => {
    const state = Linear.init(3, 2)
    const input = T.tensor([[1, 2, 3]], { requiresGrad: true })

    const output = Linear.forward(input, state)
    const grads = T.backward(output)

    // Should have gradients for input, weight, and bias
    expect(grads.get(input)).toBeDefined()
    expect(grads.get(state.weight)).toBeDefined()
    expect(grads.get(state.bias)).toBeDefined()
  })
})
