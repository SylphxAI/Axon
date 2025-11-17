import { describe, expect, test } from 'bun:test'
import * as SGD from './sgd'
import * as T from '@neuronline/tensor'

describe('SGD Optimizer', () => {
  test('init creates correct state', () => {
    const params = [T.randn([2, 3]), T.randn([2])]
    const state = SGD.init(params, { lr: 0.01, momentum: 0.9 })

    expect(state.step).toBe(0)
    expect(state.params).toHaveLength(2)
    const config = state.state.config as SGD.SGDConfig
    expect(config.lr).toBe(0.01)
    expect(config.momentum).toBe(0.9)
  })

  test('step updates parameters with gradients', () => {
    const param = T.tensor([1, 2, 3], { requiresGrad: true })
    const optState = SGD.init([param], { lr: 0.1 })

    const grads = new Map()
    grads.set(param, T.tensor([1, 1, 1]))

    const result = SGD.step(optState, [param], grads)

    // new_param = param - lr * grad = [1,2,3] - 0.1*[1,1,1] = [0.9,1.9,2.9]
    const data = Array.from(result.params[0]!.data)
    expect(data[0]).toBeCloseTo(0.9)
    expect(data[1]).toBeCloseTo(1.9)
    expect(data[2]).toBeCloseTo(2.9)
  })

  test('handles multiple parameters', () => {
    const param1 = T.tensor([1, 2], { requiresGrad: true })
    const param2 = T.tensor([3, 4], { requiresGrad: true })
    const optState = SGD.init([param1, param2], { lr: 0.1 })

    const grads = new Map()
    grads.set(param1, T.tensor([1, 1]))
    grads.set(param2, T.tensor([2, 2]))

    const result = SGD.step(optState, [param1, param2], grads)

    expect(result.params).toHaveLength(2)
    const data1 = Array.from(result.params[0]!.data)
    const data2 = Array.from(result.params[1]!.data)
    expect(data1[0]).toBeCloseTo(0.9)
    expect(data1[1]).toBeCloseTo(1.9)
    expect(data2[0]).toBeCloseTo(2.8)
    expect(data2[1]).toBeCloseTo(3.8)
  })
})
