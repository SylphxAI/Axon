import { describe, expect, test } from 'bun:test'
import * as T from './index'

describe('Tensor Creation', () => {
  test('creates tensor from array', () => {
    const t = T.tensor([1, 2, 3])
    expect(t.data).toEqual(new Float32Array([1, 2, 3]))
    expect(t.shape).toEqual([3])
  })

  test('creates 2D tensor', () => {
    const t = T.tensor([[1, 2], [3, 4]])
    expect(t.data).toEqual(new Float32Array([1, 2, 3, 4]))
    expect(t.shape).toEqual([2, 2])
  })

  test('creates zeros', () => {
    const t = T.zeros([2, 3])
    expect(t.data).toEqual(new Float32Array([0, 0, 0, 0, 0, 0]))
    expect(t.shape).toEqual([2, 3])
  })

  test('creates ones', () => {
    const t = T.ones([2, 2])
    expect(t.data).toEqual(new Float32Array([1, 1, 1, 1]))
  })

  test('creates scalar', () => {
    const t = T.scalar(5)
    expect(T.item(t)).toBe(5)
  })
})

describe('Tensor Operations', () => {
  test('add tensors', () => {
    const a = T.tensor([1, 2, 3])
    const b = T.tensor([4, 5, 6])
    const c = T.add(a, b)
    expect(Array.from(c.data)).toEqual([5, 7, 9])
  })

  test('multiply tensors', () => {
    const a = T.tensor([1, 2, 3])
    const b = T.tensor([2, 2, 2])
    const c = T.mul(a, b)
    expect(Array.from(c.data)).toEqual([2, 4, 6])
  })

  test('matrix multiplication', () => {
    const a = T.tensor([[1, 2], [3, 4]])
    const b = T.tensor([[5, 6], [7, 8]])
    const c = T.matmul(a, b)
    // [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]]
    // [[19, 22], [43, 50]]
    expect(Array.from(c.data)).toEqual([19, 22, 43, 50])
    expect(c.shape).toEqual([2, 2])
  })

  test('transpose', () => {
    const a = T.tensor([[1, 2, 3], [4, 5, 6]])
    const b = T.transpose(a)
    expect(b.shape).toEqual([3, 2])
    expect(T.toArray(b)).toEqual([[1, 4], [2, 5], [3, 6]])
  })

  test('sum', () => {
    const a = T.tensor([1, 2, 3, 4])
    const s = T.sum(a)
    expect(T.item(s)).toBe(10)
  })

  test('mean', () => {
    const a = T.tensor([1, 2, 3, 4])
    const m = T.mean(a)
    expect(T.item(m)).toBe(2.5)
  })
})

describe('Autograd', () => {
  test('backward on simple addition', () => {
    const a = T.tensor([2, 3], { requiresGrad: true })
    const b = T.tensor([4, 5], { requiresGrad: true })
    const c = T.add(a, b)

    const grads = T.backward(c)

    // dc/da = 1, dc/db = 1
    expect(Array.from(grads.get(a)!.data)).toEqual([1, 1])
    expect(Array.from(grads.get(b)!.data)).toEqual([1, 1])
  })

  test('backward on multiplication', () => {
    const a = T.tensor([2, 3], { requiresGrad: true })
    const b = T.tensor([4, 5], { requiresGrad: true })
    const c = T.mul(a, b)

    const grads = T.backward(c)

    // dc/da = b, dc/db = a
    expect(Array.from(grads.get(a)!.data)).toEqual([4, 5])
    expect(Array.from(grads.get(b)!.data)).toEqual([2, 3])
  })

  test('backward on chain', () => {
    const x = T.tensor([2], { requiresGrad: true })
    const y = T.mul(x, x) // y = x^2
    const z = T.add(y, y) // z = 2x^2

    const grads = T.backward(z)

    // dz/dx = 4x = 4*2 = 8
    expect(T.item(grads.get(x)!)).toBe(8)
  })

  test('backward on matmul', () => {
    const a = T.tensor([[1, 2]], { requiresGrad: true })
    const b = T.tensor([[3], [4]], { requiresGrad: true })
    const c = T.matmul(a, b)

    const grads = T.backward(c)

    // dc/da = b^T, dc/db = a^T
    expect(T.toArray(grads.get(a)!)).toEqual([[3, 4]])
    expect(T.toArray(grads.get(b)!)).toEqual([[1], [2]])
  })
})
