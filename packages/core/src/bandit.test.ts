import { describe, expect, test } from 'bun:test'
import {
  createBanditState,
  epsilonGreedy,
  getBanditStats,
  thompsonSampling,
  ucbSelection,
  updateBandit,
} from './bandit'

describe('Multi-Armed Bandit', () => {
  test('create initial state', () => {
    const state = createBanditState(['A', 'B', 'C'])

    expect(state.arms.length).toBe(3)
    expect(state.totalPulls).toBe(0)
    expect(state.arms[0]?.alpha).toBe(1)
    expect(state.arms[0]?.beta).toBe(1)
  })

  test('thompson sampling selects arm', () => {
    const state = createBanditState(['A', 'B', 'C'])
    const selection = thompsonSampling(state)

    expect(selection.armId).toBeDefined()
    expect(['A', 'B', 'C']).toContain(selection.armId)
    expect(selection.expectedReward).toBeGreaterThanOrEqual(0)
    expect(selection.expectedReward).toBeLessThanOrEqual(1)
  })

  test('ucb selection', () => {
    const state = createBanditState(['A', 'B', 'C'])
    const selection = ucbSelection(state)

    expect(selection.armId).toBeDefined()
    expect(['A', 'B', 'C']).toContain(selection.armId)
  })

  test('epsilon greedy', () => {
    const state = createBanditState(['A', 'B', 'C'])
    const selection = epsilonGreedy(state, 0.1)

    expect(selection.armId).toBeDefined()
  })

  test('update with reward', () => {
    let state = createBanditState(['A', 'B'])

    state = updateBandit(state, 'A', 1)
    expect(state.totalPulls).toBe(1)

    const armA = state.arms.find((a) => a.id === 'A')
    expect(armA?.alpha).toBe(2)
    expect(armA?.beta).toBe(1)
  })

  test('learns best arm over time', () => {
    let state = createBanditState(['good', 'bad'])

    for (let i = 0; i < 100; i++) {
      const selection = thompsonSampling(state)

      const reward = selection.armId === 'good' ? 1 : 0
      state = updateBandit(state, selection.armId, reward)
    }

    const stats = getBanditStats(state)
    const goodArm = stats.find((s) => s.id === 'good')!
    const badArm = stats.find((s) => s.id === 'bad')!

    expect(goodArm.pulls).toBeGreaterThan(badArm.pulls)
    expect(goodArm.successRate).toBeGreaterThan(badArm.successRate)
  })

  test('get bandit statistics', () => {
    let state = createBanditState(['A', 'B'])
    state = updateBandit(state, 'A', 1)
    state = updateBandit(state, 'A', 1)
    state = updateBandit(state, 'B', 0)

    const stats = getBanditStats(state)

    expect(stats.length).toBe(2)
    const armA = stats.find((s) => s.id === 'A')!
    expect(armA.pulls).toBe(2)
    expect(armA.successRate).toBeGreaterThan(0.5)
  })
})
