import type { Vector } from './types'

export interface BanditArm {
  readonly id: string
  readonly features: Vector
  readonly alpha: number
  readonly beta: number
}

export interface BanditState {
  readonly arms: ReadonlyArray<BanditArm>
  readonly totalPulls: number
}

export interface BanditSelection {
  readonly armId: string
  readonly expectedReward: number
  readonly sampledValue: number
}

function betaSample(alpha: number, beta: number): number {
  const a = alpha
  const b = beta

  if (a <= 0 || b <= 0) return 0.5

  const gamma1 = gammaSample(a)
  const gamma2 = gammaSample(b)
  return gamma1 / (gamma1 + gamma2)
}

function gammaSample(shape: number): number {
  if (shape < 1) {
    return gammaSample(shape + 1) * Math.random() ** (1 / shape)
  }

  const d = shape - 1 / 3
  const c = 1 / Math.sqrt(9 * d)

  while (true) {
    const x = normalSample()
    const v = 1 + c * x
    if (v <= 0) continue

    const v3 = v * v * v
    const u = Math.random()

    if (u < 1 - 0.0331 * x * x * x * x) return d * v3
    if (Math.log(u) < 0.5 * x * x + d * (1 - v3 + Math.log(v3))) return d * v3
  }
}

function normalSample(): number {
  const u1 = Math.random()
  const u2 = Math.random()
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2)
}

export function thompsonSampling(state: BanditState): BanditSelection {
  const samples = state.arms.map((arm) => ({
    armId: arm.id,
    sampledValue: betaSample(arm.alpha, arm.beta),
    expectedReward: arm.alpha / (arm.alpha + arm.beta),
  }))

  const best = samples.reduce((max, curr) => (curr.sampledValue > max.sampledValue ? curr : max))

  return best
}

export function ucbSelection(state: BanditState, c = 2): BanditSelection {
  const samples = state.arms.map((arm) => {
    const meanReward = arm.alpha / (arm.alpha + arm.beta)
    const pulls = arm.alpha + arm.beta - 2
    const exploration =
      pulls > 0 ? Math.sqrt((c * Math.log(state.totalPulls)) / pulls) : Number.POSITIVE_INFINITY

    return {
      armId: arm.id,
      expectedReward: meanReward,
      sampledValue: meanReward + exploration,
    }
  })

  const best = samples.reduce((max, curr) => (curr.sampledValue > max.sampledValue ? curr : max))

  return best
}

export function epsilonGreedy(state: BanditState, epsilon = 0.1): BanditSelection {
  if (Math.random() < epsilon) {
    const randomArm = state.arms[Math.floor(Math.random() * state.arms.length)]!
    return {
      armId: randomArm.id,
      expectedReward: randomArm.alpha / (randomArm.alpha + randomArm.beta),
      sampledValue: Math.random(),
    }
  }

  const best = state.arms.reduce((max, curr) => {
    const currReward = curr.alpha / (curr.alpha + curr.beta)
    const maxReward = max.alpha / (max.alpha + max.beta)
    return currReward > maxReward ? curr : max
  })

  return {
    armId: best.id,
    expectedReward: best.alpha / (best.alpha + best.beta),
    sampledValue: 1,
  }
}

export function updateBandit(state: BanditState, armId: string, reward: number): BanditState {
  return {
    ...state,
    arms: state.arms.map((arm) =>
      arm.id === armId
        ? {
            ...arm,
            alpha: arm.alpha + reward,
            beta: arm.beta + (1 - reward),
          }
        : arm
    ),
    totalPulls: state.totalPulls + 1,
  }
}

export function createBanditState(armIds: string[]): BanditState {
  return {
    arms: armIds.map((id) => ({
      id,
      features: new Float32Array(0),
      alpha: 1,
      beta: 1,
    })),
    totalPulls: 0,
  }
}

export function getBanditStats(state: BanditState) {
  return state.arms.map((arm) => ({
    id: arm.id,
    pulls: arm.alpha + arm.beta - 2,
    successRate: arm.alpha / (arm.alpha + arm.beta),
    alpha: arm.alpha,
    beta: arm.beta,
  }))
}
