export interface Variant {
  readonly id: string
  readonly name: string
  readonly config: Record<string, unknown>
}

export interface ABTestState {
  readonly experimentId: string
  readonly variants: ReadonlyArray<Variant>
  readonly metrics: ReadonlyMap<
    string,
    {
      impressions: number
      conversions: number
      revenue: number
      sumSquares: number
    }
  >
  readonly trafficAllocation: ReadonlyMap<string, number>
}

export interface ABTestResult {
  readonly variant: Variant
  readonly impressions: number
  readonly conversions: number
  readonly conversionRate: number
  readonly revenue: number
  readonly revenuePerUser: number
  readonly confidence: number
}

export interface StatisticalTest {
  readonly controlVariant: string
  readonly treatmentVariant: string
  readonly lift: number
  readonly pValue: number
  readonly isSignificant: boolean
  readonly confidenceInterval: [number, number]
}

export function createABTest(
  experimentId: string,
  variants: Variant[],
  trafficAllocation?: Map<string, number>
): ABTestState {
  const defaultAllocation = new Map(variants.map((v) => [v.id, 1 / variants.length]))

  const metrics = new Map(
    variants.map((v) => [v.id, { impressions: 0, conversions: 0, revenue: 0, sumSquares: 0 }])
  )

  return {
    experimentId,
    variants,
    metrics,
    trafficAllocation: trafficAllocation ?? defaultAllocation,
  }
}

export function assignVariant(state: ABTestState, userId: string): Variant {
  const hash = hashString(userId + state.experimentId)
  const randomValue = (hash % 10000) / 10000

  let cumulative = 0
  for (const variant of state.variants) {
    cumulative += state.trafficAllocation.get(variant.id) ?? 0
    if (randomValue < cumulative) {
      return variant
    }
  }

  return state.variants[state.variants.length - 1]!
}

export function trackImpression(state: ABTestState, variantId: string): ABTestState {
  const newMetrics = new Map(state.metrics)
  const current = newMetrics.get(variantId)!
  newMetrics.set(variantId, {
    ...current,
    impressions: current.impressions + 1,
  })

  return {
    ...state,
    metrics: newMetrics,
  }
}

export function trackConversion(state: ABTestState, variantId: string, revenue = 0): ABTestState {
  const newMetrics = new Map(state.metrics)
  const current = newMetrics.get(variantId)!
  newMetrics.set(variantId, {
    ...current,
    conversions: current.conversions + 1,
    revenue: current.revenue + revenue,
    sumSquares: current.sumSquares + revenue * revenue,
  })

  return {
    ...state,
    metrics: newMetrics,
  }
}

export function getResults(state: ABTestState): ABTestResult[] {
  return state.variants.map((variant) => {
    const metrics = state.metrics.get(variant.id)!
    const conversionRate = metrics.impressions > 0 ? metrics.conversions / metrics.impressions : 0
    const revenuePerUser = metrics.impressions > 0 ? metrics.revenue / metrics.impressions : 0

    const variance =
      metrics.impressions > 1
        ? (metrics.sumSquares - metrics.revenue ** 2 / metrics.impressions) /
          (metrics.impressions - 1)
        : 0

    const standardError = Math.sqrt(variance / metrics.impressions)
    const confidence = metrics.impressions > 0 ? 1 - 2 * standardError : 0

    return {
      variant,
      impressions: metrics.impressions,
      conversions: metrics.conversions,
      conversionRate,
      revenue: metrics.revenue,
      revenuePerUser,
      confidence: Math.max(0, Math.min(1, confidence)),
    }
  })
}

export function statisticalTest(
  state: ABTestState,
  controlId: string,
  treatmentId: string,
  alpha = 0.05
): StatisticalTest {
  const control = state.metrics.get(controlId)!
  const treatment = state.metrics.get(treatmentId)!

  const p1 = control.conversions / control.impressions
  const p2 = treatment.conversions / treatment.impressions

  const pooled =
    (control.conversions + treatment.conversions) / (control.impressions + treatment.impressions)

  const se = Math.sqrt(
    pooled * (1 - pooled) * (1 / control.impressions + 1 / treatment.impressions)
  )

  const zScore = (p2 - p1) / se
  const pValue = 2 * (1 - normalCDF(Math.abs(zScore)))

  const lift = ((p2 - p1) / p1) * 100

  const marginOfError = 1.96 * se
  const confidenceInterval: [number, number] = [
    (p2 - p1 - marginOfError) * 100,
    (p2 - p1 + marginOfError) * 100,
  ]

  return {
    controlVariant: controlId,
    treatmentVariant: treatmentId,
    lift,
    pValue,
    isSignificant: pValue < alpha,
    confidenceInterval,
  }
}

export function getWinner(state: ABTestState, minimumSamples = 100): string | null {
  const results = getResults(state)
  const validResults = results.filter((r) => r.impressions >= minimumSamples)

  if (validResults.length < 2) return null

  const sorted = validResults.sort((a, b) => b.conversionRate - a.conversionRate)

  const winner = sorted[0]!
  const runnerUp = sorted[1]!

  const test = statisticalTest(state, runnerUp.variant.id, winner.variant.id)

  return test.isSignificant ? winner.variant.id : null
}

function hashString(str: string): number {
  let hash = 0
  for (let i = 0; i < str.length; i++) {
    const char = str.charCodeAt(i)
    hash = (hash << 5) - hash + char
    hash = hash & hash
  }
  return Math.abs(hash)
}

function normalCDF(x: number): number {
  const t = 1 / (1 + 0.2316419 * Math.abs(x))
  const d = 0.3989423 * Math.exp((-x * x) / 2)
  const probability =
    d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))))
  return x > 0 ? 1 - probability : probability
}

export function calculateSampleSize(
  baselineRate: number,
  minimumDetectableEffect: number,
  _alpha = 0.05,
  _power = 0.8
): number {
  const zAlpha = 1.96
  const zBeta = 0.84

  const p1 = baselineRate
  const p2 = baselineRate * (1 + minimumDetectableEffect)

  const pooled = (p1 + p2) / 2
  const diff = p2 - p1

  const n = (2 * pooled * (1 - pooled) * (zAlpha + zBeta) ** 2) / diff ** 2

  return Math.ceil(n)
}
