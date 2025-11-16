import type { BanditSelection, BanditState } from '@sylphx/neuronline-core'
import {
  createBanditState,
  getBanditStats,
  thompsonSampling,
  updateBandit,
} from '@sylphx/neuronline-core'

export interface Product {
  readonly id: string
  readonly name: string
  readonly price: number
  readonly category: string
  readonly features: ReadonlyArray<number>
}

export interface UserContext {
  readonly userId: string
  readonly device: 'mobile' | 'tablet' | 'desktop'
  readonly location?: string
  readonly recentViews: ReadonlyArray<string>
  readonly recentPurchases: ReadonlyArray<string>
}

export interface RecommendationResult {
  readonly product: Product
  readonly score: number
  readonly reason: 'exploration' | 'exploitation'
}

export interface RecommenderState {
  readonly banditState: BanditState
  readonly products: ReadonlyMap<string, Product>
  readonly totalRecommendations: number
  readonly totalConversions: number
}

export function createRecommenderState(products: Product[]): RecommenderState {
  return {
    banditState: createBanditState(products.map((p) => p.id)),
    products: new Map(products.map((p) => [p.id, p])),
    totalRecommendations: 0,
    totalConversions: 0,
  }
}

export function recommend(
  state: RecommenderState,
  _context: UserContext,
  count = 5
): {
  recommendations: RecommendationResult[]
  newState: RecommenderState
} {
  const selections: BanditSelection[] = []
  const currentBanditState = state.banditState

  for (let i = 0; i < count; i++) {
    const availableArms = currentBanditState.arms.filter(
      (arm) => !selections.some((s) => s.armId === arm.id)
    )

    if (availableArms.length === 0) break

    const tempState = { ...currentBanditState, arms: availableArms }
    const selection = thompsonSampling(tempState)
    selections.push(selection)
  }

  const recommendations = selections.map((selection) => {
    const product = state.products.get(selection.armId)!
    const reason: 'exploration' | 'exploitation' =
      selection.sampledValue > selection.expectedReward ? 'exploration' : 'exploitation'
    return {
      product,
      score: selection.expectedReward,
      reason,
    }
  })

  return {
    recommendations,
    newState: {
      ...state,
      totalRecommendations: state.totalRecommendations + count,
    },
  }
}

export function trackFeedback(
  state: RecommenderState,
  productId: string,
  action: 'view' | 'click' | 'purchase'
): RecommenderState {
  const reward = action === 'purchase' ? 1 : action === 'click' ? 0.3 : 0.1

  return {
    ...state,
    banditState: updateBandit(state.banditState, productId, reward),
    totalConversions: action === 'purchase' ? state.totalConversions + 1 : state.totalConversions,
  }
}

export function getRecommenderStats(state: RecommenderState) {
  const banditStats = getBanditStats(state.banditState)
  const conversionRate =
    state.totalRecommendations > 0 ? state.totalConversions / state.totalRecommendations : 0

  return {
    products: banditStats.map((stat: {
      id: string
      pulls: number
      successRate: number
      alpha: number
      beta: number
    }) => ({
      id: stat.id,
      name: state.products.get(stat.id)?.name ?? 'Unknown',
      recommendations: stat.pulls,
      successRate: stat.successRate,
      confidence: calculateConfidence(stat.alpha, stat.beta),
    })),
    overall: {
      totalRecommendations: state.totalRecommendations,
      totalConversions: state.totalConversions,
      conversionRate,
    },
  }
}

function calculateConfidence(alpha: number, beta: number): number {
  const variance = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))
  const stdDev = Math.sqrt(variance)
  return Math.min(1, 1 - 2 * stdDev)
}

export function getTopProducts(state: RecommenderState, count = 10) {
  const stats = getBanditStats(state.banditState)
  return stats
    .sort((a: { successRate: number }, b: { successRate: number }) => b.successRate - a.successRate)
    .slice(0, count)
    .map((stat: { id: string; successRate: number; pulls: number }) => ({
      product: state.products.get(stat.id)!,
      successRate: stat.successRate,
      totalRecommendations: stat.pulls,
    }))
}
