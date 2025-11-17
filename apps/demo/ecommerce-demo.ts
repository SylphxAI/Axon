import type { Product, UserContext } from '@sylphx/predictors'
import {
  assignVariant,
  calculateSampleSize,
  createABTest,
  createRecommenderState,
  getRecommenderStats,
  getResults,
  getTopProducts,
  getWinner,
  recommend,
  statisticalTest,
  trackConversion,
  trackFeedback,
  trackImpression,
} from '@sylphx/predictors'

console.log('🛒 E-commerce Recommendation System Demo\n')

const products: Product[] = [
  {
    id: 'p1',
    name: 'Wireless Headphones',
    price: 99.99,
    category: 'Electronics',
    features: [1, 0, 0],
  },
  {
    id: 'p2',
    name: 'Running Shoes',
    price: 79.99,
    category: 'Sports',
    features: [0, 1, 0],
  },
  {
    id: 'p3',
    name: 'Coffee Maker',
    price: 49.99,
    category: 'Home',
    features: [0, 0, 1],
  },
  {
    id: 'p4',
    name: 'Yoga Mat',
    price: 29.99,
    category: 'Sports',
    features: [0, 1, 0],
  },
  {
    id: 'p5',
    name: 'Smart Watch',
    price: 199.99,
    category: 'Electronics',
    features: [1, 0, 0],
  },
]

function demoRecommendationSystem() {
  console.log('='.repeat(60))
  console.log('📊 Product Recommendation with Thompson Sampling')
  console.log('='.repeat(60))

  let state = createRecommenderState(products)

  const users: UserContext[] = [
    {
      userId: 'user1',
      device: 'mobile',
      recentViews: [],
      recentPurchases: [],
    },
    {
      userId: 'user2',
      device: 'desktop',
      recentViews: ['p1'],
      recentPurchases: [],
    },
    {
      userId: 'user3',
      device: 'tablet',
      recentViews: [],
      recentPurchases: [],
    },
  ]

  console.log('\n📝 Simulating user interactions...\n')

  for (let round = 0; round < 50; round++) {
    for (const user of users) {
      const { recommendations, newState } = recommend(state, user, 3)
      state = newState

      for (const rec of recommendations) {
        const willClick = Math.random() < getProductQuality(rec.product.id)
        if (willClick) {
          state = trackFeedback(state, rec.product.id, 'click')

          const willPurchase = Math.random() < 0.3
          if (willPurchase) {
            state = trackFeedback(state, rec.product.id, 'purchase')
          }
        } else {
          state = trackFeedback(state, rec.product.id, 'view')
        }
      }
    }
  }

  console.log('✅ Simulation complete!\n')

  const stats = getRecommenderStats(state)
  console.log('📈 Overall Performance:')
  console.log(`   Total Recommendations: ${stats.overall.totalRecommendations}`)
  console.log(`   Total Conversions: ${stats.overall.totalConversions}`)
  console.log(`   Conversion Rate: ${(stats.overall.conversionRate * 100).toFixed(2)}%`)

  console.log('\n🏆 Top Performing Products:')
  const topProducts = getTopProducts(state, 3)
  for (const { product, successRate, totalRecommendations } of topProducts) {
    console.log(
      `   ${product.name}: ${(successRate * 100).toFixed(1)}% success (${totalRecommendations} recommendations)`
    )
  }

  console.log('\n📊 All Products Statistics:')
  for (const productStat of stats.products) {
    console.log(
      `   ${productStat.name}: ${productStat.recommendations} recs, ${(productStat.successRate * 100).toFixed(1)}% success, ${(productStat.confidence * 100).toFixed(1)}% confidence`
    )
  }

  console.log('\n💡 Recommendation Example:')
  const testUser: UserContext = {
    userId: 'test-user',
    device: 'mobile',
    recentViews: [],
    recentPurchases: [],
  }
  const { recommendations } = recommend(state, testUser, 3)
  for (const rec of recommendations) {
    console.log(`   ${rec.product.name} - Score: ${(rec.score * 100).toFixed(1)}% (${rec.reason})`)
  }

  console.log('\n')
}

function getProductQuality(productId: string): number {
  const quality: Record<string, number> = {
    p1: 0.8,
    p2: 0.6,
    p3: 0.4,
    p4: 0.5,
    p5: 0.9,
  }
  return quality[productId] ?? 0.5
}

function demoABTesting() {
  console.log('='.repeat(60))
  console.log('🧪 A/B Testing Framework Demo')
  console.log('='.repeat(60))

  let state = createABTest('checkout-flow-test', [
    { id: 'control', name: 'Original Checkout', config: { steps: 3 } },
    { id: 'treatment', name: 'Simplified Checkout', config: { steps: 1 } },
  ])

  console.log('\n📝 Running experiment...\n')

  for (let i = 0; i < 1000; i++) {
    const userId = `user-${i}`
    const variant = assignVariant(state, userId)

    state = trackImpression(state, variant.id)

    const conversionRate = variant.id === 'treatment' ? 0.15 : 0.12
    const willConvert = Math.random() < conversionRate

    if (willConvert) {
      const revenue = 50 + Math.random() * 100
      state = trackConversion(state, variant.id, revenue)
    }
  }

  console.log('✅ Experiment complete!\n')

  const results = getResults(state)
  console.log('📊 Experiment Results:\n')

  for (const result of results) {
    console.log(`${result.variant.name}:`)
    console.log(`  Impressions: ${result.impressions}`)
    console.log(`  Conversions: ${result.conversions}`)
    console.log(`  Conversion Rate: ${(result.conversionRate * 100).toFixed(2)}%`)
    console.log(`  Revenue: $${result.revenue.toFixed(2)}`)
    console.log(`  Revenue per User: $${result.revenuePerUser.toFixed(2)}`)
    console.log(`  Confidence: ${(result.confidence * 100).toFixed(1)}%\n`)
  }

  const test = statisticalTest(state, 'control', 'treatment')
  console.log('📈 Statistical Analysis:')
  console.log(`  Lift: ${test.lift.toFixed(2)}%`)
  console.log(`  P-Value: ${test.pValue.toFixed(4)}`)
  console.log(`  Significant: ${test.isSignificant ? '✅ Yes' : '❌ No'} (α = 0.05)`)
  console.log(
    `  95% CI: [${test.confidenceInterval[0].toFixed(2)}%, ${test.confidenceInterval[1].toFixed(2)}%]`
  )

  const winner = getWinner(state)
  console.log(
    `\n🏆 Winner: ${winner ? results.find((r) => r.variant.id === winner)?.variant.name : 'No clear winner yet'}`
  )

  const sampleSize = calculateSampleSize(0.12, 0.1)
  console.log(
    `\n📏 Sample Size Calculation: ${sampleSize} users per variant needed to detect 10% lift`
  )

  console.log('\n')
}

demoRecommendationSystem()
demoABTesting()

console.log('✨ E-commerce Demo Complete!\n')
console.log('💡 Key Takeaways:')
console.log('   1. Thompson Sampling automatically learns best products')
console.log(
  '   2. Balances exploration (trying new items) vs exploitation (showing proven winners)'
)
console.log('   3. A/B testing provides statistical rigor for decision making')
console.log('   4. Pure functional design = easy to test, debug, and reason about')
