import { ClickPredictor, SequencePredictor } from '@sylphx/predictors'

console.log('🚀 NeuronLine Demo\n')

function demoClickPredictor() {
  console.log('📍 Click Prediction Demo')
  console.log('='.repeat(50))

  const predictor = new ClickPredictor({
    learningRate: 0.1,
    algorithm: 'ftrl',
  })

  const trainingData = [
    {
      context: {
        position: { x: 500, y: 300 },
        viewport: { width: 1920, height: 1080 },
        elementType: 'button',
        deviceType: 'desktop' as const,
        timeOnPage: 5000,
      },
      clicked: true,
    },
    {
      context: {
        position: { x: 1800, y: 50 },
        viewport: { width: 1920, height: 1080 },
        elementType: 'text',
        deviceType: 'desktop' as const,
        timeOnPage: 1000,
      },
      clicked: false,
    },
    {
      context: {
        position: { x: 960, y: 540 },
        viewport: { width: 1920, height: 1080 },
        elementType: 'button',
        deviceType: 'mobile' as const,
        timeOnPage: 10000,
      },
      clicked: true,
    },
  ]

  console.log('Training on user click patterns...')
  for (let epoch = 0; epoch < 100; epoch++) {
    for (const example of trainingData) {
      predictor.learn(example)
    }
  }

  console.log('\n✅ Training complete')
  console.log(`📊 Accuracy: ${(predictor.getAccuracy() * 100).toFixed(1)}%`)
  console.log(`📈 Total updates: ${predictor.getMetrics().updates}`)

  console.log('\n🔮 Predictions:')

  const testCases = [
    {
      name: 'Button center screen',
      context: {
        position: { x: 960, y: 540 },
        viewport: { width: 1920, height: 1080 },
        elementType: 'button',
        deviceType: 'desktop' as const,
      },
    },
    {
      name: 'Text top-right corner',
      context: {
        position: { x: 1800, y: 100 },
        viewport: { width: 1920, height: 1080 },
        elementType: 'text',
        deviceType: 'desktop' as const,
      },
    },
  ]

  for (const testCase of testCases) {
    const probability = predictor.predict(testCase.context)
    const willClick = predictor.willClick(testCase.context)
    console.log(
      `  ${testCase.name}: ${(probability * 100).toFixed(1)}% ${willClick ? '✓ Will click' : "✗ Won't click"}`
    )
  }

  console.log('\n')
}

function demoSequencePredictor() {
  console.log('🔄 Sequence Prediction Demo')
  console.log('='.repeat(50))

  const predictor = new SequencePredictor({
    learningRate: 0.05,
    maxSequenceLength: 5,
  })

  const commonPatterns = [
    { sequence: ['home', 'products', 'details', 'cart', 'checkout'], outcome: 1 },
    { sequence: ['home', 'products', 'details', 'back', 'exit'], outcome: 0 },
    { sequence: ['home', 'search', 'results', 'details', 'cart'], outcome: 1 },
    { sequence: ['home', 'about', 'contact', 'exit'], outcome: 0 },
  ]

  console.log('Training on navigation patterns...')
  for (let epoch = 0; epoch < 200; epoch++) {
    for (const pattern of commonPatterns) {
      predictor.learn(pattern.sequence, pattern.outcome)
    }
  }

  console.log('\n✅ Training complete')
  console.log(`📈 Total updates: ${predictor.getMetrics().updates}`)

  console.log('\n🔮 Predictions:')

  predictor.addAction('home')
  predictor.addAction('products')
  predictor.addAction('details')

  const possibleActions = ['cart', 'back', 'exit', 'search']
  const predictions = predictor.predictNext(possibleActions)

  console.log('  Current sequence: home → products → details')
  console.log('  Next action probabilities:')

  for (const [action, score] of Array.from(predictions.entries()).sort((a, b) => b[1] - a[1])) {
    console.log(`    ${action}: ${(score * 100).toFixed(1)}%`)
  }

  const mostLikely = predictor.getMostLikely(possibleActions)
  console.log(`\n  ⭐ Most likely: ${mostLikely}`)

  console.log('\n')
}

function demoBenchmark() {
  console.log('⚡ Performance Benchmark')
  console.log('='.repeat(50))

  const predictor = new ClickPredictor()

  const context = {
    position: { x: 500, y: 300 },
    viewport: { width: 1920, height: 1080 },
    elementType: 'button',
    deviceType: 'desktop' as const,
  }

  const iterations = 10000

  const predictionStart = performance.now()
  for (let i = 0; i < iterations; i++) {
    predictor.predict(context)
  }
  const predictionTime = performance.now() - predictionStart

  const learningStart = performance.now()
  for (let i = 0; i < iterations; i++) {
    predictor.learn({ context, clicked: i % 2 === 0 })
  }
  const learningTime = performance.now() - learningStart

  console.log(`Prediction: ${(predictionTime / iterations).toFixed(3)}ms per call`)
  console.log(`Learning: ${(learningTime / iterations).toFixed(3)}ms per update`)
  console.log(`Throughput: ${Math.floor(iterations / (predictionTime / 1000))} predictions/sec`)

  console.log('\n')
}

demoClickPredictor()
demoSequencePredictor()
demoBenchmark()

console.log('✨ Demo complete!')
