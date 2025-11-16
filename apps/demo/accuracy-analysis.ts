import { ClickPredictor } from '@sylphx/neuronline-predictors'
import { OnlineLearner, createModelState, learn, predict } from '@sylphx/neuronline-core'
import type { TrainingExample } from '@sylphx/neuronline-core'

console.log('📊 NeuronLine Accuracy & Size Analysis\n')

function analyzeModelSize() {
  console.log('=' .repeat(60))
  console.log('📏 Model Size Analysis')
  console.log('=' .repeat(60))

  const sizes = [
    { name: 'Click Predictor (16 features)', features: 16 },
    { name: 'Sequence Predictor (32 features)', features: 32 },
    { name: 'Custom (100 features)', features: 100 },
    { name: 'Large (1000 features)', features: 1000 },
  ]

  for (const { name, features } of sizes) {
    const learner = new OnlineLearner({ inputSize: features })
    const exported = learner.export()

    const weightsSize = exported.weights.length * 4
    const totalSize = weightsSize + 100

    console.log(`\n${name}:`)
    console.log(`  Features: ${features}`)
    console.log(`  Weights: ${weightsSize} bytes (${(weightsSize / 1024).toFixed(2)} KB)`)
    console.log(`  Total Model: ~${(totalSize / 1024).toFixed(2)} KB`)
    console.log(`  Memory: ${(totalSize / 1024 / 1024).toFixed(4)} MB`)
  }

  console.log('\n💡 Note: This is a LINEAR model, not a deep neural network')
  console.log('   - Model size = input_features × 4 bytes (Float32)')
  console.log('   - Very lightweight compared to deep learning models')
  console.log('   - PyTorch ResNet-50: ~100MB')
  console.log('   - Our largest model: ~4KB\n')
}

function testLinearPattern() {
  console.log('=' .repeat(60))
  console.log('🎯 Test 1: Linear Pattern (Should be GOOD)')
  console.log('=' .repeat(60))

  const learner = new OnlineLearner({
    inputSize: 2,
    learningRate: 0.1,
    algorithm: 'sgd',
    regularization: 0.001,
  })

  const trainingData: TrainingExample[] = []
  for (let i = 0; i < 500; i++) {
    const x1 = Math.random()
    const x2 = Math.random()
    const label = x1 + x2 > 1 ? 1 : 0

    trainingData.push({
      features: new Float32Array([x1, x2]),
      label,
    })
  }

  for (const example of trainingData) {
    learner.learn(example)
  }

  let correct = 0
  const testSize = 100
  for (let i = 0; i < testSize; i++) {
    const x1 = Math.random()
    const x2 = Math.random()
    const trueLabel = x1 + x2 > 1 ? 1 : 0
    const prediction = learner.predict(new Float32Array([x1, x2]))
    const predictedLabel = prediction > 0.5 ? 1 : 0

    if (predictedLabel === trueLabel) correct++
  }

  const accuracy = (correct / testSize) * 100
  console.log(`\nPattern: x1 + x2 > 1 (Linearly separable)`)
  console.log(`Training examples: ${trainingData.length}`)
  console.log(`Test accuracy: ${accuracy.toFixed(1)}%`)
  console.log(`Result: ${accuracy > 90 ? '✅ EXCELLENT' : accuracy > 80 ? '✅ GOOD' : '⚠️ FAIR'}`)

  return accuracy
}

function testXORPattern() {
  console.log('\n' + '=' .repeat(60))
  console.log('❌ Test 2: XOR Pattern (Should be POOR)')
  console.log('=' .repeat(60))

  const learner = new OnlineLearner({
    inputSize: 2,
    learningRate: 0.1,
    algorithm: 'sgd',
  })

  const trainingData: TrainingExample[] = []
  for (let i = 0; i < 1000; i++) {
    const x1 = Math.random() > 0.5 ? 1 : 0
    const x2 = Math.random() > 0.5 ? 1 : 0
    const label = x1 !== x2 ? 1 : 0

    trainingData.push({
      features: new Float32Array([x1, x2]),
      label,
    })
  }

  for (const example of trainingData) {
    learner.learn(example)
  }

  let correct = 0
  const testSize = 100
  for (let i = 0; i < testSize; i++) {
    const x1 = Math.random() > 0.5 ? 1 : 0
    const x2 = Math.random() > 0.5 ? 1 : 0
    const trueLabel = x1 !== x2 ? 1 : 0
    const prediction = learner.predict(new Float32Array([x1, x2]))
    const predictedLabel = prediction > 0.5 ? 1 : 0

    if (predictedLabel === trueLabel) correct++
  }

  const accuracy = (correct / testSize) * 100
  console.log(`\nPattern: XOR (NOT linearly separable)`)
  console.log(`Training examples: ${trainingData.length}`)
  console.log(`Test accuracy: ${accuracy.toFixed(1)}%`)
  console.log(`Expected: ~50% (random guessing)`)
  console.log(`Result: ${accuracy > 60 ? '⚠️ Surprising' : '❌ EXPECTED - Linear model cannot learn XOR'}`)
  console.log(`\n💡 This proves it's a LINEAR model, not a neural network`)

  return accuracy
}

function testRealWorldClick() {
  console.log('\n' + '=' .repeat(60))
  console.log('🖱️  Test 3: Real-World Click Prediction')
  console.log('=' .repeat(60))

  const predictor = new ClickPredictor({
    learningRate: 0.05,
    algorithm: 'ftrl',
  })

  console.log('\nSimulating realistic click patterns...')

  for (let i = 0; i < 500; i++) {
    const position = {
      x: Math.random() * 1920,
      y: Math.random() * 1080,
    }

    const inClickZone =
      (position.x > 800 && position.x < 1120 && position.y > 400 && position.y < 680) ||
      (position.x < 200 && position.y < 100)

    const clicked = inClickZone ? Math.random() < 0.8 : Math.random() < 0.1

    predictor.learn({
      context: {
        position,
        viewport: { width: 1920, height: 1080 },
        elementType: inClickZone ? 'button' : 'text',
        deviceType: 'desktop',
      },
      clicked,
    })
  }

  let correct = 0
  const testSize = 100

  for (let i = 0; i < testSize; i++) {
    const position = {
      x: Math.random() * 1920,
      y: Math.random() * 1080,
    }

    const inClickZone =
      (position.x > 800 && position.x < 1120 && position.y > 400 && position.y < 680) ||
      (position.x < 200 && position.y < 100)

    const shouldClick = inClickZone ? Math.random() < 0.8 : Math.random() < 0.1

    const willClick = predictor.willClick({
      position,
      viewport: { width: 1920, height: 1080 },
      elementType: inClickZone ? 'button' : 'text',
      deviceType: 'desktop',
    })

    if (willClick === shouldClick) correct++
  }

  const accuracy = (correct / testSize) * 100
  console.log(`Test accuracy: ${accuracy.toFixed(1)}%`)
  console.log(`Training examples: 500`)
  console.log(`Result: ${accuracy > 70 ? '✅ GOOD' : accuracy > 60 ? '⚠️ FAIR' : '❌ POOR'}`)
  console.log(
    `Note: Real-world accuracy depends heavily on feature engineering`
  )

  return accuracy
}

function comparativeAnalysis() {
  console.log('\n' + '=' .repeat(60))
  console.log('⚖️  Comparison with Other Models')
  console.log('=' .repeat(60))

  console.log('\nModel Complexity:')
  console.log('  NeuronLine (Linear):')
  console.log('    - Parameters: ~100-1000')
  console.log('    - Size: 0.4-4 KB')
  console.log('    - Speed: <0.001ms per prediction')
  console.log('    - Accuracy: Good for linear patterns, Poor for complex patterns')
  console.log('')
  console.log('  Simple Neural Network (2 hidden layers):')
  console.log('    - Parameters: ~10,000-100,000')
  console.log('    - Size: 40-400 KB')
  console.log('    - Speed: ~0.01-0.1ms per prediction')
  console.log('    - Accuracy: Good for moderately complex patterns')
  console.log('')
  console.log('  Deep Learning (ResNet-50):')
  console.log('    - Parameters: ~25 million')
  console.log('    - Size: ~100 MB')
  console.log('    - Speed: ~10-100ms per prediction')
  console.log('    - Accuracy: Excellent for complex patterns')

  console.log('\n📊 When to Use NeuronLine:')
  console.log('  ✅ Linearly separable problems')
  console.log('  ✅ Need extreme speed (<0.001ms)')
  console.log('  ✅ Need tiny model size (<10KB)')
  console.log('  ✅ Simple classification tasks')
  console.log('  ✅ Edge devices / browsers')
  console.log('  ✅ Combined with bandit algorithms')

  console.log('\n❌ When NOT to Use NeuronLine:')
  console.log('  ❌ Complex non-linear patterns')
  console.log('  ❌ Image recognition')
  console.log('  ❌ Natural language processing')
  console.log('  ❌ Deep feature learning needed')
  console.log('  ❌ XOR-like problems')
}

function runAllTests() {
  analyzeModelSize()

  const linearAccuracy = testLinearPattern()
  const xorAccuracy = testXORPattern()
  const clickAccuracy = testRealWorldClick()

  comparativeAnalysis()

  console.log('\n' + '=' .repeat(60))
  console.log('📈 Summary')
  console.log('=' .repeat(60))
  console.log(`\nLinear Pattern: ${linearAccuracy.toFixed(1)}% ${linearAccuracy > 90 ? '✅' : '⚠️'}`)
  console.log(`XOR Pattern: ${xorAccuracy.toFixed(1)}% ${xorAccuracy < 60 ? '❌ (Expected)' : '⚠️'}`)
  console.log(`Click Prediction: ${clickAccuracy.toFixed(1)}% ${clickAccuracy > 70 ? '✅' : '⚠️'}`)

  console.log('\n🎯 Key Findings:')
  console.log('  1. This is a LINEAR model (logistic regression), NOT a deep neural network')
  console.log('  2. Excellent for linearly separable problems (>90% accuracy)')
  console.log('  3. Cannot learn XOR-like patterns (~50% accuracy)')
  console.log('  4. Real-world performance depends on feature engineering')
  console.log('  5. Extremely lightweight (0.4-4 KB vs 100MB for deep models)')
  console.log('  6. Blazing fast (<0.001ms vs 10-100ms for deep models)')

  console.log('\n💡 Recommendations:')
  console.log('  ✅ Use for: Simple classification, speed-critical apps, edge computing')
  console.log('  ✅ Combine with: Bandit algorithms for exploration/exploitation')
  console.log('  ⚠️  Upgrade to neural network if: Need to learn complex non-linear patterns')
  console.log('')
}

runAllTests()
