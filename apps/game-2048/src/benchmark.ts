/**
 * Benchmark training performance
 * Track speed metrics for optimization
 */

import * as game from './game'
import * as agent from './agent'
import { loadAcceleration } from '@sylphx/tensor'

console.log('⚡ NeuronLine Performance Benchmark\n')

// Load WASM acceleration
console.log('Loading WASM acceleration...')
const wasmLoaded = await loadAcceleration()
console.log(`WASM acceleration: ${wasmLoaded ? 'ENABLED ⚡' : 'DISABLED'}\n`)

// Benchmark config
const EPISODES = 100 // Fast benchmark
const BATCH_SIZE = 32
const REPLAY_BUFFER_SIZE = 1000

// Performance metrics
const metrics = {
  startTime: 0,
  endTime: 0,
  totalTime: 0,
  episodesPerSecond: 0,
  avgEpisodeTime: 0,
  totalSteps: 0,
  stepsPerSecond: 0,
  totalTrainingTime: 0,
  trainingSteps: 0,
}

// Replay buffer
const replayBuffer: agent.Experience[] = []

// Initialize agent
let agentState = agent.initAgent()

// Training loop
metrics.startTime = performance.now()
let totalSteps = 0
let trainingSteps = 0
const trainingTimeStart = performance.now()

for (let episode = 0; episode < EPISODES; episode++) {
  let gameState = game.init()

  // Play one episode
  while (!gameState.gameOver) {
    const currentState = game.gridToArray(gameState.grid)
    const availableActions = game.getAvailableActions(gameState)
    if (availableActions.length === 0) break

    const { action, actionIndex } = agent.selectAction(
      currentState,
      agentState,
      availableActions
    )

    const newGameState = game.move(gameState, action)
    const reward = newGameState.score - gameState.score

    const experience: agent.Experience = {
      state: currentState,
      action: actionIndex,
      reward,
      nextState: game.gridToArray(newGameState.grid),
      done: newGameState.gameOver,
    }

    replayBuffer.push(experience)
    if (replayBuffer.length > REPLAY_BUFFER_SIZE) {
      replayBuffer.shift()
    }

    gameState = newGameState
    totalSteps++
  }

  // Train on batch
  if (replayBuffer.length >= BATCH_SIZE) {
    const trainStart = performance.now()
    const batch = sampleBatch(replayBuffer, BATCH_SIZE)
    agentState = agent.train(agentState, batch)
    metrics.totalTrainingTime += performance.now() - trainStart
    trainingSteps++
  }

  agentState = agent.decayEpsilon(agentState)
}

metrics.endTime = performance.now()
metrics.totalTime = metrics.endTime - metrics.startTime
metrics.episodesPerSecond = (EPISODES / metrics.totalTime) * 1000
metrics.avgEpisodeTime = metrics.totalTime / EPISODES
metrics.totalSteps = totalSteps
metrics.stepsPerSecond = (totalSteps / metrics.totalTime) * 1000

// Results
console.log('📊 Benchmark Results\n')
console.log('Training:')
console.log(`  Episodes: ${EPISODES}`)
console.log(`  Total Time: ${(metrics.totalTime / 1000).toFixed(2)}s`)
console.log(`  Episodes/sec: ${metrics.episodesPerSecond.toFixed(2)}`)
console.log(`  Avg Episode Time: ${metrics.avgEpisodeTime.toFixed(2)}ms`)

console.log('\nSteps:')
console.log(`  Total Steps: ${metrics.totalSteps}`)
console.log(`  Steps/sec: ${metrics.stepsPerSecond.toFixed(0)}`)

console.log('\nTraining:')
console.log(`  Training Steps: ${trainingSteps}`)
console.log(`  Training Time: ${(metrics.totalTrainingTime / 1000).toFixed(2)}s`)
console.log(`  Avg Training/step: ${(metrics.totalTrainingTime / trainingSteps).toFixed(2)}ms`)

console.log('\nMemory:')
const memUsage = process.memoryUsage()
console.log(`  Heap Used: ${(memUsage.heapUsed / 1024 / 1024).toFixed(2)} MB`)
console.log(`  Heap Total: ${(memUsage.heapTotal / 1024 / 1024).toFixed(2)} MB`)

// Save results
const results = {
  timestamp: new Date().toISOString(),
  version: '3.0.0-pure-functional',
  metrics: {
    episodes: EPISODES,
    totalTimeMs: metrics.totalTime,
    episodesPerSecond: metrics.episodesPerSecond,
    avgEpisodeTimeMs: metrics.avgEpisodeTime,
    totalSteps: metrics.totalSteps,
    stepsPerSecond: metrics.stepsPerSecond,
    trainingSteps: trainingSteps,
    trainingTimeMs: metrics.totalTrainingTime,
    avgTrainingPerStepMs: metrics.totalTrainingTime / trainingSteps,
    heapUsedMB: memUsage.heapUsed / 1024 / 1024,
    heapTotalMB: memUsage.heapTotal / 1024 / 1024,
  },
}

// Append to benchmark log
const fs = require('fs')
const logPath = './benchmark-log.jsonl'
fs.appendFileSync(logPath, JSON.stringify(results) + '\n')

console.log(`\n✅ Results saved to ${logPath}`)

function sampleBatch(buffer: agent.Experience[], size: number): agent.Experience[] {
  const batch: agent.Experience[] = []
  for (let i = 0; i < size; i++) {
    const idx = Math.floor(Math.random() * buffer.length)
    batch.push(buffer[idx]!)
  }
  return batch
}
