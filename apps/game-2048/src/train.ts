/**
 * Train DQN agent to play 2048
 */

import * as game from './game'
import * as agent from './agent'

console.log('🤖 Training DQN Agent for 2048\n')

// Training config
const NUM_EPISODES = 1000
const BATCH_SIZE = 32
const REPLAY_BUFFER_SIZE = 10000
const TRAIN_FREQUENCY = 4
const PRINT_FREQUENCY = 50

// Replay buffer
const replayBuffer: agent.Experience[] = []

// Initialize agent
let agentState = agent.initAgent()

// Training stats
const stats = {
  maxTile: 0,
  maxScore: 0,
  avgScore: 0,
  scores: [] as number[],
}

// Training loop
for (let episode = 0; episode < NUM_EPISODES; episode++) {
  let gameState = game.init()
  const episodeExperiences: agent.Experience[] = []

  // Play one episode
  while (!gameState.gameOver) {
    const currentState = game.gridToArray(gameState.grid)
    const availableActions = game.getAvailableActions(gameState)

    if (availableActions.length === 0) break

    // Select action
    const { action, actionIndex } = agent.selectAction(
      currentState,
      agentState,
      availableActions
    )

    // Take action
    const newGameState = game.move(gameState, action)

    // Calculate reward
    const reward = calculateReward(gameState, newGameState)

    // Store experience
    const experience: agent.Experience = {
      state: currentState,
      action: actionIndex,
      reward,
      nextState: game.gridToArray(newGameState.grid),
      done: newGameState.gameOver,
    }

    episodeExperiences.push(experience)

    // Add to replay buffer
    replayBuffer.push(experience)
    if (replayBuffer.length > REPLAY_BUFFER_SIZE) {
      replayBuffer.shift()
    }

    gameState = newGameState
  }

  // Train on batch from replay buffer
  if (episode % TRAIN_FREQUENCY === 0 && replayBuffer.length >= BATCH_SIZE) {
    const batch = sampleBatch(replayBuffer, BATCH_SIZE)
    agentState = agent.train(agentState, batch)
  }

  // Decay epsilon
  agentState = agent.decayEpsilon(agentState)

  // Update stats
  const finalScore = gameState.score
  const maxTile = game.getMaxTile(gameState)

  stats.scores.push(finalScore)
  if (finalScore > stats.maxScore) stats.maxScore = finalScore
  if (maxTile > stats.maxTile) stats.maxTile = maxTile

  // Calculate moving average
  const window = Math.min(100, stats.scores.length)
  stats.avgScore =
    stats.scores.slice(-window).reduce((a, b) => a + b, 0) / window

  // Print progress
  if ((episode + 1) % PRINT_FREQUENCY === 0) {
    console.log(
      `Episode ${episode + 1}/${NUM_EPISODES} | ` +
        `Avg Score: ${stats.avgScore.toFixed(0)} | ` +
        `Max Score: ${stats.maxScore} | ` +
        `Max Tile: ${stats.maxTile} | ` +
        `ε: ${agentState.epsilon.toFixed(3)}`
    )
  }
}

// Final stats
console.log('\n📊 Training Complete!')
console.log(`  Max Score: ${stats.maxScore}`)
console.log(`  Max Tile: ${stats.maxTile}`)
console.log(`  Avg Score (last 100): ${stats.avgScore.toFixed(0)}`)

// Test final policy (no exploration)
console.log('\n🎮 Testing learned policy (no exploration)...\n')
agentState = { ...agentState, epsilon: 0 }

const testGames = 10
const testScores: number[] = []
const testTiles: number[] = []

for (let i = 0; i < testGames; i++) {
  let gameState = game.init()

  while (!gameState.gameOver) {
    const currentState = game.gridToArray(gameState.grid)
    const availableActions = game.getAvailableActions(gameState)

    if (availableActions.length === 0) break

    const { action } = agent.selectAction(currentState, agentState, availableActions)
    gameState = game.move(gameState, action)
  }

  testScores.push(gameState.score)
  testTiles.push(game.getMaxTile(gameState))

  console.log(`  Game ${i + 1}: Score ${gameState.score}, Max Tile ${game.getMaxTile(gameState)}`)
}

const avgTestScore = testScores.reduce((a, b) => a + b, 0) / testScores.length
const maxTestTile = Math.max(...testTiles)

console.log(`\n📈 Test Results:`)
console.log(`  Average Score: ${avgTestScore.toFixed(0)}`)
console.log(`  Best Score: ${Math.max(...testScores)}`)
console.log(`  Max Tile Reached: ${maxTestTile}`)

/**
 * Calculate reward for transition
 */
function calculateReward(oldState: game.GameState, newState: game.GameState): number {
  // Reward based on score difference
  const scoreDiff = newState.score - oldState.score

  // Small penalty for each move to encourage efficiency
  let reward = scoreDiff > 0 ? scoreDiff : -0.1

  // Bonus for achieving higher tiles
  const oldMaxTile = game.getMaxTile(oldState)
  const newMaxTile = game.getMaxTile(newState)
  if (newMaxTile > oldMaxTile) {
    reward += newMaxTile
  }

  // Large penalty for game over
  if (newState.gameOver) {
    reward -= 100
  }

  return reward
}

/**
 * Sample random batch from replay buffer
 */
function sampleBatch(buffer: agent.Experience[], size: number): agent.Experience[] {
  const batch: agent.Experience[] = []
  for (let i = 0; i < size; i++) {
    const idx = Math.floor(Math.random() * buffer.length)
    batch.push(buffer[idx]!)
  }
  return batch
}
