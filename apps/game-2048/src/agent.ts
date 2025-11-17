/**
 * DQN Agent for 2048
 * Pure functional implementation with new API
 */

import * as T from '../../../packages/tensor/src/index'
import * as F from '../../../packages/functional/src/index'
import { Linear, ReLU, Sequential } from '../../../packages/nn/src/index'
import { Adam } from '../../../packages/optim/src/index'
import { getParams, trainStep } from '../../../packages/train/src/index'
import type { GameState, Direction } from './game'
import { gridToArray } from './game'

/**
 * Q-Network model
 * Input: 16 (flattened 4x4 grid)
 * Output: 4 (Q-values for up, down, left, right)
 */
const createQNetwork = () => Sequential(
  Linear(16, 64),
  ReLU(),
  Linear(64, 64),
  ReLU(),
  Linear(64, 4)
)

/**
 * Agent state
 */
export type AgentState = {
  model: ReturnType<typeof createQNetwork>
  modelState: any
  optimizer: ReturnType<typeof Adam>
  optState: any
  epsilon: number // Exploration rate
  totalReward: number
  gamesPlayed: number
}

/**
 * Experience tuple for replay buffer
 */
export type Experience = {
  state: number[]
  action: number
  reward: number
  nextState: number[]
  done: boolean
}

/**
 * Initialize agent
 */
export function initAgent(): AgentState {
  const model = createQNetwork()
  const modelState = model.init()
  const optimizer = Adam({ lr: 0.001 })
  const optState = optimizer.init(getParams(modelState))

  return {
    model,
    modelState,
    optimizer,
    optState,
    epsilon: 1.0, // Start with full exploration
    totalReward: 0,
    gamesPlayed: 0,
  }
}

/**
 * Forward pass through network
 */
function forward(state: number[], agent: AgentState): T.Tensor {
  const input = T.tensor([state], { requiresGrad: false })
  return agent.model.forward(input, agent.modelState)
}

/**
 * Select action using epsilon-greedy policy
 */
export function selectAction(
  state: number[],
  agent: AgentState,
  availableActions: Direction[]
): { action: Direction; actionIndex: number } {
  // Epsilon-greedy: explore or exploit
  if (Math.random() < agent.epsilon) {
    // Explore: random action
    const actionIndex = Math.floor(Math.random() * availableActions.length)
    return { action: availableActions[actionIndex]!, actionIndex }
  }

  // Exploit: best action according to Q-network
  const qValues = forward(state, agent)
  const qArray = T.toArray(qValues)[0] as number[]

  // Map directions to indices
  const directionIndices = availableActions.map((dir) => directionToIndex(dir))

  // Find best available action
  let bestIndex = directionIndices[0]!
  let bestQValue = qArray[bestIndex]!

  for (const idx of directionIndices) {
    if (qArray[idx]! > bestQValue) {
      bestQValue = qArray[idx]!
      bestIndex = idx
    }
  }

  return { action: indexToDirection(bestIndex), actionIndex: bestIndex }
}

/**
 * Train on a batch of experiences
 */
export function train(
  agent: AgentState,
  experiences: Experience[],
  gamma: number = 0.99
): AgentState {
  if (experiences.length === 0) return agent

  const batchSize = experiences.length

  // Prepare batched states
  const states: number[][] = []
  const nextStates: number[][] = []
  const actions: number[] = []
  const rewards: number[] = []
  const dones: boolean[] = []

  for (const exp of experiences) {
    states.push(exp.state)
    nextStates.push(exp.nextState)
    actions.push(exp.action)
    rewards.push(exp.reward)
    dones.push(exp.done)
  }

  // Forward pass for current states
  const statesTensor = T.tensor(states, { requiresGrad: true })
  const qValuesBatch = agent.model.forward(statesTensor, agent.modelState)

  // Forward pass for next states (no grad)
  const nextStatesTensor = T.tensor(nextStates, { requiresGrad: false })
  const nextQValuesBatch = agent.model.forward(nextStatesTensor, agent.modelState)
  const nextQArray = T.toArray(nextQValuesBatch)

  // Build target tensor
  const qArray = T.toArray(qValuesBatch)
  const targetArray: number[][] = []

  for (let i = 0; i < batchSize; i++) {
    const qValues = qArray[i]!
    const targetQValues = [...qValues]

    // Compute target for this experience
    if (dones[i]) {
      targetQValues[actions[i]!] = rewards[i]!
    } else {
      const nextQValues = nextQArray[i]!
      const maxNextQ = Math.max(...nextQValues)
      targetQValues[actions[i]!] = rewards[i]! + gamma * maxNextQ
    }

    targetArray.push(targetQValues)
  }

  const target = T.tensor(targetArray, { requiresGrad: false })

  // Train step using new API
  const result = trainStep({
    model: agent.model,
    modelState: agent.modelState,
    optimizer: agent.optimizer,
    optState: agent.optState,
    input: statesTensor,
    target: target,
    lossFn: F.mse
  })

  return {
    ...agent,
    modelState: result.modelState,
    optState: result.optState,
  }
}

/**
 * Decay epsilon (reduce exploration over time)
 */
export function decayEpsilon(agent: AgentState, minEpsilon: number = 0.1, decay: number = 0.995): AgentState {
  return {
    ...agent,
    epsilon: Math.max(minEpsilon, agent.epsilon * decay),
  }
}

/**
 * Direction to index mapping
 */
function directionToIndex(dir: Direction): number {
  switch (dir) {
    case 'up':
      return 0
    case 'down':
      return 1
    case 'left':
      return 2
    case 'right':
      return 3
  }
}

/**
 * Index to direction mapping
 */
function indexToDirection(idx: number): Direction {
  switch (idx) {
    case 0:
      return 'up'
    case 1:
      return 'down'
    case 2:
      return 'left'
    case 3:
      return 'right'
    default:
      return 'up'
  }
}
