/**
 * DQN Agent for 2048
 * Pure functional implementation
 */

import * as T from '../../../packages/tensor/src/index'
import * as F from '../../../packages/functional/src/index'
import * as nn from '../../../packages/nn/src/index'
import * as optim from '../../../packages/optim/src/index'
import type { GameState, Direction } from './game'
import { gridToArray } from './game'

/**
 * Neural network model for Q-learning
 * Input: 16 (flattened 4x4 grid)
 * Output: 4 (Q-values for up, down, left, right)
 */
export type QNetwork = {
  linear1: nn.LinearState
  linear2: nn.LinearState
  linear3: nn.LinearState
}

/**
 * Agent state
 */
export type AgentState = {
  network: QNetwork
  optimizer: optim.OptimizerState
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
 * Initialize Q-network
 */
export function initNetwork(): QNetwork {
  return {
    linear1: nn.linear.init(16, 64),
    linear2: nn.linear.init(64, 64),
    linear3: nn.linear.init(64, 4),
  }
}

/**
 * Forward pass through network
 */
export function forward(state: number[], network: QNetwork): T.Tensor {
  const input = T.tensor([state], { requiresGrad: false })
  let h = nn.linear.forward(input, network.linear1)
  h = F.relu(h)
  h = nn.linear.forward(h, network.linear2)
  h = F.relu(h)
  h = nn.linear.forward(h, network.linear3)
  return h
}

/**
 * Initialize agent
 */
export function initAgent(): AgentState {
  const network = initNetwork()
  const params = getNetworkParams(network)
  const optimizer = optim.adam.init(params, { lr: 0.001 })

  return {
    network,
    optimizer,
    epsilon: 1.0, // Start with full exploration
    totalReward: 0,
    gamesPlayed: 0,
  }
}

/**
 * Get all network parameters
 */
function getNetworkParams(network: QNetwork): T.Tensor[] {
  return [
    network.linear1.weight,
    network.linear1.bias,
    network.linear2.weight,
    network.linear2.bias,
    network.linear3.weight,
    network.linear3.bias,
  ]
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
  const qValues = forward(state, agent.network)
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

  let totalLoss = 0

  // Process each experience
  for (const exp of experiences) {
    // Current Q-values
    const qValues = forward(exp.state, agent.network)

    // Target Q-value for the action taken
    let targetQ: number
    if (exp.done) {
      targetQ = exp.reward
    } else {
      const nextQValues = forward(exp.nextState, agent.network)
      const nextQArray = T.toArray(nextQValues)[0] as number[]
      const maxNextQ = Math.max(...nextQArray)
      targetQ = exp.reward + gamma * maxNextQ
    }

    // Create target tensor (same as qValues but with updated action)
    const qArray = T.toArray(qValues)[0] as number[]
    qArray[exp.action] = targetQ
    const target = T.tensor([qArray], { requiresGrad: false })

    // Compute loss
    const loss = F.mse(qValues, target)
    totalLoss += T.item(loss)

    // Backward pass
    const grads = T.backward(loss)

    // Update network
    const result = optim.adam.step(agent.optimizer, getNetworkParams(agent.network), grads)

    // Rebuild network with new parameters
    const newNetwork: QNetwork = {
      linear1: {
        weight: result.params[0]!,
        bias: result.params[1]!,
      },
      linear2: {
        weight: result.params[2]!,
        bias: result.params[3]!,
      },
      linear3: {
        weight: result.params[4]!,
        bias: result.params[5]!,
      },
    }

    // Update agent (this will be overwritten in loop, but that's fine for batch training)
    agent = {
      ...agent,
      network: newNetwork,
      optimizer: result.state,
    }
  }

  return agent
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
