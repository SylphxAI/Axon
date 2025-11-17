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
 * Train on a batch of experiences (batched version - uses WASM for large batches)
 * Processes all experiences together for better performance
 */
export function train(
  agent: AgentState,
  experiences: Experience[],
  gamma: number = 0.99
): AgentState {
  if (experiences.length === 0) return agent

  const batchSize = experiences.length

  // Prepare batched states: [batchSize, stateSize]
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

  // Batched forward pass: [batchSize, 16] → [batchSize, 4]
  // For batch size 32: [32, 16] @ [16, 64] → [32, 64] (2048 elements) - WASM activates!
  const statesTensor = T.tensor(states, { requiresGrad: true })
  const qValuesBatch = forwardBatch(statesTensor, agent.network)

  // Compute target Q-values
  const nextStatesTensor = T.tensor(nextStates, { requiresGrad: false })
  const nextQValuesBatch = forwardBatch(nextStatesTensor, agent.network)
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

  // Compute loss for entire batch
  const loss = F.mse(qValuesBatch, target)

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

  return {
    ...agent,
    network: newNetwork,
    optimizer: result.state,
  }
}

/**
 * Batched forward pass through network
 * Input: [batchSize, inputSize]
 * Output: [batchSize, outputSize]
 */
function forwardBatch(stateBatch: T.Tensor, network: QNetwork): T.Tensor {
  let h = nn.linear.forward(stateBatch, network.linear1)
  h = F.relu(h)
  h = nn.linear.forward(h, network.linear2)
  h = F.relu(h)
  h = nn.linear.forward(h, network.linear3)
  return h
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
