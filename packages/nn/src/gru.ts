/**
 * GRU layer - Pure functional implementation
 * Gated Recurrent Unit for sequence modeling
 * Simpler than LSTM with fewer parameters
 */

import type { Tensor } from '@neuronline/tensor'
import { xavierNormal, zeros, matmul, add, mul, sub, ones } from '@neuronline/tensor'
import { sigmoid, tanh } from '@neuronline/functional'

/**
 * GRU layer state
 */
export type GRUState = {
  // Reset gate
  readonly Wr: Tensor // [inputSize, hiddenSize]
  readonly Ur: Tensor // [hiddenSize, hiddenSize]
  readonly br: Tensor // [hiddenSize]

  // Update gate
  readonly Wz: Tensor // [inputSize, hiddenSize]
  readonly Uz: Tensor // [hiddenSize, hiddenSize]
  readonly bz: Tensor // [hiddenSize]

  // Hidden candidate
  readonly Wh: Tensor // [inputSize, hiddenSize]
  readonly Uh: Tensor // [hiddenSize, hiddenSize]
  readonly bh: Tensor // [hiddenSize]
}

/**
 * GRU hidden state (carried across time steps)
 */
export type GRUHidden = {
  readonly h: Tensor // Hidden state [batch, hiddenSize]
}

/**
 * Initialize GRU layer
 */
export function init(inputSize: number, hiddenSize: number): GRUState {
  return {
    // Reset gate
    Wr: xavierNormal([inputSize, hiddenSize], { requiresGrad: true }),
    Ur: xavierNormal([hiddenSize, hiddenSize], { requiresGrad: true }),
    br: zeros([hiddenSize], { requiresGrad: true }),

    // Update gate
    Wz: xavierNormal([inputSize, hiddenSize], { requiresGrad: true }),
    Uz: xavierNormal([hiddenSize, hiddenSize], { requiresGrad: true }),
    bz: zeros([hiddenSize], { requiresGrad: true }),

    // Hidden candidate
    Wh: xavierNormal([inputSize, hiddenSize], { requiresGrad: true }),
    Uh: xavierNormal([hiddenSize, hiddenSize], { requiresGrad: true }),
    bh: zeros([hiddenSize], { requiresGrad: true }),
  }
}

/**
 * Initialize hidden state
 */
export function initHidden(batchSize: number, hiddenSize: number): GRUHidden {
  return {
    h: zeros([batchSize, hiddenSize]),
  }
}

/**
 * Forward pass through GRU (single time step)
 *
 * x: [batch, inputSize]
 * hidden: { h: [batch, hiddenSize] }
 *
 * Returns: { output: [batch, hiddenSize], hidden: GRUHidden }
 */
export function forward(
  x: Tensor,
  hidden: GRUHidden,
  state: GRUState
): { output: Tensor; hidden: GRUHidden } {
  const { h: prevH } = hidden

  // Reset gate: r_t = sigmoid(W_r @ x_t + U_r @ h_{t-1} + b_r)
  const rGate = sigmoid(
    add(add(matmul(x, state.Wr), matmul(prevH, state.Ur)), state.br)
  )

  // Update gate: z_t = sigmoid(W_z @ x_t + U_z @ h_{t-1} + b_z)
  const zGate = sigmoid(
    add(add(matmul(x, state.Wz), matmul(prevH, state.Uz)), state.bz)
  )

  // Candidate hidden: h_tilde = tanh(W_h @ x_t + U_h @ (r_t * h_{t-1}) + b_h)
  const hCandidate = tanh(
    add(
      add(matmul(x, state.Wh), matmul(mul(rGate, prevH), state.Uh)),
      state.bh
    )
  )

  // New hidden: h_t = z_t * h_{t-1} + (1 - z_t) * h_tilde
  // h_t = z_t * h_{t-1} + h_tilde - z_t * h_tilde
  const newH = add(
    mul(zGate, prevH),
    mul(sub(ones(zGate.shape), zGate), hCandidate)
  )

  return {
    output: newH,
    hidden: { h: newH },
  }
}

/**
 * Forward pass through sequence
 *
 * x: [batch, seqLen, inputSize]
 * hidden: { h: [batch, hiddenSize] }
 *
 * Returns: { outputs: [batch, seqLen, hiddenSize], hidden: GRUHidden }
 */
export function forwardSequence(
  x: Tensor,
  initialHidden: GRUHidden,
  state: GRUState
): { outputs: Tensor[]; hidden: GRUHidden } {
  // x shape: [batch, seqLen, inputSize]
  const [batch, seqLen, inputSize] = x.shape
  const outputs: Tensor[] = []
  let hidden = initialHidden

  // Process each time step
  for (let t = 0; t < seqLen!; t++) {
    // Extract time step: [batch, inputSize]
    const xt = {
      data: x.data.slice(t * batch! * inputSize!, (t + 1) * batch! * inputSize!),
      shape: [batch!, inputSize!] as [number, number],
      requiresGrad: x.requiresGrad,
    }

    const result = forward(xt, hidden, state)
    outputs.push(result.output)
    hidden = result.hidden
  }

  return {
    outputs,
    hidden,
  }
}
