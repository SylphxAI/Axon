/**
 * LSTM layer - Pure functional implementation
 * Long Short-Term Memory for sequence modeling
 */

import type { Tensor } from '@neuronline/tensor'
import { xavierNormal, zeros, matmul, add, mul } from '@neuronline/tensor'
import { sigmoid, tanh } from '@neuronline/functional'

/**
 * LSTM layer state
 */
export type LSTMState = {
  // Input gate
  readonly Wi: Tensor // [inputSize, hiddenSize]
  readonly Ui: Tensor // [hiddenSize, hiddenSize]
  readonly bi: Tensor // [hiddenSize]

  // Forget gate
  readonly Wf: Tensor // [inputSize, hiddenSize]
  readonly Uf: Tensor // [hiddenSize, hiddenSize]
  readonly bf: Tensor // [hiddenSize]

  // Cell gate
  readonly Wc: Tensor // [inputSize, hiddenSize]
  readonly Uc: Tensor // [hiddenSize, hiddenSize]
  readonly bc: Tensor // [hiddenSize]

  // Output gate
  readonly Wo: Tensor // [inputSize, hiddenSize]
  readonly Uo: Tensor // [hiddenSize, hiddenSize]
  readonly bo: Tensor // [hiddenSize]
}

/**
 * LSTM hidden state (carried across time steps)
 */
export type LSTMHidden = {
  readonly h: Tensor // Hidden state [batch, hiddenSize]
  readonly c: Tensor // Cell state [batch, hiddenSize]
}

/**
 * Initialize LSTM layer
 */
export function init(inputSize: number, hiddenSize: number): LSTMState {
  return {
    // Input gate
    Wi: xavierNormal([inputSize, hiddenSize], { requiresGrad: true }),
    Ui: xavierNormal([hiddenSize, hiddenSize], { requiresGrad: true }),
    bi: zeros([hiddenSize], { requiresGrad: true }),

    // Forget gate
    Wf: xavierNormal([inputSize, hiddenSize], { requiresGrad: true }),
    Uf: xavierNormal([hiddenSize, hiddenSize], { requiresGrad: true }),
    bf: zeros([hiddenSize], { requiresGrad: true }),

    // Cell gate
    Wc: xavierNormal([inputSize, hiddenSize], { requiresGrad: true }),
    Uc: xavierNormal([hiddenSize, hiddenSize], { requiresGrad: true }),
    bc: zeros([hiddenSize], { requiresGrad: true }),

    // Output gate
    Wo: xavierNormal([inputSize, hiddenSize], { requiresGrad: true }),
    Uo: xavierNormal([hiddenSize, hiddenSize], { requiresGrad: true }),
    bo: zeros([hiddenSize], { requiresGrad: true }),
  }
}

/**
 * Initialize hidden state
 */
export function initHidden(batchSize: number, hiddenSize: number): LSTMHidden {
  return {
    h: zeros([batchSize, hiddenSize]),
    c: zeros([batchSize, hiddenSize]),
  }
}

/**
 * Forward pass through LSTM (single time step)
 *
 * x: [batch, inputSize]
 * hidden: { h: [batch, hiddenSize], c: [batch, hiddenSize] }
 *
 * Returns: { output: [batch, hiddenSize], hidden: LSTMHidden }
 */
export function forward(
  x: Tensor,
  hidden: LSTMHidden,
  state: LSTMState
): { output: Tensor; hidden: LSTMHidden } {
  const { h: prevH, c: prevC } = hidden

  // Input gate: i_t = sigmoid(W_i @ x_t + U_i @ h_{t-1} + b_i)
  const iGate = sigmoid(
    add(add(matmul(x, state.Wi), matmul(prevH, state.Ui)), state.bi)
  )

  // Forget gate: f_t = sigmoid(W_f @ x_t + U_f @ h_{t-1} + b_f)
  const fGate = sigmoid(
    add(add(matmul(x, state.Wf), matmul(prevH, state.Uf)), state.bf)
  )

  // Cell gate: c̃_t = tanh(W_c @ x_t + U_c @ h_{t-1} + b_c)
  const cGate = tanh(
    add(add(matmul(x, state.Wc), matmul(prevH, state.Uc)), state.bc)
  )

  // Cell state: c_t = f_t * c_{t-1} + i_t * c̃_t
  const newC = add(mul(fGate, prevC), mul(iGate, cGate))

  // Output gate: o_t = sigmoid(W_o @ x_t + U_o @ h_{t-1} + b_o)
  const oGate = sigmoid(
    add(add(matmul(x, state.Wo), matmul(prevH, state.Uo)), state.bo)
  )

  // Hidden state: h_t = o_t * tanh(c_t)
  const newH = mul(oGate, tanh(newC))

  return {
    output: newH,
    hidden: { h: newH, c: newC },
  }
}

/**
 * Process sequence through LSTM
 *
 * sequence: [seqLen, batch, inputSize]
 * Returns: [seqLen, batch, hiddenSize]
 */
export function forwardSequence(
  sequence: Tensor[],
  initialHidden: LSTMHidden,
  state: LSTMState
): { outputs: Tensor[]; finalHidden: LSTMHidden } {
  const outputs: Tensor[] = []
  let hidden = initialHidden

  for (const x of sequence) {
    const result = forward(x, hidden, state)
    outputs.push(result.output)
    hidden = result.hidden
  }

  return { outputs, finalHidden: hidden }
}
