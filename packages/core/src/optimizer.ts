/**
 * Optimizers for Neural Network Training
 */

export type OptimizerConfig = {
  learningRate: number
  clipValue?: number
}

export type OptimizerState = {
  t: number // iteration counter
  [key: string]: unknown
}

export interface Optimizer {
  update(
    weights: Float32Array,
    gradients: Float32Array,
    state: OptimizerState
  ): { weights: Float32Array; state: OptimizerState }
}

/**
 * Stochastic Gradient Descent (SGD)
 * w = w - lr * gradient
 */
export class SGDOptimizer implements Optimizer {
  constructor(private config: OptimizerConfig) {}

  update(
    weights: Float32Array,
    gradients: Float32Array,
    state: OptimizerState
  ): { weights: Float32Array; state: OptimizerState } {
    const newWeights = new Float32Array(weights.length)
    const clipValue = this.config.clipValue ?? Number.POSITIVE_INFINITY

    for (let i = 0; i < weights.length; i++) {
      // Gradient clipping
      let grad = gradients[i]!
      grad = Math.max(-clipValue, Math.min(clipValue, grad))

      newWeights[i] = weights[i]! - this.config.learningRate * grad
    }

    return {
      weights: newWeights,
      state: { ...state, t: state.t + 1 },
    }
  }
}

/**
 * SGD with Momentum
 * v = beta * v + gradient
 * w = w - lr * v
 */
export class MomentumOptimizer implements Optimizer {
  constructor(
    private config: OptimizerConfig & { momentum?: number }
  ) {}

  update(
    weights: Float32Array,
    gradients: Float32Array,
    state: OptimizerState
  ): { weights: Float32Array; state: OptimizerState } {
    const beta = this.config.momentum ?? 0.9
    const clipValue = this.config.clipValue ?? Number.POSITIVE_INFINITY

    // Initialize velocity if first iteration
    let velocity = (state.velocity as Float32Array) ?? new Float32Array(weights.length)

    const newWeights = new Float32Array(weights.length)
    const newVelocity = new Float32Array(weights.length)

    for (let i = 0; i < weights.length; i++) {
      // Gradient clipping
      let grad = gradients[i]!
      grad = Math.max(-clipValue, Math.min(clipValue, grad))

      // Update velocity
      newVelocity[i] = beta * velocity[i]! + grad

      // Update weights
      newWeights[i] = weights[i]! - this.config.learningRate * newVelocity[i]!
    }

    return {
      weights: newWeights,
      state: { ...state, t: state.t + 1, velocity: newVelocity },
    }
  }
}

/**
 * Adam Optimizer (Adaptive Moment Estimation)
 * Combines momentum and RMSprop
 */
export class AdamOptimizer implements Optimizer {
  constructor(
    private config: OptimizerConfig & {
      beta1?: number
      beta2?: number
      epsilon?: number
    }
  ) {}

  update(
    weights: Float32Array,
    gradients: Float32Array,
    state: OptimizerState
  ): { weights: Float32Array; state: OptimizerState } {
    const beta1 = this.config.beta1 ?? 0.9
    const beta2 = this.config.beta2 ?? 0.999
    const epsilon = this.config.epsilon ?? 1e-8
    const clipValue = this.config.clipValue ?? Number.POSITIVE_INFINITY

    // Initialize moments if first iteration
    let m = (state.m as Float32Array) ?? new Float32Array(weights.length)
    let v = (state.v as Float32Array) ?? new Float32Array(weights.length)
    const t = state.t + 1

    const newWeights = new Float32Array(weights.length)
    const newM = new Float32Array(weights.length)
    const newV = new Float32Array(weights.length)

    for (let i = 0; i < weights.length; i++) {
      // Gradient clipping
      let grad = gradients[i]!
      grad = Math.max(-clipValue, Math.min(clipValue, grad))

      // Update biased first moment estimate
      newM[i] = beta1 * m[i]! + (1 - beta1) * grad

      // Update biased second raw moment estimate
      newV[i] = beta2 * v[i]! + (1 - beta2) * grad * grad

      // Compute bias-corrected first moment estimate
      const mHat = newM[i]! / (1 - beta1 ** t)

      // Compute bias-corrected second raw moment estimate
      const vHat = newV[i]! / (1 - beta2 ** t)

      // Update weights
      newWeights[i] = weights[i]! - (this.config.learningRate * mHat) / (Math.sqrt(vHat) + epsilon)
    }

    return {
      weights: newWeights,
      state: { t, m: newM, v: newV },
    }
  }
}

/**
 * RMSprop Optimizer
 */
export class RMSpropOptimizer implements Optimizer {
  constructor(
    private config: OptimizerConfig & {
      decay?: number
      epsilon?: number
    }
  ) {}

  update(
    weights: Float32Array,
    gradients: Float32Array,
    state: OptimizerState
  ): { weights: Float32Array; state: OptimizerState } {
    const decay = this.config.decay ?? 0.9
    const epsilon = this.config.epsilon ?? 1e-8
    const clipValue = this.config.clipValue ?? Number.POSITIVE_INFINITY

    // Initialize cache if first iteration
    let cache = (state.cache as Float32Array) ?? new Float32Array(weights.length)

    const newWeights = new Float32Array(weights.length)
    const newCache = new Float32Array(weights.length)

    for (let i = 0; i < weights.length; i++) {
      // Gradient clipping
      let grad = gradients[i]!
      grad = Math.max(-clipValue, Math.min(clipValue, grad))

      // Update cache
      newCache[i] = decay * cache[i]! + (1 - decay) * grad * grad

      // Update weights
      newWeights[i] = weights[i]! - (this.config.learningRate * grad) / (Math.sqrt(newCache[i]!) + epsilon)
    }

    return {
      weights: newWeights,
      state: { ...state, t: state.t + 1, cache: newCache },
    }
  }
}

/**
 * Create optimizer by name
 */
export function createOptimizer(
  name: string,
  config: OptimizerConfig & { beta1?: number; beta2?: number; momentum?: number; decay?: number; epsilon?: number }
): Optimizer {
  switch (name.toLowerCase()) {
    case 'sgd':
      return new SGDOptimizer(config)
    case 'momentum':
      return new MomentumOptimizer(config)
    case 'adam':
      return new AdamOptimizer(config)
    case 'rmsprop':
      return new RMSpropOptimizer(config)
    default:
      throw new Error(`Unknown optimizer: ${name}`)
  }
}
