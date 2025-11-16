/**
 * Loss Functions for Neural Networks
 * Each function returns both loss value and gradient
 */

export type LossFunction = {
  compute: (predicted: number, actual: number) => number
  gradient: (predicted: number, actual: number) => number
}

/**
 * Mean Squared Error (MSE)
 * L = (predicted - actual)^2
 * dL/dpredicted = 2 * (predicted - actual)
 */
export const mse: LossFunction = {
  compute: (predicted: number, actual: number) => {
    const diff = predicted - actual
    return diff * diff
  },
  gradient: (predicted: number, actual: number) => 2 * (predicted - actual),
}

/**
 * Binary Cross-Entropy
 * L = -[actual * log(predicted) + (1 - actual) * log(1 - predicted)]
 * dL/dpredicted = -actual/predicted + (1 - actual)/(1 - predicted)
 */
export const binaryCrossEntropy: LossFunction = {
  compute: (predicted: number, actual: number) => {
    // Clip predictions to avoid log(0)
    const p = Math.max(1e-15, Math.min(1 - 1e-15, predicted))
    return -(actual * Math.log(p) + (1 - actual) * Math.log(1 - p))
  },
  gradient: (predicted: number, actual: number) => {
    // Clip predictions to avoid division by zero
    const p = Math.max(1e-15, Math.min(1 - 1e-15, predicted))
    return -actual / p + (1 - actual) / (1 - p)
  },
}

/**
 * Categorical Cross-Entropy (for multi-class)
 * Applied to vectors, not single values
 */
export function categoricalCrossEntropy(
  predicted: Float32Array,
  actual: Float32Array
): number {
  let loss = 0
  for (let i = 0; i < predicted.length; i++) {
    const p = Math.max(1e-15, Math.min(1 - 1e-15, predicted[i]!))
    loss -= actual[i]! * Math.log(p)
  }
  return loss
}

/**
 * Categorical Cross-Entropy Gradient
 */
export function categoricalCrossEntropyGradient(
  predicted: Float32Array,
  actual: Float32Array
): Float32Array {
  const gradient = new Float32Array(predicted.length)
  for (let i = 0; i < predicted.length; i++) {
    const p = Math.max(1e-15, Math.min(1 - 1e-15, predicted[i]!))
    gradient[i] = -actual[i]! / p
  }
  return gradient
}

/**
 * Huber Loss (robust to outliers)
 * L = 0.5 * (predicted - actual)^2 if |predicted - actual| <= delta
 * L = delta * (|predicted - actual| - 0.5 * delta) otherwise
 */
export function huberLoss(delta = 1.0): LossFunction {
  return {
    compute: (predicted: number, actual: number) => {
      const diff = Math.abs(predicted - actual)
      if (diff <= delta) {
        return 0.5 * diff * diff
      }
      return delta * (diff - 0.5 * delta)
    },
    gradient: (predicted: number, actual: number) => {
      const diff = predicted - actual
      if (Math.abs(diff) <= delta) {
        return diff
      }
      return delta * Math.sign(diff)
    },
  }
}

/**
 * Get loss function by name
 */
export function getLoss(name: string): LossFunction {
  switch (name.toLowerCase()) {
    case 'mse':
    case 'mean_squared_error':
      return mse
    case 'bce':
    case 'binary_crossentropy':
    case 'binary-crossentropy':
      return binaryCrossEntropy
    case 'huber':
      return huberLoss()
    default:
      throw new Error(`Unknown loss function: ${name}`)
  }
}
