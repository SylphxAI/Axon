/**
 * WASM-accelerated tensor operations
 * Written in AssemblyScript for maximum performance
 */

/**
 * Matrix multiplication (optimized WASM version)
 * C = A @ B where A is [m, k] and B is [k, n]
 */
export function matmul(
  a: Float32Array,
  b: Float32Array,
  c: Float32Array,
  m: i32,
  k: i32,
  n: i32
): void {
  // Optimized matmul with loop tiling for cache efficiency
  const TILE_SIZE: i32 = 32

  for (let i: i32 = 0; i < m; i += TILE_SIZE) {
    for (let j: i32 = 0; j < n; j += TILE_SIZE) {
      for (let kk: i32 = 0; kk < k; kk += TILE_SIZE) {
        // Tile boundaries
        const iMax = min(i + TILE_SIZE, m)
        const jMax = min(j + TILE_SIZE, n)
        const kMax = min(kk + TILE_SIZE, k)

        // Compute tile
        for (let ii: i32 = i; ii < iMax; ii++) {
          for (let jj: i32 = j; jj < jMax; jj++) {
            let sum: f32 = c[ii * n + jj]

            for (let kkk: i32 = kk; kkk < kMax; kkk++) {
              sum += a[ii * k + kkk] * b[kkk * n + jj]
            }

            c[ii * n + jj] = sum
          }
        }
      }
    }
  }
}

/**
 * Element-wise addition with loop unrolling
 */
export function add(a: Float32Array, b: Float32Array, c: Float32Array, len: i32): void {
  let i: i32 = 0

  // Unroll by 8 for SIMD-like performance
  const len8 = len - (len % 8)
  for (; i < len8; i += 8) {
    c[i] = a[i] + b[i]
    c[i + 1] = a[i + 1] + b[i + 1]
    c[i + 2] = a[i + 2] + b[i + 2]
    c[i + 3] = a[i + 3] + b[i + 3]
    c[i + 4] = a[i + 4] + b[i + 4]
    c[i + 5] = a[i + 5] + b[i + 5]
    c[i + 6] = a[i + 6] + b[i + 6]
    c[i + 7] = a[i + 7] + b[i + 7]
  }

  // Handle remainder
  for (; i < len; i++) {
    c[i] = a[i] + b[i]
  }
}

/**
 * Element-wise multiplication with loop unrolling
 */
export function mul(a: Float32Array, b: Float32Array, c: Float32Array, len: i32): void {
  let i: i32 = 0

  // Unroll by 8
  const len8 = len - (len % 8)
  for (; i < len8; i += 8) {
    c[i] = a[i] * b[i]
    c[i + 1] = a[i + 1] * b[i + 1]
    c[i + 2] = a[i + 2] * b[i + 2]
    c[i + 3] = a[i + 3] * b[i + 3]
    c[i + 4] = a[i + 4] * b[i + 4]
    c[i + 5] = a[i + 5] * b[i + 5]
    c[i + 6] = a[i + 6] * b[i + 6]
    c[i + 7] = a[i + 7] * b[i + 7]
  }

  for (; i < len; i++) {
    c[i] = a[i] * b[i]
  }
}

/**
 * ReLU activation
 */
export function relu(input: Float32Array, output: Float32Array, len: i32): void {
  for (let i: i32 = 0; i < len; i++) {
    output[i] = max(0, input[i])
  }
}

/**
 * Sigmoid activation (approximation for speed)
 */
export function sigmoid(input: Float32Array, output: Float32Array, len: i32): void {
  for (let i: i32 = 0; i < len; i++) {
    const x = input[i]
    // Fast sigmoid approximation: 0.5 + 0.5 * x / (1 + abs(x))
    output[i] = 0.5 + 0.5 * x / (1 + abs(x))
  }
}

/**
 * Tanh activation
 */
export function tanh(input: Float32Array, output: Float32Array, len: i32): void {
  for (let i: i32 = 0; i < len; i++) {
    const x = input[i]
    const e2x = exp(2 * x)
    output[i] = (e2x - 1) / (e2x + 1)
  }
}

/**
 * Helper: min
 */
function min(a: i32, b: i32): i32 {
  return a < b ? a : b
}

/**
 * Helper: max
 */
function max(a: f32, b: f32): f32 {
  return a > b ? a : b
}

/**
 * Helper: abs
 */
function abs(x: f32): f32 {
  return x < 0 ? -x : x
}

/**
 * Helper: exp (approximation)
 */
function exp(x: f32): f32 {
  // Fast exp approximation using Padé approximant
  if (x > 10) return 22026.465794806718 // e^10
  if (x < -10) return 0.000045399929762484854 // e^-10

  const a = 1.0 + x / 256.0
  const a2 = a * a
  const a4 = a2 * a2
  const a8 = a4 * a4
  const a16 = a8 * a8
  const a32 = a16 * a16
  const a64 = a32 * a32
  const a128 = a64 * a64
  const a256 = a128 * a128

  return a256
}
