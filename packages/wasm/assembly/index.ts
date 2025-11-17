/**
 * WASM-accelerated tensor operations
 * Written in AssemblyScript for maximum performance
 */

/**
 * Matrix multiplication (optimized WASM version)
 * C = A @ B where A is [m, k] and B is [k, n]
 * Works with raw pointers to float data
 */
export function matmul(
  aPtr: usize,
  bPtr: usize,
  cPtr: usize,
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
            const cIdx = (ii * n + jj) << 2
            let sum: f32 = load<f32>(cPtr + cIdx)

            for (let kkk: i32 = kk; kkk < kMax; kkk++) {
              const aIdx = (ii * k + kkk) << 2
              const bIdx = (kkk * n + jj) << 2
              sum += load<f32>(aPtr + aIdx) * load<f32>(bPtr + bIdx)
            }

            store<f32>(cPtr + cIdx, sum)
          }
        }
      }
    }
  }
}

/**
 * Element-wise addition with loop unrolling
 * Works with raw pointers to float data
 */
export function add(aPtr: usize, bPtr: usize, cPtr: usize, len: i32): void {
  let i: i32 = 0

  // Unroll by 8 for SIMD-like performance
  const len8 = len - (len % 8)
  for (; i < len8; i += 8) {
    store<f32>(cPtr + (i << 2), load<f32>(aPtr + (i << 2)) + load<f32>(bPtr + (i << 2)))
    store<f32>(cPtr + ((i + 1) << 2), load<f32>(aPtr + ((i + 1) << 2)) + load<f32>(bPtr + ((i + 1) << 2)))
    store<f32>(cPtr + ((i + 2) << 2), load<f32>(aPtr + ((i + 2) << 2)) + load<f32>(bPtr + ((i + 2) << 2)))
    store<f32>(cPtr + ((i + 3) << 2), load<f32>(aPtr + ((i + 3) << 2)) + load<f32>(bPtr + ((i + 3) << 2)))
    store<f32>(cPtr + ((i + 4) << 2), load<f32>(aPtr + ((i + 4) << 2)) + load<f32>(bPtr + ((i + 4) << 2)))
    store<f32>(cPtr + ((i + 5) << 2), load<f32>(aPtr + ((i + 5) << 2)) + load<f32>(bPtr + ((i + 5) << 2)))
    store<f32>(cPtr + ((i + 6) << 2), load<f32>(aPtr + ((i + 6) << 2)) + load<f32>(bPtr + ((i + 6) << 2)))
    store<f32>(cPtr + ((i + 7) << 2), load<f32>(aPtr + ((i + 7) << 2)) + load<f32>(bPtr + ((i + 7) << 2)))
  }

  // Handle remainder
  for (; i < len; i++) {
    store<f32>(cPtr + (i << 2), load<f32>(aPtr + (i << 2)) + load<f32>(bPtr + (i << 2)))
  }
}

/**
 * Element-wise multiplication with loop unrolling
 * Works with raw pointers to float data
 */
export function mul(aPtr: usize, bPtr: usize, cPtr: usize, len: i32): void {
  let i: i32 = 0

  // Unroll by 8
  const len8 = len - (len % 8)
  for (; i < len8; i += 8) {
    store<f32>(cPtr + (i << 2), load<f32>(aPtr + (i << 2)) * load<f32>(bPtr + (i << 2)))
    store<f32>(cPtr + ((i + 1) << 2), load<f32>(aPtr + ((i + 1) << 2)) * load<f32>(bPtr + ((i + 1) << 2)))
    store<f32>(cPtr + ((i + 2) << 2), load<f32>(aPtr + ((i + 2) << 2)) * load<f32>(bPtr + ((i + 2) << 2)))
    store<f32>(cPtr + ((i + 3) << 2), load<f32>(aPtr + ((i + 3) << 2)) * load<f32>(bPtr + ((i + 3) << 2)))
    store<f32>(cPtr + ((i + 4) << 2), load<f32>(aPtr + ((i + 4) << 2)) * load<f32>(bPtr + ((i + 4) << 2)))
    store<f32>(cPtr + ((i + 5) << 2), load<f32>(aPtr + ((i + 5) << 2)) * load<f32>(bPtr + ((i + 5) << 2)))
    store<f32>(cPtr + ((i + 6) << 2), load<f32>(aPtr + ((i + 6) << 2)) * load<f32>(bPtr + ((i + 6) << 2)))
    store<f32>(cPtr + ((i + 7) << 2), load<f32>(aPtr + ((i + 7) << 2)) * load<f32>(bPtr + ((i + 7) << 2)))
  }

  for (; i < len; i++) {
    store<f32>(cPtr + (i << 2), load<f32>(aPtr + (i << 2)) * load<f32>(bPtr + (i << 2)))
  }
}

/**
 * ReLU activation
 * Works with raw pointers to float data
 */
export function relu(inputPtr: usize, outputPtr: usize, len: i32): void {
  for (let i: i32 = 0; i < len; i++) {
    const val = load<f32>(inputPtr + (i << 2))
    store<f32>(outputPtr + (i << 2), max(0, val))
  }
}

/**
 * Sigmoid activation (approximation for speed)
 * Works with raw pointers to float data
 */
export function sigmoid(inputPtr: usize, outputPtr: usize, len: i32): void {
  for (let i: i32 = 0; i < len; i++) {
    const x = load<f32>(inputPtr + (i << 2))
    // Fast sigmoid approximation: 0.5 + 0.5 * x / (1 + abs(x))
    const result = <f32>(0.5 + 0.5 * x / (1 + abs(x)))
    store<f32>(outputPtr + (i << 2), result)
  }
}

/**
 * Tanh activation
 * Works with raw pointers to float data
 */
export function tanh(inputPtr: usize, outputPtr: usize, len: i32): void {
  for (let i: i32 = 0; i < len; i++) {
    const x = load<f32>(inputPtr + (i << 2))
    const e2x = exp(2 * x)
    const result = (e2x - 1) / (e2x + 1)
    store<f32>(outputPtr + (i << 2), result)
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
  if (x > 10) return <f32>22026.465794806718 // e^10
  if (x < -10) return <f32>0.000045399929762484854 // e^-10

  const a = <f32>(1.0 + x / 256.0)
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
