/**
 * Conv2D layer - Pure functional implementation
 */

import type { Tensor } from '@sylphx/tensor'
import { heNormal, uniform, matmul, acquireBuffer } from '@sylphx/tensor'

/**
 * Conv2D layer state
 */
export type Conv2DState = {
  readonly weight: Tensor // [outChannels, inChannels, kernelH, kernelW]
  readonly bias: Tensor // [outChannels]
  readonly stride: number
  readonly padding: number
}

/**
 * Initialize Conv2D layer
 *
 * @param inChannels Number of input channels
 * @param outChannels Number of output channels
 * @param kernelSize Kernel size (square)
 * @param stride Stride (default: 1)
 * @param padding Padding (default: 0)
 */
export function init(
  inChannels: number,
  outChannels: number,
  kernelSize: number,
  stride: number = 1,
  padding: number = 0
): Conv2DState {
  const bound = 1 / Math.sqrt(inChannels * kernelSize * kernelSize)

  // Weight: [outChannels, inChannels, kernelH, kernelW]
  // For now, flatten to 2D for compatibility
  const weight = heNormal(
    [outChannels, inChannels * kernelSize * kernelSize],
    { requiresGrad: true }
  )

  return {
    weight,
    bias: uniform([outChannels], -bound, bound, { requiresGrad: true }),
    stride,
    padding,
  }
}

/**
 * Forward pass through Conv2D using im2col
 *
 * Input: Flattened 4D tensor [batch * channels * height * width]
 * Expected logical shape: [batch, channels, height, width]
 * Output: Flattened 4D tensor [batch * outChannels * outHeight * outWidth]
 */
export function forward(
  input: Tensor,
  state: Conv2DState,
  inputShape: [number, number, number, number] // [batch, channels, height, width]
): Tensor {
  const [batch, inChannels, inH, inW] = inputShape
  const { weight, bias, stride, padding } = state

  // Extract kernel dimensions from weight shape
  // Weight shape: [outChannels, inChannels * kH * kW]
  const kernelSize = Math.sqrt(weight.shape[1]! / inChannels)
  const kH = kernelSize
  const kW = kernelSize

  if (!Number.isInteger(kernelSize)) {
    throw new Error('Invalid weight shape for square kernel')
  }

  // Calculate output dimensions
  const outH = Math.floor((inH + 2 * padding - kH) / stride + 1)
  const outW = Math.floor((inW + 2 * padding - kW) / stride + 1)

  // Apply padding if needed
  const paddedInput = padding > 0
    ? padInput(input, inputShape, padding)
    : input

  const paddedH = inH + 2 * padding
  const paddedW = inW + 2 * padding

  // im2col transformation
  const colMatrix = im2col(
    paddedInput,
    [batch, inChannels, paddedH, paddedW],
    kH,
    kW,
    stride
  )

  // colMatrix shape: [inChannels * kH * kW, batch * outH * outW]
  // weight shape: [outChannels, inChannels * kH * kW]
  // Result: [outChannels, batch * outH * outW]
  const convResult = matmul(weight, colMatrix)

  // Add bias: broadcast bias across spatial dimensions
  const output = addBias(convResult, bias, batch, outH, outW)

  // Return with flattened data, caller can reshape as needed
  return output
}

/**
 * Calculate output dimensions
 */
export function getOutputShape(
  inputH: number,
  inputW: number,
  kernelSize: number,
  stride: number,
  padding: number
): { height: number; width: number } {
  const height = Math.floor((inputH + 2 * padding - kernelSize) / stride + 1)
  const width = Math.floor((inputW + 2 * padding - kernelSize) / stride + 1)

  return { height, width }
}

/**
 * im2col transformation - converts image to column matrix
 * Extracts all patches for convolution
 *
 * Input: [batch, channels, height, width] (flattened)
 * Output: [channels * kH * kW, batch * outH * outW] (as 2D tensor)
 */
function im2col(
  input: Tensor,
  inputShape: [number, number, number, number],
  kH: number,
  kW: number,
  stride: number
): Tensor {
  const [batch, channels, height, width] = inputShape
  const outH = Math.floor((height - kH) / stride + 1)
  const outW = Math.floor((width - kW) / stride + 1)

  const colHeight = channels * kH * kW
  const colWidth = batch * outH * outW

  const colData = acquireBuffer(colHeight * colWidth)

  // For each output position
  let colIdx = 0
  for (let b = 0; b < batch; b++) {
    for (let outY = 0; outY < outH; outY++) {
      for (let outX = 0; outX < outW; outX++) {
        const startY = outY * stride
        const startX = outX * stride

        // Extract patch
        let rowIdx = 0
        for (let c = 0; c < channels; c++) {
          for (let ky = 0; ky < kH; ky++) {
            for (let kx = 0; kx < kW; kx++) {
              const y = startY + ky
              const x = startX + kx
              const inputIdx = ((b * channels + c) * height + y) * width + x
              colData[rowIdx * colWidth + colIdx] = input.data[inputIdx]!
              rowIdx++
            }
          }
        }
        colIdx++
      }
    }
  }

  return {
    data: colData,
    shape: [colHeight, colWidth],
    requiresGrad: input.requiresGrad,
  }
}

/**
 * Add padding to input tensor
 */
function padInput(
  input: Tensor,
  inputShape: [number, number, number, number],
  padding: number
): Tensor {
  const [batch, channels, height, width] = inputShape
  const paddedH = height + 2 * padding
  const paddedW = width + 2 * padding

  const paddedData = acquireBuffer(batch * channels * paddedH * paddedW)

  for (let b = 0; b < batch; b++) {
    for (let c = 0; c < channels; c++) {
      for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
          const inputIdx = ((b * channels + c) * height + y) * width + x
          const paddedY = y + padding
          const paddedX = x + padding
          const paddedIdx = ((b * channels + c) * paddedH + paddedY) * paddedW + paddedX
          paddedData[paddedIdx] = input.data[inputIdx]!
        }
      }
    }
  }

  return {
    data: paddedData,
    shape: input.shape, // Keep original shape reference
    requiresGrad: input.requiresGrad,
  }
}

/**
 * Add bias to convolution result
 * convResult: [outChannels, batch * outH * outW]
 * bias: [outChannels]
 */
function addBias(
  convResult: Tensor,
  bias: Tensor,
  batch: number,
  outH: number,
  outW: number
): Tensor {
  const outChannels = convResult.shape[0]!
  const spatialSize = batch * outH * outW

  const output = acquireBuffer(outChannels * spatialSize)

  for (let c = 0; c < outChannels; c++) {
    const biasVal = bias.data[c]!
    const offset = c * spatialSize

    // Unroll by 8 for better performance
    let i = 0
    const len8 = spatialSize - 7
    for (; i < len8; i += 8) {
      output[offset + i] = convResult.data[offset + i]! + biasVal
      output[offset + i + 1] = convResult.data[offset + i + 1]! + biasVal
      output[offset + i + 2] = convResult.data[offset + i + 2]! + biasVal
      output[offset + i + 3] = convResult.data[offset + i + 3]! + biasVal
      output[offset + i + 4] = convResult.data[offset + i + 4]! + biasVal
      output[offset + i + 5] = convResult.data[offset + i + 5]! + biasVal
      output[offset + i + 6] = convResult.data[offset + i + 6]! + biasVal
      output[offset + i + 7] = convResult.data[offset + i + 7]! + biasVal
    }

    // Handle remainder
    for (; i < spatialSize; i++) {
      output[offset + i] = convResult.data[offset + i]! + biasVal
    }
  }

  return {
    data: output,
    shape: [outChannels, spatialSize],
    requiresGrad: convResult.requiresGrad,
  }
}
