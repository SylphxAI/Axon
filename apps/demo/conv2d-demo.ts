/**
 * Conv2D Demo - Demonstrates 2D convolution with im2col
 */

import * as nn from '../../packages/nn/src/index'
import * as T from '../../packages/tensor/src/index'

console.log('🔲 Conv2D Demo - 2D Convolution with im2col\n')

// Configuration
const batch = 1
const inChannels = 1
const height = 5
const width = 5
const outChannels = 2
const kernelSize = 3
const stride = 1
const padding = 0

// Initialize Conv2D layer
console.log('Initializing Conv2D...')
const conv = nn.conv2d.init(inChannels, outChannels, kernelSize, stride, padding)
console.log(`✅ Conv2D initialized`)
console.log(`   Input: [${batch}, ${inChannels}, ${height}, ${width}]`)
console.log(`   Kernel: ${kernelSize}x${kernelSize}`)
console.log(`   Output channels: ${outChannels}`)
console.log(`   Stride: ${stride}, Padding: ${padding}\n`)

// Create a simple input (5x5 image with a simple pattern)
const inputData = new Float32Array([
  // Channel 0
  1, 2, 3, 4, 5,
  6, 7, 8, 9, 10,
  11, 12, 13, 14, 15,
  16, 17, 18, 19, 20,
  21, 22, 23, 24, 25,
])

const input = T.tensor(Array.from(inputData), { requiresGrad: true })

console.log('Input (5x5):')
for (let y = 0; y < height; y++) {
  const row = []
  for (let x = 0; x < width; x++) {
    row.push(inputData[y * width + x]!.toString().padStart(2, ' '))
  }
  console.log('  ' + row.join(' '))
}

// Calculate expected output size
const { height: outH, width: outW } = nn.conv2d.getOutputShape(
  height,
  width,
  kernelSize,
  stride,
  padding
)

console.log(`\nExpected output shape: [${batch}, ${outChannels}, ${outH}, ${outW}]`)

// Forward pass
console.log('\nRunning forward pass...')
const output = nn.conv2d.forward(input, conv, [batch, inChannels, height, width])

console.log(`✅ Forward pass complete`)
console.log(`   Output shape: [${output.shape}]`)
console.log(`   Output size: ${output.data.length} elements\n`)

// Display output for first channel
console.log('Output channel 0 (first few values):')
const channel0Start = 0
const channel0Size = outH * outW
console.log('  ' + Array.from(output.data.slice(channel0Start, channel0Start + Math.min(9, channel0Size)))
  .map(x => x.toFixed(2))
  .join(', '))

console.log('\nOutput channel 1 (first few values):')
const channel1Start = channel0Size
console.log('  ' + Array.from(output.data.slice(channel1Start, channel1Start + Math.min(9, channel0Size)))
  .map(x => x.toFixed(2))
  .join(', '))

console.log('\n✅ Conv2D demonstration complete!')

// Test with padding
console.log('\n--- Testing with padding ---')
const convPadded = nn.conv2d.init(inChannels, outChannels, kernelSize, stride, 1)
const { height: outHPad, width: outWPad } = nn.conv2d.getOutputShape(
  height,
  width,
  kernelSize,
  stride,
  1 // padding
)

console.log(`With padding=1: output shape [${batch}, ${outChannels}, ${outHPad}, ${outWPad}]`)
const outputPadded = nn.conv2d.forward(input, convPadded, [batch, inChannels, height, width])
console.log(`✅ Padded convolution complete`)
console.log(`   Output size: ${outputPadded.data.length} elements`)
