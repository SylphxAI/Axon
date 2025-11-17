/**
 * @neuronline/webgpu
 * WebGPU-accelerated tensor operations
 */

export type WebGPUDevice = {
  device: GPUDevice
  queue: GPUQueue
}

let gpuDevice: WebGPUDevice | null = null

/**
 * Initialize WebGPU device
 * Must be called before using GPU operations
 */
export async function initWebGPU(): Promise<WebGPUDevice> {
  if (gpuDevice) {
    return gpuDevice
  }

  // Check WebGPU support
  if (!navigator.gpu) {
    throw new Error('WebGPU not supported in this browser')
  }

  // Request adapter
  const adapter = await navigator.gpu.requestAdapter()
  if (!adapter) {
    throw new Error('Failed to get WebGPU adapter')
  }

  // Request device
  const device = await adapter.requestDevice()

  gpuDevice = {
    device,
    queue: device.queue,
  }

  return gpuDevice
}

/**
 * Get initialized WebGPU device
 */
export function getWebGPU(): WebGPUDevice {
  if (!gpuDevice) {
    throw new Error('WebGPU not initialized. Call initWebGPU() first.')
  }
  return gpuDevice
}

/**
 * Check if WebGPU is supported
 */
export function isWebGPUSupported(): boolean {
  return typeof navigator !== 'undefined' && 'gpu' in navigator
}

/**
 * Check if WebGPU is initialized
 */
export function isWebGPUInitialized(): boolean {
  return gpuDevice !== null
}

/**
 * Matrix multiplication using WebGPU compute shader
 * C = A @ B where A is [m, k] and B is [k, n]
 */
export async function matmulGPU(
  a: Float32Array,
  b: Float32Array,
  m: number,
  k: number,
  n: number
): Promise<Float32Array> {
  const { device, queue } = getWebGPU()

  // Create buffers
  const aBuffer = device.createBuffer({
    size: a.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  })
  new Float32Array(aBuffer.getMappedRange()).set(a)
  aBuffer.unmap()

  const bBuffer = device.createBuffer({
    size: b.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  })
  new Float32Array(bBuffer.getMappedRange()).set(b)
  bBuffer.unmap()

  const resultSize = m * n * 4 // 4 bytes per f32
  const resultBuffer = device.createBuffer({
    size: resultSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  })

  const readBuffer = device.createBuffer({
    size: resultSize,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  })

  // Create shader module
  const shaderModule = device.createShaderModule({
    code: `
      @group(0) @binding(0) var<storage, read> a: array<f32>;
      @group(0) @binding(1) var<storage, read> b: array<f32>;
      @group(0) @binding(2) var<storage, read_write> result: array<f32>;

      struct Dimensions {
        m: u32,
        k: u32,
        n: u32,
      }
      @group(0) @binding(3) var<uniform> dims: Dimensions;

      @compute @workgroup_size(8, 8)
      fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
        let row = global_id.x;
        let col = global_id.y;

        if (row >= dims.m || col >= dims.n) {
          return;
        }

        var sum = 0.0;
        for (var i = 0u; i < dims.k; i++) {
          sum += a[row * dims.k + i] * b[i * dims.n + col];
        }

        result[row * dims.n + col] = sum;
      }
    `,
  })

  // Create uniform buffer for dimensions
  const dimsBuffer = device.createBuffer({
    size: 12, // 3 u32s
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  })
  new Uint32Array(dimsBuffer.getMappedRange()).set([m, k, n])
  dimsBuffer.unmap()

  // Create bind group layout and pipeline
  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
    ],
  })

  const pipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({
      bindGroupLayouts: [bindGroupLayout],
    }),
    compute: {
      module: shaderModule,
      entryPoint: 'main',
    },
  })

  const bindGroup = device.createBindGroup({
    layout: bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: aBuffer } },
      { binding: 1, resource: { buffer: bBuffer } },
      { binding: 2, resource: { buffer: resultBuffer } },
      { binding: 3, resource: { buffer: dimsBuffer } },
    ],
  })

  // Execute compute shader
  const commandEncoder = device.createCommandEncoder()
  const passEncoder = commandEncoder.beginComputePass()
  passEncoder.setPipeline(pipeline)
  passEncoder.setBindGroup(0, bindGroup)
  passEncoder.dispatchWorkgroups(Math.ceil(m / 8), Math.ceil(n / 8))
  passEncoder.end()

  // Copy result to read buffer
  commandEncoder.copyBufferToBuffer(resultBuffer, 0, readBuffer, 0, resultSize)

  queue.submit([commandEncoder.finish()])

  // Read result
  await readBuffer.mapAsync(GPUMapMode.READ)
  const result = new Float32Array(readBuffer.getMappedRange()).slice()
  readBuffer.unmap()

  // Cleanup
  aBuffer.destroy()
  bBuffer.destroy()
  resultBuffer.destroy()
  readBuffer.destroy()
  dimsBuffer.destroy()

  return result
}

/**
 * Element-wise addition using WebGPU
 */
export async function addGPU(
  a: Float32Array,
  b: Float32Array,
  len: number
): Promise<Float32Array> {
  const { device, queue } = getWebGPU()

  const aBuffer = device.createBuffer({
    size: a.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  })
  new Float32Array(aBuffer.getMappedRange()).set(a)
  aBuffer.unmap()

  const bBuffer = device.createBuffer({
    size: b.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  })
  new Float32Array(bBuffer.getMappedRange()).set(b)
  bBuffer.unmap()

  const resultBuffer = device.createBuffer({
    size: a.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  })

  const readBuffer = device.createBuffer({
    size: a.byteLength,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  })

  const shaderModule = device.createShaderModule({
    code: `
      @group(0) @binding(0) var<storage, read> a: array<f32>;
      @group(0) @binding(1) var<storage, read> b: array<f32>;
      @group(0) @binding(2) var<storage, read_write> result: array<f32>;

      @compute @workgroup_size(256)
      fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
        let i = global_id.x;
        if (i < arrayLength(&a)) {
          result[i] = a[i] + b[i];
        }
      }
    `,
  })

  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    ],
  })

  const pipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({
      bindGroupLayouts: [bindGroupLayout],
    }),
    compute: {
      module: shaderModule,
      entryPoint: 'main',
    },
  })

  const bindGroup = device.createBindGroup({
    layout: bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: aBuffer } },
      { binding: 1, resource: { buffer: bBuffer } },
      { binding: 2, resource: { buffer: resultBuffer } },
    ],
  })

  const commandEncoder = device.createCommandEncoder()
  const passEncoder = commandEncoder.beginComputePass()
  passEncoder.setPipeline(pipeline)
  passEncoder.setBindGroup(0, bindGroup)
  passEncoder.dispatchWorkgroups(Math.ceil(len / 256))
  passEncoder.end()

  commandEncoder.copyBufferToBuffer(resultBuffer, 0, readBuffer, 0, a.byteLength)

  queue.submit([commandEncoder.finish()])

  await readBuffer.mapAsync(GPUMapMode.READ)
  const result = new Float32Array(readBuffer.getMappedRange()).slice()
  readBuffer.unmap()

  aBuffer.destroy()
  bBuffer.destroy()
  resultBuffer.destroy()
  readBuffer.destroy()

  return result
}

/**
 * ReLU activation using WebGPU
 */
export async function reluGPU(input: Float32Array, len: number): Promise<Float32Array> {
  const { device, queue } = getWebGPU()

  const inputBuffer = device.createBuffer({
    size: input.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  })
  new Float32Array(inputBuffer.getMappedRange()).set(input)
  inputBuffer.unmap()

  const resultBuffer = device.createBuffer({
    size: input.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  })

  const readBuffer = device.createBuffer({
    size: input.byteLength,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  })

  const shaderModule = device.createShaderModule({
    code: `
      @group(0) @binding(0) var<storage, read> input: array<f32>;
      @group(0) @binding(1) var<storage, read_write> result: array<f32>;

      @compute @workgroup_size(256)
      fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
        let i = global_id.x;
        if (i < arrayLength(&input)) {
          result[i] = max(0.0, input[i]);
        }
      }
    `,
  })

  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    ],
  })

  const pipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({
      bindGroupLayouts: [bindGroupLayout],
    }),
    compute: {
      module: shaderModule,
      entryPoint: 'main',
    },
  })

  const bindGroup = device.createBindGroup({
    layout: bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: inputBuffer } },
      { binding: 1, resource: { buffer: resultBuffer } },
    ],
  })

  const commandEncoder = device.createCommandEncoder()
  const passEncoder = commandEncoder.beginComputePass()
  passEncoder.setPipeline(pipeline)
  passEncoder.setBindGroup(0, bindGroup)
  passEncoder.dispatchWorkgroups(Math.ceil(len / 256))
  passEncoder.end()

  commandEncoder.copyBufferToBuffer(resultBuffer, 0, readBuffer, 0, input.byteLength)

  queue.submit([commandEncoder.finish()])

  await readBuffer.mapAsync(GPUMapMode.READ)
  const result = new Float32Array(readBuffer.getMappedRange()).slice()
  readBuffer.unmap()

  inputBuffer.destroy()
  resultBuffer.destroy()
  readBuffer.destroy()

  return result
}
