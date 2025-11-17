# Changelog

All notable changes to NeuronLine will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.2] - 2024-11-17

### Added
- **LSTM/RNN layers** with complete forward pass implementation
  - 4 gates (input, forget, cell, output)
  - Hidden and cell state management
  - Single-step and sequence processing
  - Demo: `apps/demo/lstm-demo.ts`

- **Conv2D layer** with im2col transformation
  - Efficient convolution via matrix multiplication
  - Support for padding and stride
  - Tiled computation for cache efficiency
  - Demo: `apps/demo/conv2d-demo.ts`

- **BatchNorm layer**
  - Running statistics with momentum
  - Training and inference modes
  - Scale and shift parameters

- **Dropout layer**
  - Inverted dropout pattern
  - Training/inference mode support

- **Additional optimizers**
  - RMSprop with adaptive learning rates
  - AdaGrad with per-parameter adaptation

- **Data package** (`@neuronline/data`)
  - Dataset creation and batching
  - Train/val/test splitting
  - Data normalization utilities
  - Shuffle support

- **Model serialization**
  - Save models to JSON
  - Load models from JSON
  - File-based save/load helpers
  - Model summary generation

- **WASM package** (`@neuronline/wasm`)
  - AssemblyScript implementation
  - Tiled matmul (32x32 tiles)
  - 8x loop unrolling for element-wise ops
  - Fast activation approximations
  - Compiles to 1.4KB WASM module

- **WebGPU package** (`@neuronline/webgpu`)
  - GPU compute shaders
  - Matrix multiplication on GPU
  - Element-wise operations
  - Activation functions (ReLU, etc.)
  - Automatic buffer management

### Changed
- **Performance optimizations**
  - Tiled/blocked matrix multiplication (32x32 tiles)
  - 4x inner loop unrolling in matmul
  - 8x loop unrolling in element-wise ops (up from 4x)
  - Better cache locality and ILP
  - +22.4% performance vs baseline

- **Updated README**
  - Reflects pure functional architecture
  - Comprehensive feature documentation
  - PyTorch-like API examples
  - Performance metrics and benchmarks

### Performance
- **Baseline (v0.1.0)**: 3.35 episodes/sec
- **v0.1.1**: 4.11 episodes/sec (+22.7%)
- **v0.1.2**: 4.10 episodes/sec (+22.4%)

Benchmark: 100 episodes of 2048 DQN training

## [0.1.1] - 2024-01-16

### Added
- Performance benchmarking system
- Benchmark logging to JSONL
- PERFORMANCE.md tracking document

### Changed
- Optimized tensor operations
  - 4x loop unrolling in add/mul
  - Local variable caching in matmul
  - Pre-calculated indices
  - Reduced array access overhead

### Performance
- +22.7% faster than baseline
- 505 steps/sec (up from 390)
- 241.92ms per training step (down from 296.80ms)

## [0.1.0] - 2024-01-16

### Added
- **Pure functional architecture**
  - Immutable tensors
  - Functional operations
  - No side effects

- **Automatic differentiation (autograd)**
  - Computational graph construction
  - Backward pass for gradients
  - Gradient accumulation

- **Core tensor operations**
  - Element-wise: add, sub, mul, div, neg
  - Matrix: matmul, transpose
  - Reductions: sum
  - Broadcasting support

- **Neural network layers**
  - Linear (Dense) layer
  - Initialization: Xavier normal, zeros, ones

- **Activation functions**
  - ReLU, Leaky ReLU
  - Sigmoid, Tanh
  - Softmax

- **Loss functions**
  - Mean Squared Error (MSE)
  - Binary Cross Entropy
  - Cross Entropy
  - Huber Loss

- **Optimizers**
  - SGD with momentum
  - Adam (adaptive learning rate)

- **Examples**
  - XOR problem demo
  - Binary classification
  - Regression (sin function)
  - 2048 game with DQN

### Architecture
- Monorepo structure with 5 packages:
  - `@neuronline/tensor` - Core tensor ops
  - `@neuronline/functional` - Activations, loss
  - `@neuronline/nn` - Neural network layers
  - `@neuronline/optim` - Optimizers
  - `@neuronline/data` - Data utilities

### Performance
- Baseline: 3.35 episodes/sec on 2048 DQN
- 390 steps/sec
- Pure TypeScript, no native dependencies

## [0.0.1] - Initial Release

### Added
- Basic neural network functionality
- Online learning algorithms
- Bandit algorithms (Thompson Sampling)
- Click prediction models

---

## Legend

- **Added**: New features
- **Changed**: Changes in existing functionality
- **Deprecated**: Soon-to-be removed features
- **Removed**: Removed features
- **Fixed**: Bug fixes
- **Security**: Security fixes
- **Performance**: Performance improvements
