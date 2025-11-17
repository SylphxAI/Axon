# Project Context

## What (Internal)

NeuronLine is a pure functional, PyTorch-like neural network library for TypeScript/JavaScript. Built for production ML workloads in edge environments (browsers, serverless, embedded) where bundle size and performance are critical.

**Target**: ML practitioners who need PyTorch-like API but can't use Python or large frameworks.

**Scope**: Complete neural network library with autograd, modern layers (LSTM, Conv2D, Attention), and hardware acceleration (WASM, WebGPU).

## Why (Business/Internal)

**Problem**: Existing JavaScript ML libraries are either too large (TensorFlow.js ~146KB) or too limited (Brain.js - no modern layers, single optimizer).

**Gap**: No production-grade, functional, type-safe ML library optimized for JavaScript runtimes.

**Value**: Enable ML deployment to edge/serverless without Python dependencies or large bundles.

## Key Constraints

**Technical**:
- Pure functional (immutable tensors, no side effects)
- Bundle size <50KB gzipped (target: ~20KB)
- Performance: ≥4 eps/sec on 2048 DQN benchmark
- TypeScript-first with complete type safety
- Zero native dependencies (WASM/WebGPU optional)

**Business**:
- Must support browser, Node.js, Deno, Bun, edge runtimes
- API must be familiar to PyTorch users
- Must be faster than Brain.js for production use

**Legal**:
- MIT license (permissive, commercial-friendly)
- No GPL dependencies

## Boundaries

**In scope**:
- Neural network layers (Linear, Conv, LSTM, Attention)
- Automatic differentiation (autograd)
- Standard optimizers (SGD, Adam, RMSprop, AdaGrad)
- Model serialization
- Hardware acceleration (WASM, WebGPU)
- Training utilities (data loaders, schedulers)

**Out of scope**:
- Computer vision preprocessing (use external libraries)
- Audio/video processing
- Distributed training (may add later)
- Pre-trained model zoo (separate package)
- GUI/visualization (use external tools)

## SSOT References

<!-- VERIFY: package.json -->
- Dependencies: See `package.json` in each package
- Version: `0.1.2` (see `CHANGELOG.md`)
<!-- VERIFY: CHANGELOG.md -->
- Changes: `CHANGELOG.md`
<!-- VERIFY: PERFORMANCE.md -->
- Performance: `PERFORMANCE.md`
