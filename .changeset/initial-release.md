---
"@sylphx/tensor": minor
"@sylphx/nn": minor
"@sylphx/functional": minor
"@sylphx/optim": minor
"@sylphx/data": minor
"@sylphx/wasm": minor
"@sylphx/webgpu": minor
"@sylphx/core": minor
"@sylphx/predictors": minor
---

Initial release of Axon - Pure functional PyTorch-like neural network library

**Features:**
- Pure functional architecture with immutable tensors
- Automatic differentiation (autograd)
- Complete neural network layers: Linear, Conv2D, LSTM, GRU, BatchNorm, Dropout
- Multiple optimizers: SGD, Adam, RMSprop, AdaGrad
- Activation functions: ReLU, Leaky ReLU, Sigmoid, Tanh, Softmax
- Loss functions: MSE, Binary Cross Entropy, Cross Entropy, Huber
- WASM acceleration (2-2.7x speedup for matrix operations)
- WebGPU acceleration (async API for very large operations)
- Memory pooling (90%+ reduction in allocations)
- Batched training support (117x speedup)

**Performance:**
- 530 episodes/sec on 2048 DQN benchmark (155x faster than baseline)
- 0.87ms per training step (99.7% reduction)
- 35 MB heap usage (30x less memory)
- 29,819 examples/sec throughput

**Supported Runtimes:**
- Browser (Chrome, Edge, Firefox, Safari)
- Node.js >=18.0.0
- Bun >=1.0.0
- Deno
- Edge runtimes (Vercel, Cloudflare Workers)
