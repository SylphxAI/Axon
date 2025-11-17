import { defineConfig } from 'vitepress'

export default defineConfig({
  title: 'Axon',
  description: 'Pure functional PyTorch-like neural network library for TypeScript/JavaScript',
  
  themeConfig: {
    logo: '/logo.svg',
    
    nav: [
      { text: 'Guide', link: '/guide/getting-started' },
      { text: 'API', link: '/api/tensor' },
      { text: 'GitHub', link: 'https://github.com/SylphxAI/Axon' }
    ],

    sidebar: {
      '/guide/': [
        {
          text: 'Introduction',
          items: [
            { text: 'Getting Started', link: '/guide/getting-started' },
            { text: 'Why Axon?', link: '/guide/why-axon' },
            { text: 'Performance', link: '/guide/performance' }
          ]
        },
        {
          text: 'Core Concepts',
          items: [
            { text: 'Tensors', link: '/guide/tensors' },
            { text: 'Autograd', link: '/guide/autograd' },
            { text: 'Neural Networks', link: '/guide/neural-networks' }
          ]
        },
        {
          text: 'Advanced',
          items: [
            { text: 'WASM Acceleration', link: '/guide/wasm' },
            { text: 'WebGPU Acceleration', link: '/guide/webgpu' },
            { text: 'Memory Pooling', link: '/guide/memory-pooling' }
          ]
        }
      ],
      '/api/': [
        {
          text: 'API Reference',
          items: [
            { text: 'Tensor', link: '/api/tensor' },
            { text: 'Neural Networks', link: '/api/nn' },
            { text: 'Functional', link: '/api/functional' },
            { text: 'Optimizers', link: '/api/optim' },
            { text: 'Data', link: '/api/data' }
          ]
        }
      ]
    },

    socialLinks: [
      { icon: 'github', link: 'https://github.com/SylphxAI/Axon' }
    ],

    footer: {
      message: 'Released under the MIT License.',
      copyright: 'Copyright © 2024 SylphxAI'
    }
  }
})
