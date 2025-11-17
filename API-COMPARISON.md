# Pure Functional API - 新舊對比

## 🎯 核心理念

**舊 API (類別導向):**
- 使用 `new` 關鍵字
- 類別實例
- 隱藏狀態
- 方法調用

**新 API (純函數):**
- 無 `new` 關鍵字
- 工廠函數
- 明確狀態傳遞
- 函數組合

---

## 📦 1. 創建模型

### 舊 API
```typescript
import { Linear, Sequential } from '@sylphx/nn'

// 使用類別
const model = new Sequential([
  new Linear(2, 8),
  new Tanh(),
  new Linear(8, 1)
])

// 狀態隱藏在實例內
// 無法直接訪問或序列化
```

### 新 API ✅
```typescript
import { Sequential, Linear, Tanh } from '@sylphx/nn'

// 使用函數組合
const model = Sequential(
  Linear(2, 8),
  Tanh(),
  Linear(8, 1)
)

// 明確的狀態管理
let modelState = model.init()

// 狀態是純數據，可以序列化
console.log(modelState)
```

---

## 🔧 2. 優化器

### 舊 API
```typescript
import { Adam } from '@sylphx/optim'

// 創建優化器實例
const optimizer = new Adam(model.parameters(), {
  lr: 0.01,
  beta1: 0.9,
  beta2: 0.999
})

// 狀態隱藏在類別內
optimizer.step()
```

### 新 API ✅
```typescript
import { Adam } from '@sylphx/optim'
import { getParams } from '@sylphx/train'

// 優化器工廠
const optimizer = Adam({
  lr: 0.01,
  beta1: 0.9,
  beta2: 0.999
})

// 明確初始化
let optState = optimizer.init(getParams(modelState))

// 明確的狀態更新
const result = optimizer.step(params, grads, optState)
optState = result.optState
params = result.params
```

---

## 🎓 3. 訓練循環

### 舊 API
```typescript
// 需要手動管理所有細節
for (let epoch = 0; epoch < 1000; epoch++) {
  // Forward
  const output = model.forward(x)

  // Loss
  const loss = mse(output, y)

  // Backward
  optimizer.zeroGrad()
  loss.backward()

  // Update (狀態隱藏)
  optimizer.step()
}
```

### 新 API ✅
```typescript
import { trainStep } from '@sylphx/train'

for (let epoch = 0; epoch < 1000; epoch++) {
  const result = trainStep({
    model,
    modelState,
    optimizer,
    optState,
    input: x,
    target: y,
    lossFn: mse
  })

  // 明確更新狀態
  modelState = result.modelState
  optState = result.optState

  console.log(`Loss: ${result.loss}`)
}
```

---

## 🏗️ 4. DQN Agent (2048 遊戲)

### 舊 API
```typescript
// ❌ 286 行代碼

// 定義網絡結構
export type QNetwork = {
  linear1: nn.LinearState
  linear2: nn.LinearState
  linear3: nn.LinearState
}

// 手動初始化每一層
export function initNetwork(): QNetwork {
  return {
    linear1: nn.linear.init(16, 64),
    linear2: nn.linear.init(64, 64),
    linear3: nn.linear.init(64, 4),
  }
}

// 手動前向傳播
export function forward(state: number[], network: QNetwork): Tensor {
  const input = tensor([state], { requiresGrad: false })
  let h = nn.linear.forward(input, network.linear1)
  h = F.relu(h)
  h = nn.linear.forward(h, network.linear2)
  h = F.relu(h)
  h = nn.linear.forward(h, network.linear3)
  return h
}

// 手動獲取參數
function getNetworkParams(network: QNetwork): Tensor[] {
  return [
    network.linear1.weight,
    network.linear1.bias,
    network.linear2.weight,
    network.linear2.bias,
    network.linear3.weight,
    network.linear3.bias,
  ]
}

// 手動訓練步驟
const loss = F.mse(qValuesBatch, target)
const grads = T.backward(loss)
const result = optim.adam.step(agent.optimizer, getNetworkParams(agent.network), grads)

// 手動重建網絡
const newNetwork: QNetwork = {
  linear1: {
    weight: result.params[0]!,
    bias: result.params[1]!,
  },
  linear2: {
    weight: result.params[2]!,
    bias: result.params[3]!,
  },
  linear3: {
    weight: result.params[4]!,
    bias: result.params[5]!,
  },
}
```

### 新 API ✅
```typescript
// ✅ 234 行代碼 (-18%)

import { Sequential, Linear, ReLU } from '@sylphx/nn'
import { Adam } from '@sylphx/optim'
import { getParams, trainStep } from '@sylphx/train'

// 使用 Sequential 組合
const createQNetwork = () => Sequential(
  Linear(16, 64),
  ReLU(),
  Linear(64, 64),
  ReLU(),
  Linear(64, 4)
)

// 簡單初始化
const model = createQNetwork()
const modelState = model.init()
const optimizer = Adam({ lr: 0.001 })
const optState = optimizer.init(getParams(modelState))

// 自動前向傳播
const qValues = model.forward(input, modelState)

// 使用 trainStep 自動處理所有細節
const result = trainStep({
  model,
  modelState,
  optimizer,
  optState,
  input: statesTensor,
  target: target,
  lossFn: F.mse
})

// 簡單更新
modelState = result.modelState
optState = result.optState
```

---

## 📊 5. 完整 XOR 例子對比

### 舊 API
```typescript
import { Linear, Sequential, Tanh } from '@sylphx/nn'
import { Adam } from '@sylphx/optim'
import { mse } from '@sylphx/functional'

// 數據
const x = tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
const y = tensor([[0], [1], [1], [0]])

// 模型
const model = new Sequential([
  new Linear(2, 8),
  new Tanh(),
  new Linear(8, 1)
])

// 優化器
const optimizer = new Adam(model.parameters(), { lr: 0.05 })

// 訓練
for (let epoch = 0; epoch < 3000; epoch++) {
  const output = model.forward(x)
  const loss = mse(output, y)

  optimizer.zeroGrad()
  loss.backward()
  optimizer.step()

  if (epoch % 500 === 0) {
    console.log(`Epoch ${epoch}, Loss: ${loss.item()}`)
  }
}

// 測試
const pred = model.forward(tensor([[0, 1]]))
console.log(pred.item())
```

### 新 API ✅
```typescript
import { tensor } from '@sylphx/tensor'
import { Sequential, Linear, Tanh } from '@sylphx/nn'
import { Adam } from '@sylphx/optim'
import { mse } from '@sylphx/functional'
import { getParams, trainStep } from '@sylphx/train'

// 數據
const x = tensor([[0, 0], [0, 1], [1, 0], [1, 1]], { requiresGrad: true })
const y = tensor([[0], [1], [1], [0]], { requiresGrad: true })

// 模型
const model = Sequential(
  Linear(2, 8),
  Tanh(),
  Linear(8, 1)
)

// 初始化
let modelState = model.init()
const optimizer = Adam({ lr: 0.05 })
let optState = optimizer.init(getParams(modelState))

// 訓練
for (let epoch = 0; epoch < 3000; epoch++) {
  const result = trainStep({
    model,
    modelState,
    optimizer,
    optState,
    input: x,
    target: y,
    lossFn: mse
  })

  modelState = result.modelState
  optState = result.optState

  if (epoch % 500 === 0) {
    console.log(`Epoch ${epoch}, Loss: ${result.loss}`)
  }
}

// 測試
const pred = model.forward(tensor([[0, 1]]), modelState)
console.log(pred.data[0])
```

---

## 🎯 關鍵差異總結

| 特性 | 舊 API | 新 API |
|------|--------|--------|
| **類別** | ✅ 使用 `new` | ❌ 無類別 |
| **狀態** | 隱藏 | 明確 |
| **組合** | 數組 `[...]` | 函數 `Sequential(...)` |
| **不變性** | ❌ 可變 | ✅ 不可變 |
| **序列化** | 困難 | 簡單 |
| **測試** | 困難 (副作用) | 簡單 (純函數) |
| **代碼量** | 更多 | 更少 (-18%) |
| **可讀性** | 中等 | 高 |

---

## ✨ 新 API 優勢

1. **純函數** - 無副作用，易測試
2. **明確狀態** - 狀態可見、可控、可序列化
3. **不可變** - 函數式編程最佳實踐
4. **組合性** - 使用 Sequential 組合層
5. **簡潔** - trainStep 自動處理細節
6. **類型安全** - TypeScript 完整支援
7. **易理解** - 數據流向清晰

---

## 🚀 遷移建議

舊代碼已經不支援！所有項目必須遷移到新 API。

**步驟:**
1. 移除所有 `new` 關鍵字
2. 使用工廠函數: `Linear(2, 8)` 代替 `new Linear(2, 8)`
3. 明確管理狀態: `modelState`, `optState`
4. 使用 `trainStep` 簡化訓練
5. 使用 `Sequential` 組合層

**收益:**
- 代碼減少 ~20%
- 可讀性提升
- 更易維護
- 完全類型安全
