# NeuronLine Demo

Demo 展示 NeuronLine 嘅核心功能同性能。

## 運行 Demo

```bash
cd apps/demo
bun run dev
```

## Demo 內容

### 1. 點擊預測 (Click Prediction)
- 訓練模型學習用戶點擊行為
- 根據元素位置、類型、設備等預測點擊概率
- 展示準確率同預測結果

### 2. 序列預測 (Sequence Prediction)
- 學習用戶導航路徑
- 預測下一步最可能嘅行動
- 應用於購物流程優化

### 3. 性能測試 (Performance Benchmark)
- 測試預測速度（prediction latency）
- 測試學習速度（learning throughput）
- 驗證輕量級高效目標

## 預期結果

- **預測速度**: < 0.1ms per prediction
- **學習速度**: < 0.5ms per update
- **準確率**: > 80% after training
- **吞吐量**: > 10,000 predictions/sec
