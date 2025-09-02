# YOLOv7-Tiny Baseline vs MultiHead (1B4H) 對比指南

## 快速開始

### 1. 一鍵對比（推薦）

```bash
# 執行完整對比
./run_comparison.sh
```

這會自動執行所有對比並生成報告。

### 2. 分步對比

#### 2.1 模型架構對比

```bash
# 對比模型結構、參數量、推理速度
python compare_baseline_vs_multihead.py \
    --img-size 320 \
    --batch-sizes 1 8 16 32 \
    --plot
```

**輸出**：
- `comparison_results.json` - 詳細對比數據
- `comparison_plots.png` - 視覺化圖表

#### 2.2 訓練結果對比

```bash
# 對比訓練曲線和驗證性能（需要已完成訓練）
python compare_training_results.py \
    --baseline runs/feasibility/baseline_be_optimized \
    --multihead runs/multihead/yolov7_tiny_1b4h_320
```

**輸出**：
- `training_curves_comparison.png` - 訓練曲線對比
- `comparison_report.md` - 完整報告

## 對比維度

### 1. 架構對比

| 維度 | Baseline | MultiHead | 說明 |
|------|----------|-----------|------|
| **檢測頭** | 1 個 | 4 個 | 每頭 20 類 |
| **參數量** | ~6.0M | ~6.9M | +15% |
| **檢測層參數** | 229K | 876K | +282% |
| **策略** | 單一頭 | Strategy A | 共享 box/obj |

### 2. 性能對比

| 指標 | 預期改進 | 實際測量方法 |
|------|----------|-------------|
| **mAP@0.5** | +2-3% | test.py 驗證 |
| **小物體** | +3-5% | COCO small 類別 |
| **推理速度** | -10-15% | FPS 測試 |
| **記憶體** | +15-20% | GPU 監控 |

### 3. 訓練對比

| 方面 | Baseline | MultiHead |
|------|----------|-----------|
| **收斂速度** | 標準 | 略慢 |
| **最終 loss** | 標準 | 略低 |
| **類別平衡** | 不均 | 更均衡 |

## 測試場景

### 場景 1：未訓練模型對比

```bash
# 只比較架構和推理速度
python compare_baseline_vs_multihead.py
```

**關注點**：
- 參數量差異
- 推理速度差異
- 記憶體使用

### 場景 2：已訓練模型對比

```bash
# 需要先完成訓練
# Baseline
./train_baseline.sh  # 100 epochs

# MultiHead
./train_multihead_320.sh  # 100 epochs

# 對比
./run_comparison.sh
```

**關注點**：
- mAP 提升
- 訓練曲線
- 類別性能

### 場景 3：快速驗證對比

```bash
# 使用少量 epochs 快速驗證
python train.py --epochs 10 --cfg cfg/training/yolov7-tiny.yaml ...
python train.py --epochs 10 --cfg cfg/training/yolov7-tiny-multihead-proper.yaml ...

# 對比結果
python compare_training_results.py
```

## 關鍵指標解讀

### 1. 參數增加 vs 性能提升

```
參數增加: +15%
預期 mAP 提升: +2-3%
ROI = mAP提升 / 參數增加 = 2.5% / 15% = 0.167
```

**解讀**：每增加 1% 參數，獲得 0.167% mAP 提升

### 2. 速度損失 vs 精度提升

```
FPS 降低: -10-15%
mAP 提升: +2-3%
權衡比 = mAP提升 / FPS損失 = 2.5% / 12.5% = 0.2
```

**解讀**：犧牲 1% 速度，獲得 0.2% 精度

### 3. 類別平衡改進

MultiHead 將 80 類分為 4 組：
- Head 0: 人物與運動 (高頻)
- Head 1: 交通工具 (中頻)
- Head 2: 動物食物 (中頻)
- Head 3: 家具電子 (低頻)

**預期改進**：
- 低頻類別 AP 提升 3-5%
- 高頻類別 AP 維持
- 整體更均衡

## 視覺化結果

### 1. 架構對比圖

```python
# 生成對比圖表
python compare_baseline_vs_multihead.py --plot
```

包含：
- 參數量柱狀圖
- FPS 曲線圖
- 記憶體使用圖
- 相對性能圖

### 2. 訓練曲線對比

```python
# 需要完成訓練
python compare_training_results.py
```

包含：
- Loss 曲線 (box, obj, cls)
- mAP 曲線
- Precision/Recall 曲線

## 實驗建議

### 初次測試

1. 先跑架構對比，確認實現正確
2. 訓練 10 epochs 快速驗證
3. 如果趨勢正確，訓練 100 epochs

### 完整實驗

1. Baseline: 100 epochs
2. MultiHead: 100 epochs
3. 全面對比分析
4. 根據結果調整超參數

### 消融實驗

測試不同配置的影響：
- 2 heads vs 4 heads
- 不同類別分組策略
- 不同 anchor_t 值 (3.0, 3.5, 4.0)

## 常見問題

### Q1: MultiHead 沒有提升？

檢查：
- 類別分組是否合理
- 頭權重是否平衡
- 損失權重是否適當

### Q2: 速度下降太多？

優化：
- 減少頭數量 (4→2)
- 使用 TensorRT 優化
- 調整 batch size

### Q3: 記憶體不足？

解決：
- 降低 batch size
- 使用梯度累積
- 使用混合精度訓練

## 總結

| 使用場景 | 推薦配置 |
|----------|----------|
| **精度優先** | MultiHead |
| **速度優先** | Baseline |
| **平衡** | MultiHead + TensorRT |
| **邊緣設備** | Baseline |
| **伺服器** | MultiHead |

## 相關文件

- `TRAINING_COMPARISON.md` - 訓練指令對比
- `yolov7_1B4H PRD v3.5.md` - 產品需求文檔
- `yolov7_1B4H SDD v2.3.md` - 系統設計文檔