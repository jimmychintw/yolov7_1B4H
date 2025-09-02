# YOLOv7-Tiny 訓練指令對比

## 原始版本 vs MultiHead (1B4H) 版本

### 1. 原始 YOLOv7-Tiny 訓練指令

```bash
python train.py \
  --img-size 320 320 \
  --batch-size 384 \
  --epochs 100 \
  --data data/coco_vast.ai.yaml \
  --cfg cfg/training/yolov7-tiny.yaml \
  --weights '' \
  --hyp data/hyp.scratch.tiny.bs384.yaml \
  --device 0 \
  --workers 8 \
  --save_period 25 \
  --project runs/feasibility \
  --name baseline_be_optimized \
  --exist-ok \
  --noautoanchor \
  --cache-images
```

### 2. MultiHead (1B4H) 版本訓練指令

```bash
python train.py \
  --img-size 320 320 \
  --batch-size 384 \
  --epochs 100 \
  --data data/coco-multihead.yaml \
  --cfg cfg/training/yolov7-tiny-multihead-proper.yaml \
  --weights '' \
  --hyp data/hyp.scratch.tiny.multihead.320.yaml \
  --device 0 \
  --workers 8 \
  --save_period 25 \
  --project runs/multihead \
  --name yolov7_tiny_1b4h_320 \
  --exist-ok \
  --noautoanchor \
  --cache-images
```

## 主要差異對照表

| 參數 | 原始版本 | MultiHead 版本 | 說明 |
|------|---------|---------------|------|
| `--data` | `data/coco_vast.ai.yaml` | `data/coco-multihead.yaml` | MultiHead 包含 4 頭類別分配 |
| `--cfg` | `cfg/training/yolov7-tiny.yaml` | `cfg/training/yolov7-tiny-multihead-proper.yaml` | 使用 MultiHeadDetect 層 |
| `--hyp` | `data/hyp.scratch.tiny.bs384.yaml` | `data/hyp.scratch.tiny.multihead.320.yaml` | 針對 320×320 和 MultiHead 優化 |
| `--project` | `runs/feasibility` | `runs/multihead` | 區分實驗結果 |
| `--name` | `baseline_be_optimized` | `yolov7_tiny_1b4h_320` | 更明確的實驗名稱 |

## 超參數調整（針對 320×320）

### MultiHead 特定調整：
- **anchor_t**: 4.0 → 3.5（適應小物體）
- **mosaic**: 1.0 → 0.5（減少拼接，提高小物體檢測）
- **loss_ota**: 1 → 0（使用 ComputeLossMultiHead）

### 新增 MultiHead 參數：
```yaml
head_weight_decay: 0.99  # 動態頭權重調整衰減
head_weight_momentum: 0.9  # 頭權重更新動量
min_head_weight: 0.1  # 最小頭權重
max_head_weight: 0.4  # 最大頭權重
```

## 預期改進

| 指標 | 原始 YOLOv7-Tiny | MultiHead (1B4H) | 預期改進 |
|------|-----------------|------------------|----------|
| mAP@0.5 | Baseline | Baseline + 2-3% | +2-3% |
| 小物體檢測 | 標準 | 增強 | +3-5% |
| 類別平衡 | 單一頭 | 4 頭平衡 | 更均衡 |
| 參數量 | 6.0M | ~6.9M | +15% |
| FPS (RTX 3090) | 100% | ~85-90% | -10-15% |

## 注意事項

1. **數據路徑**：
   - 如果使用自定義 COCO 路徑，需要修改 `data/coco-multihead.yaml` 中的路徑
   - 原始的 `data/coco_vast.ai.yaml` 需要複製相關路徑設定

2. **GPU 記憶體**：
   - Batch size 384 在 320×320 下約需 20-24GB VRAM
   - 如果記憶體不足，可降低到 256 或 128

3. **訓練監控**：
   - MultiHead 會在 log 中顯示每個頭的損失
   - 注意觀察 4 個頭的平衡情況

4. **檢查點保存**：
   - 每 25 個 epoch 保存一次
   - 最終模型在 `runs/multihead/yolov7_tiny_1b4h_320/weights/`

## 快速開始

```bash
# 1. 確保數據集路徑正確
# 編輯 data/coco-multihead.yaml，設定正確的 COCO 路徑

# 2. 執行訓練
./train_multihead_320.sh

# 3. 監控訓練
tensorboard --logdir runs/multihead
```

## 驗證指令

訓練完成後，使用以下指令驗證：

```bash
python test.py \
  --data data/coco-multihead.yaml \
  --img 320 \
  --batch 32 \
  --conf 0.001 \
  --iou 0.65 \
  --device 0 \
  --weights runs/multihead/yolov7_tiny_1b4h_320/weights/best.pt \
  --name yolov7_tiny_1b4h_320_val
```