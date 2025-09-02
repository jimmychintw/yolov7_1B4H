#!/bin/bash

# YOLOv7-Tiny MultiHead (1B4H) Training Script for 320x320
# Based on your original training command with MultiHead adjustments

echo "=================================="
echo "YOLOv7-Tiny MultiHead (1B4H) Training"
echo "Resolution: 320x320"
echo "Batch Size: 384"
echo "=================================="

# Original command (for reference):
# python train.py \
#   --img-size 320 320 \
#   --batch-size 384 \
#   --epochs 100 \
#   --data data/coco_vast.ai.yaml \
#   --cfg cfg/training/yolov7-tiny.yaml \
#   --weights '' \
#   --hyp data/hyp.scratch.tiny.bs384.yaml \
#   --device 0 \
#   --workers 8 \
#   --save_period 25 \
#   --project runs/feasibility \
#   --name baseline_be_optimized \
#   --exist-ok \
#   --noautoanchor \
#   --cache-images

# MultiHead version:
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

# 主要變更說明：
# 1. --data: 改用 data/coco-multihead.yaml (包含 4 頭類別分配)
# 2. --cfg: 改用 cfg/training/yolov7-tiny-multihead-proper.yaml (MultiHeadDetect)
# 3. --hyp: 改用 data/hyp.scratch.tiny.multihead.320.yaml (針對 320x320 優化)
# 4. --project: 改為 runs/multihead (區分原始和 MultiHead 實驗)
# 5. --name: 改為 yolov7_tiny_1b4h_320 (更明確的名稱)