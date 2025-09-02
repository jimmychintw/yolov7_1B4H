#!/bin/bash
# Quick training test for MultiHead

python train.py \
    --weights '' \
    --cfg cfg/training/yolov7-tiny-multihead-proper.yaml \
    --data data/coco-multihead.yaml \
    --hyp data/hyp.scratch.tiny.yaml \
    --epochs 1 \
    --batch-size 4 \
    --img-size 640 \
    --device cpu \
    --workers 1 \
    --name multihead_test \
    --exist-ok

echo "Training test completed!"