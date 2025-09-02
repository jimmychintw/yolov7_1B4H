#!/bin/bash

# 自動化執行所有對比測試
# 用於比較 YOLOv7-Tiny Baseline vs MultiHead (1B4H)

echo "======================================"
echo "YOLOv7-Tiny Baseline vs MultiHead 對比"
echo "======================================"
echo ""

# 1. 模型架構和性能對比
echo "📊 執行模型架構和性能對比..."
python compare_baseline_vs_multihead.py \
    --baseline-cfg cfg/training/yolov7-tiny.yaml \
    --multihead-cfg cfg/training/yolov7-tiny-multihead-proper.yaml \
    --img-size 320 \
    --batch-sizes 1 8 16 32 \
    --num-runs 100 \
    --output comparison_results.json \
    --plot

echo ""
echo "✅ 模型對比完成"
echo ""

# 2. 訓練結果對比（如果有訓練結果）
if [ -d "runs/feasibility/baseline_be_optimized" ] && [ -d "runs/multihead/yolov7_tiny_1b4h_320" ]; then
    echo "📈 執行訓練結果對比..."
    python compare_training_results.py \
        --baseline runs/feasibility/baseline_be_optimized \
        --multihead runs/multihead/yolov7_tiny_1b4h_320
    
    echo ""
    echo "✅ 訓練結果對比完成"
else
    echo "⚠️  未找到訓練結果，跳過訓練對比"
fi

echo ""
echo "======================================"
echo "對比分析完成！"
echo "======================================"
echo ""
echo "📁 生成的文件："
echo "  - comparison_results.json (詳細數據)"
echo "  - comparison_plots.png (架構對比圖)"
echo "  - comparison_report.md (完整報告)"
echo ""
echo "如有訓練結果："
echo "  - training_curves_comparison.png (訓練曲線)"
echo "  - class_performance_comparison.png (類別性能)"
echo ""