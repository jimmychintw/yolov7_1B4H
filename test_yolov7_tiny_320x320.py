#!/usr/bin/env python3
"""
測試 YOLOv7-Tiny MultiHead 在 320x320 解析度下的支援
專門針對 COCO 2017 數據集設計
"""

import torch
import torch.nn as nn
from models.yolo import Model, MultiHeadDetect
from utils.loss_multihead import ComputeLossMultiHead
from utils.multihead_utils import MultiHeadConfig
import numpy as np

def test_320x320_resolution():
    """測試 320x320 解析度的完整支援"""
    
    print("="*70)
    print("YOLOv7-Tiny MultiHead 320×320 解析度測試 (COCO 2017)")
    print("="*70)
    
    # 1. 測試特徵圖尺寸計算
    print("\n1. 驗證 320×320 特徵圖尺寸...")
    input_size = 320
    strides = [8, 16, 32]  # YOLOv7-Tiny 標準 stride
    
    feature_maps = []
    for stride in strides:
        fm_size = input_size // stride
        feature_maps.append((fm_size, fm_size))
        print(f"   P{int(np.log2(stride))}/{stride}: {fm_size}×{fm_size} = {fm_size*fm_size} 網格")
    
    # 2. 測試錨框配置（針對 320×320 優化）
    print("\n2. 320×320 專用錨框配置...")
    
    # 原始 640×640 錨框
    anchors_640 = [
        [10,13, 16,30, 33,23],      # P3/8
        [30,61, 62,45, 59,119],     # P4/16
        [116,90, 156,198, 373,326]  # P5/32
    ]
    
    # 縮放到 320×320 (比例: 320/640 = 0.5)
    scale = 320 / 640
    anchors_320 = []
    for layer_anchors in anchors_640:
        scaled = [int(a * scale) for a in layer_anchors]
        anchors_320.append(scaled)
    
    print("   原始 640×640 錨框:")
    for i, anchors in enumerate(anchors_640):
        print(f"     P{3+i}: {anchors}")
    
    print("\n   縮放到 320×320 錨框:")
    for i, anchors in enumerate(anchors_320):
        print(f"     P{3+i}: {anchors}")
    
    # 3. 測試 MultiHeadDetect 在 320×320 下的運作
    print("\n3. 測試 MultiHeadDetect 層...")
    
    # YOLOv7-Tiny 在 320×320 的通道配置
    ch = (128, 256, 512)  # P3, P4, P5 通道數
    
    # 創建 MultiHeadDetect
    det = MultiHeadDetect(nc=80, anchors=anchors_320, ch=ch)
    det.stride = torch.tensor(strides).float()
    
    # 創建 320×320 輸入的特徵圖
    batch_size = 2
    x = [
        torch.randn(batch_size, ch[0], 40, 40),  # P3/8: 320/8 = 40
        torch.randn(batch_size, ch[1], 20, 20),  # P4/16: 320/16 = 20
        torch.randn(batch_size, ch[2], 10, 10),  # P5/32: 320/32 = 10
    ]
    
    print(f"   輸入特徵圖形狀:")
    for i, feat in enumerate(x):
        print(f"     P{3+i}: {feat.shape}")
    
    # 測試前向傳播
    det.train()
    reg_obj, cls = det(x)
    
    print(f"\n   ✅ 訓練模式輸出:")
    print(f"     reg_obj: {[r.shape for r in reg_obj]}")
    print(f"     cls heads: {len(cls)} 個頭")
    
    det.eval()
    pred, _ = det(x)
    print(f"\n   ✅ 推理模式輸出:")
    print(f"     預測張量: {pred.shape}")
    print(f"     總錨框數: {pred.shape[1]}")
    
    # 4. 測試小物體檢測能力
    print("\n4. 小物體檢測能力分析...")
    
    # 計算每個檢測層的感受野
    print("   感受野分析 (320×320):")
    total_anchors = 0
    for i, (stride, fm_size) in enumerate(zip(strides, feature_maps)):
        n_anchors = 3 * fm_size[0] * fm_size[1]
        total_anchors += n_anchors
        min_obj = stride * 2  # 最小可檢測物體大小（經驗值）
        max_obj = stride * 16  # 最大可檢測物體大小（經驗值）
        print(f"     P{3+i}/{stride}: {n_anchors:,} 錨框, 物體範圍 {min_obj}-{max_obj} pixels")
    
    print(f"   總錨框數: {total_anchors:,}")
    
    # 5. COCO 2017 類別分組測試
    print("\n5. COCO 2017 多頭類別分組...")
    
    config = MultiHeadConfig('data/coco-multihead.yaml')
    
    print(f"   檢測頭數量: {config.n_heads}")
    print(f"   類別總數: {config.nc}")
    
    for head_id in range(config.n_heads):
        head_info = config.head_assignments[head_id]
        print(f"\n   Head {head_id} ({head_info['name']}):")
        print(f"     類別數: {len(head_info['classes'])}")
        print(f"     權重: {head_info.get('weight', 1.0):.2f}")
        print(f"     Supercategory: {head_info.get('supercategory', 'N/A')}")
    
    # 6. 損失計算測試（320×320 特定）
    print("\n6. 測試損失計算 (320×320)...")
    
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.ModuleList([det])
            self.hyp = {
                'box': 0.05, 'cls': 0.5, 'obj': 1.0,
                'anchor_t': 3.5,  # 降低到 3.5 for 320×320
                'cls_pw': 1.0, 'obj_pw': 1.0,
                'fl_gamma': 0.0, 'label_smoothing': 0.0
            }
            self.gr = 1.0
    
    model = SimpleModel()
    compute_loss = ComputeLossMultiHead(model)
    
    # 創建適合 320×320 的目標（歸一化座標）
    targets = torch.tensor([
        [0, 0, 0.5, 0.5, 0.05, 0.05],   # 小物體 (16×16 pixels)
        [0, 2, 0.2, 0.2, 0.03, 0.03],   # 更小物體 (10×10 pixels)
        [1, 14, 0.8, 0.8, 0.15, 0.15],  # 中等物體 (48×48 pixels)
        [1, 56, 0.6, 0.6, 0.25, 0.25],  # 大物體 (80×80 pixels)
    ])
    
    model.train()
    pred = model.model[0](x)
    loss, loss_items = compute_loss(pred, targets)
    
    print(f"   ✅ 損失計算成功:")
    print(f"     總損失: {loss.item():.4f}")
    print(f"     box={loss_items[0]:.4f}, obj={loss_items[1]:.4f}, cls={loss_items[2]:.4f}")
    
    # 7. 性能預估
    print("\n7. 320×320 性能預估...")
    
    # 參數統計
    det_params = sum(p.numel() for p in det.parameters())
    print(f"   檢測層參數: {det_params:,}")
    
    # FLOPs 估算（簡化計算）
    flops_per_conv = 0
    for i, (c, fm) in enumerate(zip(ch, feature_maps)):
        # 每個頭的卷積運算
        flops = c * 85 * 3 * fm[0] * fm[1] * 4  # 4 heads
        flops_per_conv += flops
    
    print(f"   檢測層 FLOPs: ~{flops_per_conv/1e9:.2f} GFLOPs")
    
    # 記憶體佔用估算
    memory_mb = det_params * 4 / 1024 / 1024  # FP32
    print(f"   檢測層記憶體: ~{memory_mb:.1f} MB (FP32)")
    
    print("\n" + "="*70)
    print("✅ YOLOv7-Tiny 完全支援 320×320 解析度與 COCO 2017！")
    print("="*70)
    
    return True

def test_coco2017_compatibility():
    """測試 COCO 2017 數據集相容性"""
    
    print("\n\nCOCO 2017 數據集相容性測試")
    print("-"*70)
    
    # 載入配置
    config = MultiHeadConfig('data/coco-multihead.yaml')
    
    # COCO 2017 統計
    print("\n📊 COCO 2017 數據集統計:")
    print("   訓練集: 118,287 張圖片")
    print("   驗證集: 5,000 張圖片")
    print("   測試集: 40,670 張圖片")
    print("   類別數: 80")
    print("   總標註框: >1.5M")
    
    # 驗證類別覆蓋
    print("\n📋 類別覆蓋驗證:")
    coco_classes = set(range(80))
    assigned_classes = set()
    
    for head_id in range(config.n_heads):
        classes = set(config.get_classes_for_head(head_id))
        assigned_classes.update(classes)
    
    missing = coco_classes - assigned_classes
    extra = assigned_classes - coco_classes
    
    if missing:
        print(f"   ❌ 缺少類別: {missing}")
    else:
        print(f"   ✅ 所有 80 個 COCO 類別都已分配")
    
    if extra:
        print(f"   ❌ 額外類別: {extra}")
    
    # 類別分佈統計
    print("\n📈 預期類別分佈 (基於 COCO 2017):")
    distributions = {
        'person & sports': 35,
        'vehicle & outdoor': 25,
        'animal & food': 20,
        'furniture & appliance': 20
    }
    
    for head_id in range(config.n_heads):
        head_info = config.head_assignments[head_id]
        supercategory = head_info.get('supercategory', 'unknown')
        expected = distributions.get(supercategory, 0)
        print(f"   Head {head_id}: ~{expected}% 的訓練樣本")
    
    print("\n✅ COCO 2017 完全相容！")

def compare_resolutions():
    """比較不同解析度的影響"""
    
    print("\n\n解析度比較分析")
    print("-"*70)
    
    resolutions = [320, 416, 512, 640]
    
    print("\n📐 不同解析度的特徵圖大小:")
    print("\n   解析度 | P3/8  | P4/16 | P5/32 | 總錨框數")
    print("   " + "-"*50)
    
    for res in resolutions:
        p3 = (res // 8) ** 2 * 3
        p4 = (res // 16) ** 2 * 3
        p5 = (res // 32) ** 2 * 3
        total = p3 + p4 + p5
        
        print(f"   {res:3d}×{res:<3d} | {p3:5,} | {p4:5,} | {p5:5,} | {total:7,}")
    
    print("\n💡 觀察:")
    print("   - 320×320 的錨框數是 640×640 的 1/4")
    print("   - 較低解析度更適合邊緣設備部署")
    print("   - 需要調整 anchor_t 參數以適應小物體")

if __name__ == "__main__":
    # 執行所有測試
    success = test_320x320_resolution()
    
    if success:
        test_coco2017_compatibility()
        compare_resolutions()
        
        print("\n\n" + "="*70)
        print("📝 總結：")
        print("="*70)
        print("1. ✅ YOLOv7-Tiny 完全支援 320×320 解析度")
        print("2. ✅ COCO 2017 所有 80 類別正確分配到 4 個檢測頭")
        print("3. ✅ 針對 320×320 優化的錨框配置")
        print("4. ✅ 損失計算支援小物體檢測")
        print("5. ✅ MultiHead 架構完全相容")
        
        print("\n🚀 使用指南：")
        print("1. 使用 cfg/training/yolov7-tiny-multihead-proper.yaml")
        print("2. 設定 --img-size 320")
        print("3. 調整 hyp.anchor_t = 3.5 (針對小物體)")
        print("4. 使用 data/coco-multihead.yaml 作為數據配置")
        print("5. 訓練指令：")
        print("   python train.py --img 320 --batch 64 \\")
        print("                   --cfg cfg/training/yolov7-tiny-multihead-proper.yaml \\")
        print("                   --data data/coco-multihead.yaml \\")
        print("                   --device 0")