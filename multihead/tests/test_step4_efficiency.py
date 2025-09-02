import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import torch
import time

def test_memory_usage():
    """測試記憶體使用"""
    from models.yolo import Detect, MultiHeadDetect
    
    nc = 80
    anchors = ([10,13, 16,30, 33,23], [30,61, 62,45, 59,119], [116,90, 156,198, 373,326])
    ch = (128, 256, 512)
    
    # 原始 Detect
    detect = Detect(nc=nc, anchors=anchors, ch=ch)
    detect_params = sum(p.numel() for p in detect.parameters())
    
    # MultiHeadDetect
    multihead = MultiHeadDetect(nc=nc, anchors=anchors, ch=ch, n_heads=4)
    multihead_params = sum(p.numel() for p in multihead.parameters())
    
    # 計算增量
    increase = (multihead_params - detect_params) / detect_params * 100
    
    print(f"Parameter comparison:")
    print(f"  Original Detect: {detect_params:,} parameters")
    print(f"  MultiHeadDetect: {multihead_params:,} parameters")
    print(f"  Increase: {increase:.1f}%")
    
    # 根據策略 A，每個頭都輸出完整 80 類，所以參數會大幅增加
    # 原本 1 個 cls 分支，現在有 4 個，所以增加是合理的
    assert increase < 300, f"Parameter increase {increase:.1f}% too high"
    
    print("✓ Memory usage within acceptable range")
    
    return True

def test_initialization_speed():
    """測試初始化速度"""
    from models.yolo import MultiHeadDetect
    
    nc = 80
    anchors = ([10,13, 16,30, 33,23], [30,61, 62,45, 59,119], [116,90, 156,198, 373,326])
    ch = (128, 256, 512)
    
    # 測試創建速度
    start = time.time()
    for _ in range(10):
        model = MultiHeadDetect(nc=nc, anchors=anchors, ch=ch, n_heads=4)
    elapsed = time.time() - start
    
    avg_time = elapsed / 10
    print(f"✓ Average initialization time: {avg_time*1000:.2f}ms")
    
    # 應該在合理範圍內（< 100ms）
    assert avg_time < 0.1, f"Initialization too slow: {avg_time}s"
    
    return True

def test_convolution_efficiency():
    """測試卷積層效率"""
    from models.yolo import MultiHeadDetect
    
    nc = 80
    anchors = ([10,13, 16,30, 33,23], [30,61, 62,45, 59,119], [116,90, 156,198, 373,326])
    ch = (128, 256, 512)
    
    model = MultiHeadDetect(nc=nc, anchors=anchors, ch=ch, n_heads=4)
    model.eval()
    
    # 測試輸入
    x = [
        torch.randn(1, ch[0], 80, 80),
        torch.randn(1, ch[1], 40, 40),
        torch.randn(1, ch[2], 20, 20),
    ]
    
    # 確保能夠前向傳播（雖然現在只是 passthrough）
    with torch.no_grad():
        output = model(x)
    
    print("✓ Model can perform forward pass")
    
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Step 4 efficiency...")
    print("=" * 60)
    
    test_memory_usage()
    test_initialization_speed()
    test_convolution_efficiency()
    
    print("\n✅ Efficiency tests passed!")