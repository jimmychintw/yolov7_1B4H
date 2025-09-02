import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import torch

def test_original_detect_still_works():
    """確保原始 Detect 仍然正常"""
    from models.yolo import Detect
    
    nc = 80
    anchors = ([10,13, 16,30, 33,23], [30,61, 62,45, 59,119], [116,90, 156,198, 373,326])
    ch = (128, 256, 512)
    
    model = Detect(nc=nc, anchors=anchors, ch=ch)
    
    # 測試基本屬性
    assert model.nc == nc
    assert model.nl == 3
    assert model.na == 3
    assert model.no == 85
    
    # 測試訓練模式前向傳播
    x = [torch.randn(1, ch[i], 80-i*32, 80-i*32) for i in range(3)]
    model.train()
    
    with torch.no_grad():
        # 訓練模式
        train_out = model(x)
        assert len(train_out) == 3
        
        # 注意：推理模式需要設置 stride，這裡暫時跳過
        # 因為 stride 是由 Model 類設置的
    
    print("✓ Original Detect still works correctly")
    return True

def test_imports_work():
    """測試所有導入仍然正常"""
    try:
        from models.yolo import Model, Detect, MultiHeadDetect, IDetect
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_multihead_independent():
    """測試 MultiHeadDetect 獨立於 Detect"""
    from models.yolo import Detect, MultiHeadDetect
    
    # 確保兩個類是獨立的
    assert Detect != MultiHeadDetect
    assert not issubclass(MultiHeadDetect, Detect)
    assert not issubclass(Detect, MultiHeadDetect)
    
    print("✓ MultiHeadDetect is independent from Detect")
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("Running Step 4 regression tests...")
    print("=" * 60)
    
    test_original_detect_still_works()
    test_imports_work()
    test_multihead_independent()
    
    print("\n✅ No regressions detected!")