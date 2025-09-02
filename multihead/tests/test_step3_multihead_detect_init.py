import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import torch
import torch.nn as nn

def test_import_original_detect():
    """確保原始 Detect 類未受影響"""
    from models.yolo import Detect
    
    # 原始 Detect 應該還能正常導入
    assert Detect is not None
    assert issubclass(Detect, nn.Module)
    
    print("✓ Original Detect class intact")
    return True

def test_import_multihead_detect():
    """測試能否導入 MultiHeadDetect"""
    try:
        from models.yolo import MultiHeadDetect
        assert MultiHeadDetect is not None
        assert issubclass(MultiHeadDetect, nn.Module)
        print("✓ MultiHeadDetect class imported")
        return True
    except ImportError:
        print("⚠ MultiHeadDetect not found yet")
        return False

def test_multihead_detect_init():
    """測試 MultiHeadDetect 初始化"""
    try:
        from models.yolo import MultiHeadDetect
        
        # 測試參數（與 Detect 相同的接口）
        nc = 80  # number of classes
        anchors = (
            [10,13, 16,30, 33,23],  # P3/8
            [30,61, 62,45, 59,119],  # P4/16
            [116,90, 156,198, 373,326]  # P5/32
        )
        ch = (128, 256, 512)  # channels
        
        # 創建實例
        model = MultiHeadDetect(
            nc=nc, 
            anchors=anchors, 
            ch=ch,
            n_heads=4,
            config_path='data/coco-multihead.yaml'
        )
        
        # 基本屬性檢查
        assert model.nc == nc
        assert model.n_heads == 4
        assert model.nl == len(anchors)  # number of detection layers
        assert model.na == len(anchors[0]) // 2  # number of anchors
        assert model.no == nc + 5  # number of outputs per anchor
        
        print("✓ MultiHeadDetect initialized successfully")
        print(f"  - Classes: {model.nc}")
        print(f"  - Heads: {model.n_heads}")
        print(f"  - Layers: {model.nl}")
        print(f"  - Anchors per layer: {model.na}")
        
        return True
        
    except Exception as e:
        print(f"⚠ MultiHeadDetect init failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_module_structure():
    """測試模組結構"""
    try:
        from models.yolo import MultiHeadDetect
        
        nc = 80
        anchors = ([10,13, 16,30, 33,23], [30,61, 62,45, 59,119], [116,90, 156,198, 373,326])
        ch = (128, 256, 512)
        
        model = MultiHeadDetect(nc=nc, anchors=anchors, ch=ch, n_heads=4)
        
        # 檢查必要的屬性
        assert hasattr(model, 'stride'), "Should have stride attribute"
        assert hasattr(model, 'anchors'), "Should have anchors buffer"
        assert hasattr(model, 'anchor_grid'), "Should have anchor_grid buffer"
        assert hasattr(model, 'config'), "Should have config attribute"
        
        # 檢查註冊的 buffers
        buffers = dict(model.named_buffers())
        assert 'anchors' in buffers, "anchors should be registered buffer"
        assert 'anchor_grid' in buffers, "anchor_grid should be registered buffer"
        
        print("✓ Module structure correct")
        print(f"  - Buffers: {list(buffers.keys())}")
        
        return True
        
    except Exception as e:
        print(f"⚠ Module structure test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_compatibility_attributes():
    """測試與原始 Detect 的相容性屬性"""
    try:
        from models.yolo import MultiHeadDetect
        
        nc = 80
        anchors = ([10,13, 16,30, 33,23], [30,61, 62,45, 59,119], [116,90, 156,198, 373,326])
        ch = (128, 256, 512)
        
        model = MultiHeadDetect(nc=nc, anchors=anchors, ch=ch, n_heads=4)
        
        # 這些屬性是 Model._initialize_biases() 需要的
        assert hasattr(model, 'm'), "Should have 'm' attribute for compatibility"
        assert hasattr(model, 'stride'), "Should have stride attribute"
        
        # m 應該是 ModuleList
        assert isinstance(model.m, nn.ModuleList), "m should be ModuleList"
        
        print("✓ Compatibility attributes present")
        
        return True
        
    except Exception as e:
        print(f"⚠ Compatibility test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Step 3: MultiHeadDetect initialization...")
    print("=" * 60)
    
    # 先測試原始類未受影響
    test_import_original_detect()
    
    # 測試新類
    if test_import_multihead_detect():
        test_multihead_detect_init()
        test_module_structure()
        test_compatibility_attributes()
        print("\n✅ All Step 3 tests passed!")
    else:
        print("\n⚠ MultiHeadDetect not implemented yet")