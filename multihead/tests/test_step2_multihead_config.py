import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import torch
import yaml
import numpy as np

def test_multihead_config_basic():
    """測試 MultiHeadConfig 基本功能"""
    from utils.multihead_utils import MultiHeadConfig
    
    # 1. 測試載入配置
    config = MultiHeadConfig('data/coco-multihead.yaml')
    
    assert config.enabled == True
    assert config.n_heads == 4
    assert config.strategy == 'strategy_a'
    assert config.nc == 80
    
    print("✓ Configuration loaded successfully")
    print(f"  - Heads: {config.n_heads}")
    print(f"  - Strategy: {config.strategy}")
    
    return True

def test_head_masks():
    """測試類別遮罩生成"""
    from utils.multihead_utils import MultiHeadConfig
    
    config = MultiHeadConfig('data/coco-multihead.yaml')
    device = torch.device('cpu')
    
    # 測試每個頭的遮罩
    all_masked_classes = []
    for head_id in range(4):
        mask = config.get_head_mask(head_id, device)
        
        # 檢查遮罩維度
        assert mask.shape == (80,), f"Mask shape should be (80,), got {mask.shape}"
        assert mask.dtype == torch.bool, f"Mask should be bool, got {mask.dtype}"
        
        # 統計被遮罩的類別
        masked_indices = torch.where(mask)[0].tolist()
        all_masked_classes.extend(masked_indices)
        
        print(f"✓ Head {head_id} mask: {len(masked_indices)} classes")
    
    # 確保覆蓋所有80類且無重複
    assert len(all_masked_classes) == 80, f"Should cover 80 classes, got {len(all_masked_classes)}"
    assert len(set(all_masked_classes)) == 80, "Classes have duplicates!"
    
    print("✓ All classes covered without duplicates")
    
    return True

def test_head_class_mapping():
    """測試類別到頭的映射"""
    from utils.multihead_utils import MultiHeadConfig
    
    config = MultiHeadConfig('data/coco-multihead.yaml')
    
    # 測試一些已知的映射
    assert config.get_head_for_class(0) == 0  # person -> head 0
    assert config.get_head_for_class(1) == 1  # bicycle -> head 1
    assert config.get_head_for_class(14) == 2  # bird -> head 2
    assert config.get_head_for_class(56) == 3  # chair -> head 3
    
    print("✓ Class to head mapping correct")
    
    # 測試反向映射
    head0_classes = config.get_classes_for_head(0)
    assert 0 in head0_classes  # person should be in head 0
    assert len(head0_classes) == 20
    
    print("✓ Head to classes mapping correct")
    
    return True

def test_weight_normalization():
    """測試權重正規化"""
    from utils.multihead_utils import MultiHeadConfig
    
    config = MultiHeadConfig('data/coco-multihead.yaml')
    weights = config.get_head_weights()
    
    assert len(weights) == 4
    assert abs(sum(weights) - 1.0) < 1e-6, f"Weights should sum to 1.0, got {sum(weights)}"
    
    print(f"✓ Head weights normalized: {weights}")
    
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("Testing MultiHeadConfig...")
    print("=" * 60)
    
    tests = [
        test_multihead_config_basic,
        test_head_masks,
        test_head_class_mapping,
        test_weight_normalization
    ]
    
    for test in tests:
        try:
            test()
        except ImportError:
            print(f"⚠ MultiHeadConfig not found yet, creating it now...")
            break
        except Exception as e:
            print(f"✗ Test failed: {e}")
            import traceback
            traceback.print_exc()
            break
    else:
        print("\n✅ All Step 2 tests passed!")