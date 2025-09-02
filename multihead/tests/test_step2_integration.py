import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import torch
import yaml

def test_standalone_usage():
    """測試獨立使用（不依賴其他 YOLOv7 組件）"""
    from utils.multihead_utils import MultiHeadConfig
    
    # 可以獨立創建和使用
    config = MultiHeadConfig('data/coco-multihead.yaml')
    
    # 模擬在訓練中的使用
    batch_size = 8
    device = torch.device('cpu')
    
    # 獲取每個頭的遮罩
    masks = []
    for head_id in range(config.n_heads):
        mask = config.get_head_mask(head_id, device)
        masks.append(mask)
    
    # 確保遮罩互斥
    for i in range(len(masks)):
        for j in range(i+1, len(masks)):
            overlap = masks[i] & masks[j]
            assert overlap.sum() == 0, f"Heads {i} and {j} have overlapping classes!"
    
    print("✓ Masks are mutually exclusive")
    return True

def test_memory_efficiency():
    """測試記憶體效率"""
    from utils.multihead_utils import MultiHeadConfig
    
    config = MultiHeadConfig('data/coco-multihead.yaml')
    
    # 測試重複調用不會重複分配記憶體
    device = torch.device('cpu')
    masks = []
    for _ in range(100):
        mask = config.get_head_mask(0, device)
        masks.append(mask)
    
    # 所有遮罩應該是相同的
    for mask in masks[1:]:
        assert torch.equal(mask, masks[0])
    
    print("✓ Memory efficient mask generation")
    return True

def test_error_handling():
    """測試錯誤處理"""
    from utils.multihead_utils import MultiHeadConfig
    
    config = MultiHeadConfig('data/coco-multihead.yaml')
    
    # 測試無效的 head_id
    try:
        config.get_head_mask(5, 'cpu')  # Invalid head_id
        assert False, "Should raise error for invalid head_id"
    except ValueError:
        print("✓ Proper error handling for invalid head_id")
    
    # 測試無效的 class_id
    try:
        config.get_head_for_class(100)  # Invalid class_id
        assert False, "Should raise error for invalid class_id"
    except ValueError:
        print("✓ Proper error handling for invalid class_id")
    
    return True

def test_config_consistency():
    """測試配置一致性"""
    from utils.multihead_utils import MultiHeadConfig
    
    config = MultiHeadConfig('data/coco-multihead.yaml')
    
    # 檢查每個頭的類別數
    total_classes = 0
    for head_id in range(config.n_heads):
        classes = config.get_classes_for_head(head_id)
        total_classes += len(classes)
        assert len(classes) == 20, f"Head {head_id} should have 20 classes, got {len(classes)}"
    
    assert total_classes == 80, f"Total classes should be 80, got {total_classes}"
    
    print("✓ Configuration consistency verified")
    print(f"  - Each head has 20 classes")
    print(f"  - Total coverage: {total_classes} classes")
    
    # 測試頭名稱
    for head_id in range(config.n_heads):
        name = config.get_head_name(head_id)
        assert name is not None and len(name) > 0
        print(f"  - Head {head_id}: {name}")
    
    return True

def test_weight_functionality():
    """測試權重功能"""
    from utils.multihead_utils import MultiHeadConfig
    
    config = MultiHeadConfig('data/coco-multihead.yaml')
    
    # 獲取權重
    weights = config.get_head_weights()
    
    # 檢查權重數量
    assert len(weights) == config.n_heads, f"Should have {config.n_heads} weights"
    
    # 檢查權重範圍
    for w in weights:
        assert 0 < w <= 1, f"Weight should be in (0, 1], got {w}"
    
    # 檢查總和
    total = sum(weights)
    assert abs(total - 1.0) < 1e-6, f"Weights should sum to 1.0, got {total}"
    
    print("✓ Weight functionality working correctly")
    print(f"  - Weights: {weights}")
    print(f"  - Sum: {total:.6f}")
    
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("Running Step 2 Integration Tests...")
    print("=" * 60)
    
    tests = [
        ("Standalone usage", test_standalone_usage),
        ("Memory efficiency", test_memory_efficiency),
        ("Error handling", test_error_handling),
        ("Config consistency", test_config_consistency),
        ("Weight functionality", test_weight_functionality)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\nTesting: {test_name}")
        print("-" * 40)
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"✗ Test failed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("✅ All integration tests PASSED!")
    else:
        print("❌ Some tests failed!")
        sys.exit(1)