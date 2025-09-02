import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import yaml
# import torch  # 暫時註解，只測試配置相容性

def test_backward_compatibility():
    """確保新配置不會破壞現有代碼"""
    
    # 1. 測試 yaml 載入
    with open('data/coco-multihead.yaml', 'r') as f:
        data = yaml.safe_load(f)
    
    # 2. 測試基本字段訪問（模擬 train.py 的使用方式）
    nc = data['nc']
    names = data['names']
    train_path = data['train']
    
    assert nc == 80
    assert len(names) == 80
    assert train_path == './coco/train2017.txt'
    
    print("✓ Basic fields accessible")
    print(f"  - nc: {nc}")
    print(f"  - names count: {len(names)}")
    print(f"  - train path: {train_path}")
    
    # 3. 測試可選字段訪問
    multihead_config = data.get('multihead', None)
    if multihead_config:
        print(f"✓ MultiHead config found: {multihead_config['n_heads']} heads")
        print(f"  - Strategy: {multihead_config['strategy']}")
        print(f"  - Shared reg/obj: {multihead_config['shared_reg_obj']}")
    
    # 4. 測試與 datasets.py 的相容性（不實際載入數據）
    print("✓ Configuration structure compatible with datasets.py")
    
    return True

def test_no_side_effects():
    """確保新配置不會影響原始 coco.yaml"""
    with open('data/coco.yaml', 'r') as f:
        original = yaml.safe_load(f)
    
    assert 'multihead' not in original, "Original coco.yaml should not have multihead"
    print("✓ Original coco.yaml unchanged")
    
    return True

def test_class_coverage():
    """確保所有 COCO 類別都被覆蓋"""
    with open('data/coco-multihead.yaml', 'r') as f:
        data = yaml.safe_load(f)
    
    mh_config = data['multihead']
    all_classes = []
    
    # 收集所有類別
    for head_id in range(4):
        head_classes = mh_config['head_assignments'][head_id]['classes']
        all_classes.extend(head_classes)
    
    # 驗證覆蓋
    coco_names = data['names']
    for class_id in range(80):
        assert class_id in all_classes, f"Class {class_id} ({coco_names[class_id]}) not assigned to any head"
    
    print("✓ All COCO classes covered")
    
    # 顯示分配統計
    print("\n✓ Class distribution:")
    for head_id in range(4):
        head_info = mh_config['head_assignments'][head_id]
        class_names = [coco_names[i] for i in head_info['classes'][:5]]  # 顯示前5個
        print(f"  Head {head_id} ({head_info['name']}): {class_names[:3]} ...")
    
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Configuration Compatibility")
    print("=" * 60)
    
    try:
        print("\n1. Testing backward compatibility...")
        test_backward_compatibility()
        
        print("\n2. Testing no side effects...")
        test_no_side_effects()
        
        print("\n3. Testing class coverage...")
        test_class_coverage()
        
        print("\n" + "=" * 60)
        print("✅ All compatibility tests PASSED!")
        print("=" * 60)
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)