import yaml
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

def test_config_structure():
    """驗證配置文件結構"""
    # 1. 檢查原始 coco.yaml 是否存在
    assert os.path.exists('data/coco.yaml'), "Original coco.yaml not found"
    
    # 2. 載入原始配置
    with open('data/coco.yaml', 'r') as f:
        original = yaml.safe_load(f)
    
    print("✓ Original coco.yaml loaded successfully")
    print(f"  - Classes: {original['nc']}")
    print(f"  - Train path: {original['train']}")
    
    # 3. 檢查新配置是否存在
    multihead_path = 'data/coco-multihead.yaml'
    if not os.path.exists(multihead_path):
        print("⚠ coco-multihead.yaml not found yet")
        return False
    
    # 4. 載入新配置
    with open(multihead_path, 'r') as f:
        multihead = yaml.safe_load(f)
    
    # 5. 驗證必要字段
    required_fields = ['train', 'val', 'nc', 'names']
    for field in required_fields:
        assert field in multihead, f"Missing required field: {field}"
        assert multihead[field] == original[field], f"Field {field} should match original"
    
    # 6. 驗證多頭擴展
    assert 'multihead' in multihead, "Missing multihead configuration"
    mh_config = multihead['multihead']
    
    assert mh_config['enabled'] == True
    assert mh_config['n_heads'] == 4
    assert mh_config['strategy'] == 'strategy_a'
    
    # 7. 驗證類別分組無重複
    all_classes = []
    for head_id in range(4):
        head_classes = mh_config['head_assignments'][head_id]['classes']
        all_classes.extend(head_classes)
    
    assert len(all_classes) == 80, f"Should have 80 classes total, got {len(all_classes)}"
    assert len(set(all_classes)) == 80, "Classes have duplicates!"
    assert set(all_classes) == set(range(80)), "Missing some class IDs"
    
    print("✓ MultiHead configuration validated")
    print(f"  - Strategy: {mh_config['strategy']}")
    print(f"  - Heads: {mh_config['n_heads']}")
    print(f"  - No duplicate classes")
    
    # 8. 顯示每個頭的詳細信息
    print("\n✓ Head assignments:")
    for head_id in range(4):
        head_info = mh_config['head_assignments'][head_id]
        print(f"  Head {head_id} ({head_info['name']}): {len(head_info['classes'])} classes")
        print(f"    Description: {head_info['description']}")
    
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Step 1: Configuration Structure")
    print("=" * 60)
    try:
        result = test_config_structure()
        if result:
            print("\n✅ Step 1 configuration tests PASSED!")
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)