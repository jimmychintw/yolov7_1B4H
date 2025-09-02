import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import torch
import torch.nn as nn

def test_multihead_loss_import():
    """測試能否導入多頭損失函數"""
    try:
        from utils.loss_multihead import ComputeLossMultiHead
        print("✓ ComputeLossMultiHead imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Failed to import: {e}")
        return False

def test_multihead_detect_import():
    """測試能否導入多頭檢測層"""
    try:
        from models.yolo import MultiHeadDetect
        print("✓ MultiHeadDetect imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Failed to import: {e}")
        return False

def test_loss_selection():
    """測試損失函數選擇邏輯"""
    from models.yolo import MultiHeadDetect, Detect
    from utils.loss import ComputeLoss
    from utils.loss_multihead import ComputeLossMultiHead
    
    # 創建一個簡單的多頭檢測層
    multihead_det = MultiHeadDetect(
        nc=80,
        anchors=([10,13, 16,30, 33,23],
                [30,61, 62,45, 59,119],
                [116,90, 156,198, 373,326]),
        ch=(128, 256, 512)
    )
    
    # 創建一個簡單的單頭檢測層
    singlehead_det = Detect(
        nc=80,
        anchors=([10,13, 16,30, 33,23],
                [30,61, 62,45, 59,119],
                [116,90, 156,198, 373,326]),
        ch=(128, 256, 512)
    )
    
    # 測試多頭選擇邏輯
    if isinstance(multihead_det, MultiHeadDetect):
        print("✓ MultiHead detection recognized")
        # 創建一個簡單的模型來測試損失函數初始化
        class SimpleModel(nn.Module):
            def __init__(self, det):
                super().__init__()
                self.model = nn.ModuleList([det])
                self.hyp = {
                    'box': 0.05, 'cls': 0.5, 'obj': 1.0,
                    'anchor_t': 4.0, 'cls_pw': 1.0, 'obj_pw': 1.0,
                    'fl_gamma': 0.0, 'label_smoothing': 0.0
                }
                self.gr = 1.0
        
        model = SimpleModel(multihead_det)
        compute_loss = ComputeLossMultiHead(model)
        print(f"✓ ComputeLossMultiHead initialized for {multihead_det.n_heads} heads")
    
    # 測試單頭選擇邏輯
    if not isinstance(singlehead_det, MultiHeadDetect):
        print("✓ Single head detection recognized")
        model = SimpleModel(singlehead_det)
        compute_loss = ComputeLoss(model)
        print("✓ ComputeLoss initialized for single head")
    
    return True

def test_train_integration():
    """測試 train.py 修改是否正確"""
    # 檢查 train.py 是否包含多頭檢測邏輯
    with open('train.py', 'r') as f:
        content = f.read()
    
    if 'MultiHeadDetect' in content:
        print("✓ train.py contains MultiHeadDetect import")
    else:
        print("✗ train.py missing MultiHeadDetect import")
        return False
    
    if 'ComputeLossMultiHead' in content:
        print("✓ train.py contains ComputeLossMultiHead import")
    else:
        print("✗ train.py missing ComputeLossMultiHead import")
        return False
    
    if 'MultiHead Loss Selection' in content:
        print("✓ train.py contains loss selection logic")
    else:
        print("✗ train.py missing loss selection logic")
        return False
    
    return True

def test_training_step_simple():
    """測試簡單的訓練步驟"""
    from models.yolo import MultiHeadDetect
    from utils.loss_multihead import ComputeLossMultiHead
    import torch.optim as optim
    
    # 創建簡單模型
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.ModuleList([
                MultiHeadDetect(nc=80,
                              anchors=([10,13, 16,30, 33,23],),
                              ch=(128,))
            ])
            self.hyp = {
                'box': 0.05, 'cls': 0.5, 'obj': 1.0,
                'anchor_t': 4.0, 'cls_pw': 1.0, 'obj_pw': 1.0,
                'fl_gamma': 0.0, 'label_smoothing': 0.0
            }
            self.gr = 1.0
    
    model = SimpleModel()
    model.train()
    
    # 創建損失函數
    compute_loss = ComputeLossMultiHead(model)
    
    # 創建優化器
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    
    # 創建假數據
    x = [torch.randn(2, 128, 8, 8)]
    targets = torch.tensor([
        [0, 0, 0.5, 0.5, 0.1, 0.1],
        [1, 15, 0.3, 0.7, 0.2, 0.15],
    ])
    
    # 訓練模式的前向傳播
    model.model[0].train()
    predictions = model.model[0](x)
    
    # 計算損失
    loss, loss_items = compute_loss(predictions, targets)
    
    # 反向傳播
    optimizer.zero_grad()
    loss.backward()
    
    # 檢查梯度
    has_grad = False
    for p in model.parameters():
        if p.grad is not None and p.grad.norm() > 0:
            has_grad = True
            break
    
    if has_grad:
        print("✓ Gradients computed successfully")
    else:
        print("✗ No gradients computed")
        return False
    
    if not torch.isnan(loss):
        print(f"✓ Loss computed: {loss.item():.4f}")
    else:
        print("✗ Loss is NaN")
        return False
    
    # 更新權重
    optimizer.step()
    print("✓ Training step completed")
    
    return True

if __name__ == "__main__":
    print("Testing Step 7: Training Integration (Simplified)")
    print("=" * 50)
    
    all_pass = True
    
    # 運行測試
    tests = [
        ("Import MultiHead Loss", test_multihead_loss_import),
        ("Import MultiHead Detect", test_multihead_detect_import),
        ("Loss Selection Logic", test_loss_selection),
        ("Train.py Integration", test_train_integration),
        ("Simple Training Step", test_training_step_simple),
    ]
    
    for name, test_func in tests:
        print(f"\n{name}:")
        try:
            result = test_func()
            all_pass = all_pass and result
        except Exception as e:
            print(f"✗ Test failed with error: {e}")
            import traceback
            traceback.print_exc()
            all_pass = False
    
    print("\n" + "=" * 50)
    if all_pass:
        print("✅ All Step 7 tests passed!")
    else:
        print("❌ Some tests failed - review needed")