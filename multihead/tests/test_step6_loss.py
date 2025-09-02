import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import torch
import torch.nn as nn
import yaml

def test_loss_import():
    """測試能否導入損失類"""
    try:
        from utils.loss import ComputeLoss  # 原始損失
        print("✓ Original ComputeLoss exists")
        
        from utils.loss_multihead import ComputeLossMultiHead  # 新損失
        print("✓ ComputeLossMultiHead imported")
        return True
    except ImportError as e:
        print(f"⚠ Import failed: {e}")
        return False

def test_loss_initialization():
    """測試損失函數初始化"""
    from models.yolo import MultiHeadDetect
    from utils.loss_multihead import ComputeLossMultiHead
    
    # 創建簡單模型
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.ModuleList([
                MultiHeadDetect(nc=80, 
                              anchors=([10,13, 16,30, 33,23], 
                                      [30,61, 62,45, 59,119], 
                                      [116,90, 156,198, 373,326]),
                              ch=(128, 256, 512))
            ])
            self.hyp = {
                'box': 0.05,
                'cls': 0.5,
                'obj': 1.0,
                'anchor_t': 4.0,
                'cls_pw': 1.0,
                'obj_pw': 1.0,
                'fl_gamma': 0.0,
                'label_smoothing': 0.0
            }
            self.gr = 1.0  # gradient ratio
    
    model = SimpleModel()
    
    try:
        # 創建損失函數
        compute_loss = ComputeLossMultiHead(model)
        
        # 檢查屬性
        assert hasattr(compute_loss, 'nc')
        assert hasattr(compute_loss, 'nl')
        assert hasattr(compute_loss, 'anchors')
        assert hasattr(compute_loss, 'n_heads')
        assert compute_loss.n_heads == 4
        
        print("✓ Loss function initialized correctly")
        print(f"  Heads: {compute_loss.n_heads}")
        print(f"  Head weights: {compute_loss.head_weights}")
        return True
        
    except Exception as e:
        print(f"✗ Initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_loss_forward():
    """測試損失計算前向傳播"""
    from models.yolo import MultiHeadDetect
    from utils.loss_multihead import ComputeLossMultiHead
    
    # 創建簡單模型
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.ModuleList([
                MultiHeadDetect(nc=80, 
                              anchors=([10,13, 16,30, 33,23], 
                                      [30,61, 62,45, 59,119], 
                                      [116,90, 156,198, 373,326]),
                              ch=(128, 256, 512))
            ])
            self.hyp = {
                'box': 0.05,
                'cls': 0.5,
                'obj': 1.0,
                'anchor_t': 4.0,
                'cls_pw': 1.0,
                'obj_pw': 1.0,
                'fl_gamma': 0.0,
                'label_smoothing': 0.0
            }
            self.gr = 1.0
    
    model = SimpleModel()
    compute_loss = ComputeLossMultiHead(model)
    
    # 創建假數據
    batch_size = 2
    x = [
        torch.randn(batch_size, 128, 80, 80),
        torch.randn(batch_size, 256, 40, 40),
        torch.randn(batch_size, 512, 20, 20),
    ]
    
    # 創建假目標 (image_idx, class, x, y, w, h)
    targets = torch.tensor([
        [0, 0, 0.5, 0.5, 0.1, 0.1],  # person (head 0)
        [0, 1, 0.3, 0.3, 0.2, 0.2],  # bicycle (head 1)
        [1, 14, 0.7, 0.7, 0.15, 0.15], # bird (head 2)
    ])
    
    # 前向傳播獲取預測
    model.model[0].train()
    predictions = model.model[0](x)
    
    # 計算損失
    loss, loss_items = compute_loss(predictions, targets)
    
    # 檢查輸出
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0  # scalar
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)
    
    assert loss_items.shape == (3,)  # box, obj, cls
    
    print("✓ Loss computation works")
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Components: box={loss_items[0]:.4f}, obj={loss_items[1]:.4f}, cls={loss_items[2]:.4f}")
    
    return True

def test_backward_compatibility():
    """測試原始 ComputeLoss 是否仍然工作"""
    from utils.loss import ComputeLoss
    from models.yolo import Detect
    
    # 簡單測試原始損失是否未受影響
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.ModuleList([
                Detect(nc=80, 
                      anchors=([10,13, 16,30, 33,23], 
                              [30,61, 62,45, 59,119], 
                              [116,90, 156,198, 373,326]),
                      ch=(128, 256, 512))
            ])
            self.hyp = {
                'box': 0.05,
                'cls': 0.5,
                'obj': 1.0,
                'anchor_t': 4.0,
                'cls_pw': 1.0,
                'obj_pw': 1.0,
                'fl_gamma': 0.0,
                'label_smoothing': 0.0
            }
            self.gr = 1.0
    
    model = SimpleModel()
    compute_loss = ComputeLoss(model)
    
    print("✓ Original ComputeLoss still works")
    return True

def test_gradient_flow():
    """測試梯度流動"""
    from models.yolo import MultiHeadDetect
    from utils.loss_multihead import ComputeLossMultiHead
    
    # 創建模型
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.ModuleList([
                MultiHeadDetect(nc=80, 
                              anchors=([10,13, 16,30, 33,23], 
                                      [30,61, 62,45, 59,119], 
                                      [116,90, 156,198, 373,326]),
                              ch=(128, 256, 512))
            ])
            self.hyp = {
                'box': 0.05,
                'cls': 0.5, 
                'obj': 1.0,
                'anchor_t': 4.0,
                'cls_pw': 1.0,
                'obj_pw': 1.0,
                'fl_gamma': 0.0,
                'label_smoothing': 0.0
            }
            self.gr = 1.0
    
    model = SimpleModel()
    compute_loss = ComputeLossMultiHead(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    # 準備數據
    batch_size = 2
    x = [
        torch.randn(batch_size, 128, 16, 16, requires_grad=True),
        torch.randn(batch_size, 256, 8, 8, requires_grad=True),
        torch.randn(batch_size, 512, 4, 4, requires_grad=True),
    ]
    
    targets = torch.tensor([
        [0, 0, 0.5, 0.5, 0.1, 0.1],
        [1, 15, 0.3, 0.7, 0.2, 0.15],
    ])
    
    # 前向傳播
    model.model[0].train()
    predictions = model.model[0](x)
    
    # 計算損失
    loss, _ = compute_loss(predictions, targets)
    
    # 反向傳播
    optimizer.zero_grad()
    loss.backward()
    
    # 檢查梯度
    has_grad = False
    for name, param in model.named_parameters():
        if param.grad is not None:
            has_grad = True
            assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
            assert not torch.isinf(param.grad).any(), f"Inf gradient in {name}"
    
    assert has_grad, "No gradients computed"
    
    print("✓ Gradient flow verified")
    return True

if __name__ == "__main__":
    print("Testing Step 6: Loss implementation...")
    print("-" * 50)
    
    if test_loss_import():
        test_loss_initialization()
        test_loss_forward()
        test_backward_compatibility()
        test_gradient_flow()
        print("\n✅ All Step 6 tests passed!")
    else:
        print("\n⚠ Need to implement ComputeLossMultiHead")