import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import torch
import torch.nn as nn

def test_end_to_end_gradient():
    """測試端到端梯度流"""
    from models.yolo import MultiHeadDetect
    from utils.loss_multihead import ComputeLossMultiHead
    
    # 創建模型
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            # 簡化的 backbone
            self.backbone = nn.ModuleList([
                nn.Conv2d(3, 128, 3, 2, 1),
                nn.Conv2d(128, 256, 3, 2, 1),
                nn.Conv2d(256, 512, 3, 2, 1),
            ])
            # Detection head
            self.model = nn.ModuleList([
                MultiHeadDetect(nc=80,
                              anchors=([10,13, 16,30, 33,23],
                                      [30,61, 62,45, 59,119],
                                      [116,90, 156,198, 373,326]),
                              ch=(128, 256, 512))
            ])
            self.hyp = {
                'box': 0.05, 'cls': 0.5, 'obj': 1.0,
                'anchor_t': 4.0, 'cls_pw': 1.0, 'obj_pw': 1.0,
                'fl_gamma': 0.0, 'label_smoothing': 0.0
            }
            self.gr = 1.0
    
    model = TestModel()
    compute_loss = ComputeLossMultiHead(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    # 前向傳播
    x = torch.randn(2, 3, 320, 320)
    
    # Backbone
    features = []
    feat = x
    for conv in model.backbone:
        feat = conv(feat)
        features.append(feat)
    
    # Detection
    model.model[0].train()
    predictions = model.model[0](features)
    
    # 假目標
    targets = torch.tensor([
        [0, 0, 0.5, 0.5, 0.1, 0.1],
        [1, 15, 0.3, 0.7, 0.2, 0.15],
    ])
    
    # 計算損失
    loss, loss_items = compute_loss(predictions, targets)
    
    # 反向傳播
    optimizer.zero_grad()
    loss.backward()
    
    # 檢查梯度
    has_grad = False
    for name, param in model.named_parameters():
        if param.grad is not None:
            has_grad = True
            assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
    
    assert has_grad, "No gradients computed"
    
    # 更新參數
    optimizer.step()
    
    print("✓ End-to-end gradient flow works")
    print(f"  Final loss: {loss.item():.4f}")
    print(f"  Loss items: box={loss_items[0]:.4f}, obj={loss_items[1]:.4f}, cls={loss_items[2]:.4f}")
    
    return True

def test_multi_batch_training():
    """測試多批次訓練"""
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
                'box': 0.05, 'cls': 0.5, 'obj': 1.0,
                'anchor_t': 4.0, 'cls_pw': 1.0, 'obj_pw': 1.0,
                'fl_gamma': 0.0, 'label_smoothing': 0.0
            }
            self.gr = 1.0
    
    model = SimpleModel()
    compute_loss = ComputeLossMultiHead(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    
    # 訓練幾個批次
    losses = []
    model.model[0].train()
    
    for batch in range(3):
        # 生成隨機數據
        x = [
            torch.randn(2, 128, 40, 40),
            torch.randn(2, 256, 20, 20),
            torch.randn(2, 512, 10, 10),
        ]
        
        # 隨機目標
        targets = torch.rand(5, 6)
        targets[:, 0] = torch.randint(0, 2, (5,))  # batch index
        targets[:, 1] = torch.randint(0, 80, (5,))  # class
        
        # 前向傳播
        predictions = model.model[0](x)
        loss, _ = compute_loss(predictions, targets)
        
        # 反向傳播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        print(f"  Batch {batch}: loss={loss.item():.4f}")
    
    # 檢查損失是否在變化
    assert len(set(losses)) > 1, "Loss not changing across batches"
    
    print("✓ Multi-batch training works")
    return True

def test_head_specific_loss():
    """測試每個頭的損失計算"""
    from models.yolo import MultiHeadDetect
    from utils.loss_multihead import ComputeLossMultiHead
    
    # 創建模型
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
    compute_loss = ComputeLossMultiHead(model)
    
    # 測試不同頭的目標
    test_cases = [
        (0, "Head 0 (person)"),  # person class
        (1, "Head 1 (bicycle)"),  # bicycle class
        (14, "Head 2 (bird)"),    # bird class
        (56, "Head 3 (chair)"),   # chair class
    ]
    
    model.model[0].train()
    
    for class_id, head_name in test_cases:
        x = [torch.randn(1, 128, 8, 8)]
        targets = torch.tensor([[0, class_id, 0.5, 0.5, 0.2, 0.2]])
        
        predictions = model.model[0](x)
        loss, loss_items = compute_loss(predictions, targets)
        
        print(f"  {head_name}: loss={loss.item():.4f}, cls={loss_items[2]:.4f}")
        
        assert not torch.isnan(loss), f"NaN loss for {head_name}"
    
    print("✓ Head-specific loss computation works")
    return True

if __name__ == "__main__":
    print("Testing Step 6 integration...")
    print("-" * 50)
    
    test_end_to_end_gradient()
    test_multi_batch_training()
    test_head_specific_loss()
    
    print("\n✅ All integration tests passed!")