import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import torch
import yaml
import tempfile
import shutil

def test_loss_selection_logic():
    """測試損失函數選擇邏輯"""
    from models.yolo import Model, MultiHeadDetect, Detect
    from utils.loss import ComputeLoss
    from utils.loss_multihead import ComputeLossMultiHead
    
    # 測試多頭模型
    multihead_cfg = """
nc: 80
depth_multiple: 0.33
width_multiple: 0.50

anchors:
  - [10,13, 16,30, 33,23]
  - [30,61, 62,45, 59,119]
  - [116,90, 156,198, 373,326]

backbone:
  - [-1, 1, Conv, [32, 3, 2]]  # 0
  - [-1, 1, Conv, [64, 3, 2]]  # 1
  - [-1, 1, Conv, [128, 3, 2]] # 2
  - [-1, 1, Conv, [256, 3, 2]] # 3
  - [-1, 1, Conv, [512, 3, 2]] # 4

head:
  - [[2, 3, 4], 1, MultiHeadDetect, [nc, anchors]]
"""
    
    # 測試單頭模型
    singlehead_cfg = """
nc: 80
depth_multiple: 0.33
width_multiple: 0.50

anchors:
  - [10,13, 16,30, 33,23]
  - [30,61, 62,45, 59,119]
  - [116,90, 156,198, 373,326]

backbone:
  - [-1, 1, Conv, [32, 3, 2]]  # 0
  - [-1, 1, Conv, [64, 3, 2]]  # 1
  - [-1, 1, Conv, [128, 3, 2]] # 2
  - [-1, 1, Conv, [256, 3, 2]] # 3
  - [-1, 1, Conv, [512, 3, 2]] # 4

head:
  - [[2, 3, 4], 1, Detect, [nc, anchors]]
"""
    
    # 創建臨時文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(multihead_cfg)
        multihead_cfg_path = f.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(singlehead_cfg)
        singlehead_cfg_path = f.name
    
    try:
        # 測試多頭模型選擇正確的損失
        multihead_model = Model(multihead_cfg_path)
        det = multihead_model.model[-1]
        assert isinstance(det, MultiHeadDetect), "Should be MultiHeadDetect"
        
        # 模擬 train.py 的損失選擇邏輯
        if isinstance(det, MultiHeadDetect):
            compute_loss = ComputeLossMultiHead(multihead_model)
            print("✓ MultiHead model → ComputeLossMultiHead")
        
        # 測試單頭模型選擇正確的損失
        singlehead_model = Model(singlehead_cfg_path)
        det = singlehead_model.model[-1]
        assert isinstance(det, Detect), "Should be Detect"
        
        if not isinstance(det, MultiHeadDetect):
            compute_loss = ComputeLoss(singlehead_model)
            print("✓ Single head model → ComputeLoss")
        
    finally:
        # 清理
        os.unlink(multihead_cfg_path)
        os.unlink(singlehead_cfg_path)
    
    return True

def test_training_step():
    """測試單個訓練步驟"""
    from models.yolo import Model
    from utils.loss_multihead import ComputeLossMultiHead
    import torch.optim as optim
    
    # 創建簡單的多頭配置
    cfg = """
nc: 80
depth_multiple: 0.33
width_multiple: 0.50

anchors:
  - [10,13, 16,30, 33,23]
  - [30,61, 62,45, 59,119]
  - [116,90, 156,198, 373,326]

backbone:
  - [-1, 1, Conv, [32, 3, 2]]  # 0
  - [-1, 1, Conv, [64, 3, 2]]  # 1
  - [-1, 1, Conv, [128, 3, 2]] # 2
  - [-1, 1, Conv, [256, 3, 2]] # 3
  - [-1, 1, Conv, [512, 3, 2]] # 4

head:
  - [[2, 3, 4], 1, MultiHeadDetect, [nc, anchors]]
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(cfg)
        cfg_path = f.name
    
    try:
        # 創建模型
        model = Model(cfg_path)
        model.train()
        
        # 創建優化器
        optimizer = optim.SGD(model.parameters(), lr=0.001)
        
        # 創建損失函數
        compute_loss = ComputeLossMultiHead(model)
        
        # 模擬一個訓練步驟
        imgs = torch.randn(2, 3, 640, 640)
        targets = torch.tensor([
            [0, 0, 0.5, 0.5, 0.1, 0.1],
            [1, 15, 0.3, 0.7, 0.2, 0.15],
        ])
        
        # 前向傳播
        pred = model(imgs)
        
        # 計算損失
        loss, loss_items = compute_loss(pred, targets)
        
        # 反向傳播
        loss.backward()
        
        # 檢查梯度
        total_grad_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                total_grad_norm += p.grad.data.norm(2).item()
        
        assert total_grad_norm > 0, "Should have gradients"
        assert not torch.isnan(loss), "Loss should not be NaN"
        
        # 更新權重
        optimizer.step()
        optimizer.zero_grad()
        
        print("✓ Training step executed successfully")
        print(f"  Loss: {loss.item():.4f}")
        print(f"  Gradient norm: {total_grad_norm:.4f}")
        
    finally:
        os.unlink(cfg_path)
    
    return True

if __name__ == "__main__":
    print("Testing Step 7: Training integration...")
    print("-" * 50)
    
    test_loss_selection_logic()
    test_training_step()
    
    print("\n✅ All Step 7 tests passed!")