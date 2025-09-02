import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import torch
import torch.nn as nn

def test_forward_training_mode():
    """測試訓練模式的前向傳播"""
    from models.yolo import MultiHeadDetect
    
    nc = 80
    anchors = ([10,13, 16,30, 33,23], [30,61, 62,45, 59,119], [116,90, 156,198, 373,326])
    ch = (128, 256, 512)
    batch_size = 2
    
    model = MultiHeadDetect(nc=nc, anchors=anchors, ch=ch, n_heads=4)
    model.train()  # 設置為訓練模式
    
    # 創建測試輸入
    x = [
        torch.randn(batch_size, ch[0], 80, 80),  # P3/8
        torch.randn(batch_size, ch[1], 40, 40),  # P4/16
        torch.randn(batch_size, ch[2], 20, 20),  # P5/32
    ]
    
    # 前向傳播
    output = model(x)
    
    # 訓練模式應該返回 tuple: (reg_obj_outputs, cls_outputs)
    assert isinstance(output, tuple), "Training mode should return tuple"
    assert len(output) == 2, "Should return (reg_obj, cls)"
    
    reg_obj_outputs, cls_outputs = output
    
    # 檢查 reg_obj 輸出
    assert len(reg_obj_outputs) == 3, "Should have 3 scale outputs for reg_obj"
    for i, reg_obj in enumerate(reg_obj_outputs):
        expected_shape = (batch_size, 3, x[i].shape[2], x[i].shape[3], 5)
        assert reg_obj.shape == expected_shape, f"Scale {i} reg_obj shape mismatch: {reg_obj.shape} != {expected_shape}"
    
    # 檢查 cls 輸出
    assert len(cls_outputs) == 4, "Should have 4 heads"
    for head_id, head_outputs in enumerate(cls_outputs):
        assert len(head_outputs) == 3, f"Head {head_id} should have 3 scales"
        for i, cls in enumerate(head_outputs):
            expected_shape = (batch_size, 3, x[i].shape[2], x[i].shape[3], nc)
            assert cls.shape == expected_shape, f"Head {head_id} scale {i} cls shape mismatch"
    
    print("✓ Training mode forward pass correct")
    return True

def test_forward_inference_mode():
    """測試推理模式的前向傳播"""
    from models.yolo import MultiHeadDetect
    
    nc = 80
    anchors = ([10,13, 16,30, 33,23], [30,61, 62,45, 59,119], [116,90, 156,198, 373,326])
    ch = (128, 256, 512)
    batch_size = 1
    
    model = MultiHeadDetect(nc=nc, anchors=anchors, ch=ch, n_heads=4)
    model.eval()  # 設置為推理模式
    
    # 設置 stride（模擬 Model 類的行為）
    model.stride = torch.tensor([8., 16., 32.])
    
    # 創建測試輸入
    x = [
        torch.randn(batch_size, ch[0], 80, 80),
        torch.randn(batch_size, ch[1], 40, 40),
        torch.randn(batch_size, ch[2], 20, 20),
    ]
    
    # 前向傳播
    with torch.no_grad():
        output = model(x)
    
    # 推理模式應該返回 (predictions, x)
    assert isinstance(output, tuple), "Inference should return tuple"
    assert len(output) == 2, "Should return (pred, x)"
    
    pred, x_pass = output
    
    # 檢查預測輸出
    total_anchors = (80*80 + 40*40 + 20*20) * 3  # 3 anchors per cell
    expected_pred_shape = (batch_size, total_anchors, 85)
    assert pred.shape == expected_pred_shape, f"Prediction shape mismatch: {pred.shape} != {expected_pred_shape}"
    
    # 檢查 x 是否原樣返回
    assert len(x_pass) == 3, "Should pass through original x"
    for i in range(3):
        assert torch.equal(x_pass[i], x[i]), f"x[{i}] not passed through correctly"
    
    print("✓ Inference mode forward pass correct")
    return True

def test_grid_generation():
    """測試網格生成"""
    from models.yolo import MultiHeadDetect
    
    nc = 80
    anchors = ([10,13, 16,30, 33,23], [30,61, 62,45, 59,119], [116,90, 156,198, 373,326])
    ch = (128, 256, 512)
    
    model = MultiHeadDetect(nc=nc, anchors=anchors, ch=ch, n_heads=4)
    
    # 測試 _make_grid
    grid = model._make_grid(10, 10)
    assert grid.shape == (1, 1, 10, 10, 2)
    
    # 檢查網格值
    assert grid[0, 0, 0, 0, 0] == 0  # x at (0,0)
    assert grid[0, 0, 0, 0, 1] == 0  # y at (0,0)
    assert grid[0, 0, 9, 9, 0] == 9  # x at (9,9)
    assert grid[0, 0, 9, 9, 1] == 9  # y at (9,9)
    
    print("✓ Grid generation correct")
    return True

def test_training_vs_eval_mode():
    """測試訓練和評估模式的區別"""
    from models.yolo import MultiHeadDetect
    
    nc = 80
    anchors = ([10,13, 16,30, 33,23], [30,61, 62,45, 59,119], [116,90, 156,198, 373,326])
    ch = (128, 256, 512)
    
    model = MultiHeadDetect(nc=nc, anchors=anchors, ch=ch, n_heads=4)
    model.stride = torch.tensor([8., 16., 32.])
    
    x = [
        torch.randn(1, ch[0], 80, 80),
        torch.randn(1, ch[1], 40, 40),
        torch.randn(1, ch[2], 20, 20),
    ]
    
    # 訓練模式
    model.train()
    train_out = model(x)
    assert isinstance(train_out, tuple) and len(train_out) == 2
    
    # 評估模式
    model.eval()
    with torch.no_grad():
        eval_out = model(x)
    assert isinstance(eval_out, tuple) and len(eval_out) == 2
    
    # 輸出格式應該不同
    # 訓練模式返回 (reg_obj_outputs, cls_outputs)
    # 推理模式返回 (pred, x)
    assert isinstance(train_out[0], list), "Training should return list of reg_obj"
    assert isinstance(eval_out[0], torch.Tensor), "Eval should return tensor pred"
    
    print("✓ Training/eval mode distinction works")
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Step 5: Forward implementation...")
    print("=" * 60)
    
    tests = [
        test_forward_training_mode,
        test_forward_inference_mode,
        test_grid_generation,
        test_training_vs_eval_mode
    ]
    
    all_passed = True
    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"✗ Test failed: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False
            break
    
    if all_passed:
        print("\n✅ All Step 5 forward tests passed!")