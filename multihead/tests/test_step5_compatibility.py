import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import torch
import torch.nn as nn

def test_interface_compatibility():
    """測試介面相容性"""
    from models.yolo import Detect, MultiHeadDetect
    
    nc = 80
    anchors = ([10,13, 16,30, 33,23], [30,61, 62,45, 59,119], [116,90, 156,198, 373,326])
    ch = (128, 256, 512)
    
    # 創建兩個模型
    original = Detect(nc=nc, anchors=anchors, ch=ch)
    multihead = MultiHeadDetect(nc=nc, anchors=anchors, ch=ch, n_heads=4)
    
    # 檢查屬性相容性
    assert hasattr(multihead, 'stride'), "Missing stride attribute"
    assert hasattr(multihead, 'anchors'), "Missing anchors attribute"
    assert hasattr(multihead, 'anchor_grid'), "Missing anchor_grid attribute"
    assert hasattr(multihead, 'grid'), "Missing grid attribute"
    assert hasattr(multihead, 'm'), "Missing m attribute"
    assert hasattr(multihead, 'nl'), "Missing nl attribute"
    assert hasattr(multihead, 'na'), "Missing na attribute"
    assert hasattr(multihead, 'nc'), "Missing nc attribute"
    assert hasattr(multihead, 'no'), "Missing no attribute"
    
    # 檢查數值相容性
    assert multihead.nl == original.nl, "nl mismatch"
    assert multihead.na == original.na, "na mismatch"
    assert multihead.nc == original.nc, "nc mismatch"
    assert multihead.no == original.no, "no mismatch"
    
    print("✓ Interface compatibility verified")
    return True

def test_training_output_format():
    """測試訓練輸出格式"""
    from models.yolo import MultiHeadDetect
    
    nc = 80
    anchors = ([10,13, 16,30, 33,23], [30,61, 62,45, 59,119], [116,90, 156,198, 373,326])
    ch = (128, 256, 512)
    batch_size = 2
    
    model = MultiHeadDetect(nc=nc, anchors=anchors, ch=ch, n_heads=4)
    model.train()
    
    x = [
        torch.randn(batch_size, ch[0], 80, 80),
        torch.randn(batch_size, ch[1], 40, 40),
        torch.randn(batch_size, ch[2], 20, 20),
    ]
    
    output = model(x)
    
    # 檢查輸出格式
    assert isinstance(output, tuple), "Should return tuple"
    assert len(output) == 2, "Should return (reg_obj, cls)"
    
    reg_obj_outputs, cls_outputs = output
    
    # 檢查 reg_obj 格式
    assert isinstance(reg_obj_outputs, list), "reg_obj should be list"
    assert len(reg_obj_outputs) == 3, "Should have 3 scales"
    
    # 檢查 cls 格式
    assert isinstance(cls_outputs, list), "cls should be list"
    assert len(cls_outputs) == 4, "Should have 4 heads"
    
    for head_outputs in cls_outputs:
        assert isinstance(head_outputs, list), "Each head should have list of scales"
        assert len(head_outputs) == 3, "Each head should have 3 scales"
    
    print("✓ Training output format correct")
    return True

def test_inference_output_format():
    """測試推理輸出格式"""
    from models.yolo import MultiHeadDetect
    
    nc = 80
    anchors = ([10,13, 16,30, 33,23], [30,61, 62,45, 59,119], [116,90, 156,198, 373,326])
    ch = (128, 256, 512)
    batch_size = 1
    
    model = MultiHeadDetect(nc=nc, anchors=anchors, ch=ch, n_heads=4)
    model.eval()
    model.stride = torch.tensor([8., 16., 32.])
    
    x = [
        torch.randn(batch_size, ch[0], 80, 80),
        torch.randn(batch_size, ch[1], 40, 40),
        torch.randn(batch_size, ch[2], 20, 20),
    ]
    
    with torch.no_grad():
        output = model(x)
    
    # 檢查輸出格式
    assert isinstance(output, tuple), "Should return tuple"
    assert len(output) == 2, "Should return (pred, x)"
    
    pred, x_out = output
    
    # 檢查預測格式
    assert isinstance(pred, torch.Tensor), "pred should be tensor"
    assert pred.dim() == 3, "pred should be 3D"
    assert pred.shape[0] == batch_size, "Wrong batch size"
    assert pred.shape[2] == 85, "Wrong output channels (should be 85)"
    
    # 檢查 x 傳遞
    assert len(x_out) == len(x), "x not passed through"
    
    print("✓ Inference output format correct")
    return True

def test_loss_compatibility():
    """測試與損失函數的相容性"""
    from models.yolo import MultiHeadDetect
    
    nc = 80
    anchors = ([10,13, 16,30, 33,23], [30,61, 62,45, 59,119], [116,90, 156,198, 373,326])
    ch = (128, 256, 512)
    batch_size = 2
    
    model = MultiHeadDetect(nc=nc, anchors=anchors, ch=ch, n_heads=4)
    model.train()
    
    x = [
        torch.randn(batch_size, ch[0], 80, 80),
        torch.randn(batch_size, ch[1], 40, 40),
        torch.randn(batch_size, ch[2], 20, 20),
    ]
    
    reg_obj_outputs, cls_outputs = model(x)
    
    # 模擬損失計算
    loss = 0
    
    # Reg/obj loss
    for reg_obj in reg_obj_outputs:
        # 檢查形狀是否正確
        assert reg_obj.shape[-1] == 5, "reg_obj should have 5 channels"
        loss = loss + reg_obj.mean()  # 簡單的虛擬損失
    
    # Cls loss
    for head_id, head_outputs in enumerate(cls_outputs):
        for scale_id, cls in enumerate(head_outputs):
            # 檢查形狀是否正確
            assert cls.shape[-1] == nc, f"cls should have {nc} channels"
            loss = loss + cls.mean()  # 簡單的虛擬損失
    
    # 確保可以反向傳播
    loss.backward()
    
    # 檢查梯度
    has_grad = False
    for param in model.parameters():
        if param.grad is not None:
            has_grad = True
            break
    
    assert has_grad, "No gradients computed"
    
    print("✓ Loss compatibility verified")
    return True

def test_model_integration_ready():
    """測試是否準備好整合到 Model 類"""
    from models.yolo import MultiHeadDetect
    
    nc = 80
    anchors = ([10,13, 16,30, 33,23], [30,61, 62,45, 59,119], [116,90, 156,198, 373,326])
    ch = (128, 256, 512)
    
    model = MultiHeadDetect(nc=nc, anchors=anchors, ch=ch, n_heads=4)
    
    # 檢查是否可以設置 stride（Model 類會設置）
    model.stride = torch.tensor([8., 16., 32.])
    assert model.stride is not None, "Cannot set stride"
    
    # 檢查是否可以設置 anchors（Model 類可能會調整）
    assert model.anchors is not None, "Anchors not initialized"
    
    # 檢查 m 屬性（用於打印模型結構）
    # MultiHeadDetect has 3 reg_obj convs + 4 heads * 3 scales = 15 total
    assert len(model.m) == 15, "m attribute wrong length"
    
    # 測試不同 batch size
    for batch_size in [1, 2, 4]:
        model.eval()
        x = [
            torch.randn(batch_size, ch[0], 80, 80),
            torch.randn(batch_size, ch[1], 40, 40),
            torch.randn(batch_size, ch[2], 20, 20),
        ]
        
        with torch.no_grad():
            pred, _ = model(x)
        
        assert pred.shape[0] == batch_size, f"Wrong batch size handling for batch={batch_size}"
    
    print("✓ Model integration ready")
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Step 5: Interface Compatibility...")
    print("=" * 60)
    
    tests = [
        test_interface_compatibility,
        test_training_output_format,
        test_inference_output_format,
        test_loss_compatibility,
        test_model_integration_ready
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
        print("\n✅ All Step 5 compatibility tests passed!")