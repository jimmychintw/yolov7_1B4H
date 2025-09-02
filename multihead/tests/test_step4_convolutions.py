import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import torch
import torch.nn as nn

def test_convolution_layers():
    """測試卷積層是否正確創建"""
    from models.yolo import MultiHeadDetect
    
    nc = 80
    anchors = ([10,13, 16,30, 33,23], [30,61, 62,45, 59,119], [116,90, 156,198, 373,326])
    ch = (128, 256, 512)  # 輸入通道數
    
    model = MultiHeadDetect(nc=nc, anchors=anchors, ch=ch, n_heads=4)
    
    # 檢查 reg_obj 卷積層
    assert len(model.reg_obj_convs) == 3, f"Should have 3 reg_obj convs, got {len(model.reg_obj_convs)}"
    
    for i, conv in enumerate(model.reg_obj_convs):
        assert isinstance(conv, nn.Conv2d), f"reg_obj_convs[{i}] should be Conv2d"
        assert conv.in_channels == ch[i], f"Input channels mismatch at layer {i}"
        assert conv.out_channels == model.na * 5, f"Output should be na*5 for box+obj"
        assert conv.kernel_size == (1, 1), "Should be 1x1 convolution"
        
    print("✓ Reg/Obj convolution layers correct")
    
    # 檢查 cls 卷積層
    assert len(model.cls_convs) == 4, "Should have 4 classification heads"
    
    for head_id in range(4):
        head_convs = model.cls_convs[head_id]
        assert len(head_convs) == 3, f"Head {head_id} should have 3 conv layers"
        
        for i, conv in enumerate(head_convs):
            assert isinstance(conv, nn.Conv2d), f"cls_convs[{head_id}][{i}] should be Conv2d"
            assert conv.in_channels == ch[i], f"Input channels mismatch"
            assert conv.out_channels == model.na * nc, f"Output should be na*nc for classes"
            assert conv.kernel_size == (1, 1), "Should be 1x1 convolution"
    
    print("✓ Classification convolution layers correct")
    
    return True

def test_m_attribute():
    """測試 m 屬性是否正確設置（用於 bias 初始化）"""
    from models.yolo import MultiHeadDetect
    
    nc = 80
    anchors = ([10,13, 16,30, 33,23], [30,61, 62,45, 59,119], [116,90, 156,198, 373,326])
    ch = (128, 256, 512)
    
    model = MultiHeadDetect(nc=nc, anchors=anchors, ch=ch, n_heads=4)
    
    # m 應該包含所有輸出卷積層
    assert hasattr(model, 'm'), "Should have 'm' attribute"
    assert isinstance(model.m, nn.ModuleList), "'m' should be ModuleList"
    
    # 計算預期的卷積層數量
    # 3 個 reg_obj + 4 heads * 3 layers = 15 total
    expected_conv_count = 3 + 4 * 3
    assert len(model.m) == expected_conv_count, f"Should have {expected_conv_count} convs in 'm'"
    
    # 所有都應該是 Conv2d
    for i, conv in enumerate(model.m):
        assert isinstance(conv, nn.Conv2d), f"m[{i}] should be Conv2d"
        assert hasattr(conv, 'bias'), f"Conv at m[{i}] should have bias"
    
    print(f"✓ 'm' attribute correct with {len(model.m)} convolutions")
    
    return True

def test_parameter_count():
    """測試參數數量是否合理"""
    from models.yolo import MultiHeadDetect
    
    nc = 80
    anchors = ([10,13, 16,30, 33,23], [30,61, 62,45, 59,119], [116,90, 156,198, 373,326])
    ch = (128, 256, 512)
    na = 3  # anchors per layer
    
    model = MultiHeadDetect(nc=nc, anchors=anchors, ch=ch, n_heads=4)
    
    # 計算預期參數
    # Reg/Obj: 3 layers * (ch[i] * 5 * na + 5 * na)
    reg_obj_params = sum((ch[i] + 1) * 5 * na for i in range(3))
    
    # Cls: 4 heads * 3 layers * (ch[i] * nc * na + nc * na)
    cls_params = 4 * sum((ch[i] + 1) * nc * na for i in range(3))
    
    expected_params = reg_obj_params + cls_params
    
    # 計算實際參數
    actual_params = sum(p.numel() for p in model.parameters())
    
    print(f"✓ Parameters: {actual_params:,} (expected ~{expected_params:,})")
    
    # 允許小誤差
    assert abs(actual_params - expected_params) < 100, "Parameter count mismatch"
    
    return True

def test_forward_shape():
    """測試前向傳播輸出形狀"""
    from models.yolo import MultiHeadDetect
    
    nc = 80
    anchors = ([10,13, 16,30, 33,23], [30,61, 62,45, 59,119], [116,90, 156,198, 373,326])
    ch = (128, 256, 512)
    
    model = MultiHeadDetect(nc=nc, anchors=anchors, ch=ch, n_heads=4)
    
    # 創建測試輸入
    batch_size = 2
    x = [
        torch.randn(batch_size, ch[0], 80, 80),  # P3
        torch.randn(batch_size, ch[1], 40, 40),  # P4
        torch.randn(batch_size, ch[2], 20, 20),  # P5
    ]
    
    # 測試前向傳播（目前還是返回 x）
    model.eval()
    with torch.no_grad():
        output = model(x)
    
    # Step 4 仍然返回原始輸入（forward 還未實現）
    assert len(output) == 3, "Should return list of 3 tensors for now"
    
    print("✓ Forward pass shape check (placeholder)")
    
    return True

def test_no_identity_layers():
    """確保沒有 Identity 佔位層了"""
    from models.yolo import MultiHeadDetect
    
    nc = 80
    anchors = ([10,13, 16,30, 33,23], [30,61, 62,45, 59,119], [116,90, 156,198, 373,326])
    ch = (128, 256, 512)
    
    model = MultiHeadDetect(nc=nc, anchors=anchors, ch=ch, n_heads=4)
    
    # 檢查所有模組
    for name, module in model.named_modules():
        assert not isinstance(module, nn.Identity), f"Found Identity layer at {name}"
    
    print("✓ No Identity placeholder layers")
    
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Step 4: Convolution implementation...")
    print("=" * 60)
    
    tests = [
        test_convolution_layers,
        test_m_attribute,
        test_parameter_count,
        test_forward_shape,
        test_no_identity_layers
    ]
    
    all_passed = True
    for test in tests:
        try:
            test()
        except AssertionError as e:
            print(f"✗ Test failed: {e}")
            all_passed = False
            break
        except Exception as e:
            print(f"✗ Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False
            break
    
    if all_passed:
        print("\n✅ All Step 4 tests passed!")