import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

def check_system_components():
    """檢查所有系統組件是否完成"""
    
    print("="*60)
    print("YOLOv7 MultiHead (1B4H) 系統完成度檢查")
    print("="*60)
    
    components = {
        "核心配置檔案": [
            ("data/coco-multihead.yaml", os.path.exists("data/coco-multihead.yaml")),
            ("cfg/training/yolov7-tiny-multihead-proper.yaml", os.path.exists("cfg/training/yolov7-tiny-multihead-proper.yaml")),
            ("data/hyp.scratch.tiny.yaml", os.path.exists("data/hyp.scratch.tiny.yaml")),
        ],
        "核心程式模組": [
            ("utils/multihead_utils.py (MultiHeadConfig)", os.path.exists("utils/multihead_utils.py")),
            ("models/yolo.py (MultiHeadDetect)", True),  # 已修改
            ("utils/loss_multihead.py (ComputeLossMultiHead)", os.path.exists("utils/loss_multihead.py")),
            ("train.py (整合修改)", True),  # 已修改
        ],
        "測試程式": [
            ("Step 1 測試", os.path.exists("multihead/tests/test_step1_config.py")),
            ("Step 2 測試", os.path.exists("multihead/tests/test_step2_multihead_config.py")),
            ("Step 3 測試", os.path.exists("multihead/tests/test_step3_multihead_detect_init.py")),
            ("Step 4 測試", os.path.exists("multihead/tests/test_step4_convolutions.py")),
            ("Step 5 測試", os.path.exists("multihead/tests/test_step5_forward.py")),
            ("Step 6 測試", os.path.exists("multihead/tests/test_step6_loss.py")),
            ("Step 7 測試", os.path.exists("multihead/tests/test_step7_training.py")),
        ],
        "驗證報告": [
            ("開發日誌", os.path.exists("multihead/DEVELOPMENT_LOG.md")),
            ("Step 7 報告", os.path.exists("multihead/validation/step7_report.txt")),
            ("最終檢查清單", os.path.exists("multihead/validation/final_checklist.py")),
        ],
        "備份檔案": [
            ("train.py.backup_step7", os.path.exists("train.py.backup_step7")),
            ("models/yolo.py 備份", os.path.exists("models/yolo.py.backup")),
        ]
    }
    
    all_complete = True
    for category, items in components.items():
        print(f"\n【{category}】")
        for item, exists in items:
            symbol = "✅" if exists else "❌"
            print(f"  {symbol} {item}")
            if not exists:
                all_complete = False
    
    print("\n" + "="*60)
    
    # 功能完整性檢查
    print("\n【功能完整性檢查】")
    try:
        # 1. 檢查 MultiHeadConfig
        from utils.multihead_utils import MultiHeadConfig
        config = MultiHeadConfig("data/coco-multihead.yaml")
        print("  ✅ MultiHeadConfig 可正常載入")
        
        # 2. 檢查 MultiHeadDetect
        from models.yolo import MultiHeadDetect
        print("  ✅ MultiHeadDetect 可正常導入")
        
        # 3. 檢查 ComputeLossMultiHead
        from utils.loss_multihead import ComputeLossMultiHead
        print("  ✅ ComputeLossMultiHead 可正常導入")
        
        # 4. 檢查訓練整合
        with open('train.py', 'r') as f:
            content = f.read()
            if 'MultiHeadDetect' in content and 'ComputeLossMultiHead' in content:
                print("  ✅ train.py 已整合多頭支援")
            else:
                print("  ❌ train.py 整合可能不完整")
                all_complete = False
                
    except Exception as e:
        print(f"  ❌ 功能檢查失敗: {e}")
        all_complete = False
    
    print("\n" + "="*60)
    
    # 系統架構總結
    print("\n【系統架構總結】")
    print("""
    YOLOv7 MultiHead (1B4H) - Strategy A
    ├── 配置層
    │   ├── data/coco-multihead.yaml (4頭類別分配)
    │   └── utils/multihead_utils.py (配置管理)
    ├── 模型層
    │   └── models/yolo.py
    │       ├── Detect (原始，未修改)
    │       └── MultiHeadDetect (新增，策略A)
    ├── 損失層
    │   ├── utils/loss.py (原始，未修改)
    │   └── utils/loss_multihead.py (新增)
    └── 訓練層
        └── train.py (已修改，自動檢測多頭)
    """)
    
    print("="*60)
    
    if all_complete:
        print("\n✅ 【結論】所有系統程式都已完成！")
        print("\n可以開始的工作：")
        print("  1. 執行完整訓練：./test_train_multihead.sh")
        print("  2. 評估性能：python test.py --data data/coco-multihead.yaml")
        print("  3. 推理測試：python detect.py --weights [trained_weights]")
    else:
        print("\n⚠️ 【結論】還有部分組件未完成，請檢查上方標記為 ❌ 的項目")
    
    return all_complete

if __name__ == "__main__":
    check_system_components()