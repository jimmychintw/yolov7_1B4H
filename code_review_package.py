#!/usr/bin/env python3
"""
生成 Code Review 包，方便其他 AI 系統審查
"""

import os
import shutil
import subprocess
from datetime import datetime

def create_review_package():
    """創建 code review 包"""
    
    # 創建輸出目錄
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"code_review_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Creating code review package in: {output_dir}/")
    
    # 1. 複製核心檔案
    core_files = {
        "1_配置檔案": [
            "data/coco-multihead.yaml",
            "cfg/training/yolov7-tiny-multihead-proper.yaml",
        ],
        "2_核心實作": [
            "utils/multihead_utils.py",
            "utils/loss_multihead.py",
        ],
        "3_測試檔案": [
            "multihead/tests/test_step5_forward.py",
            "multihead/tests/test_step6_loss.py",
            "multihead/tests/test_step7_simple.py",
        ],
        "4_文檔": [
            "CODE_REVIEW_GUIDE.md",
            "multihead/DEVELOPMENT_LOG.md",
        ]
    }
    
    for category, files in core_files.items():
        cat_dir = os.path.join(output_dir, category)
        os.makedirs(cat_dir, exist_ok=True)
        for file in files:
            if os.path.exists(file):
                shutil.copy2(file, cat_dir)
                print(f"  ✓ Copied {file}")
    
    # 2. 生成 diff 檔案
    diff_dir = os.path.join(output_dir, "5_程式碼差異")
    os.makedirs(diff_dir, exist_ok=True)
    
    # models/yolo.py 的差異
    if os.path.exists("models/yolo.py.backup_step3"):
        subprocess.run([
            "diff", "-u", "models/yolo.py.backup_step3", "models/yolo.py"
        ], stdout=open(os.path.join(diff_dir, "yolo.py.diff"), "w"))
        print("  ✓ Generated yolo.py diff")
    
    # train.py 的差異
    if os.path.exists("train.py.backup_step7"):
        subprocess.run([
            "diff", "-u", "train.py.backup_step7", "train.py"
        ], stdout=open(os.path.join(diff_dir, "train.py.diff"), "w"))
        print("  ✓ Generated train.py diff")
    
    # 3. 提取關鍵程式碼片段
    snippets_dir = os.path.join(output_dir, "6_關鍵程式碼片段")
    os.makedirs(snippets_dir, exist_ok=True)
    
    # 提取 MultiHeadDetect 類
    extract_class("models/yolo.py", "MultiHeadDetect", 
                  os.path.join(snippets_dir, "MultiHeadDetect.py"))
    
    # 4. 創建 README
    create_readme(output_dir)
    
    print(f"\n✅ Code review package created: {output_dir}/")
    print("\n建議的使用方式：")
    print("1. 將整個資料夾壓縮：tar -czf code_review.tar.gz " + output_dir)
    print("2. 提供給其他 AI 系統時，先給 CODE_REVIEW_GUIDE.md")
    print("3. 根據 AI 的反饋，提供特定的檔案內容")
    
    return output_dir

def extract_class(file_path, class_name, output_path):
    """提取特定類別的程式碼"""
    if not os.path.exists(file_path):
        return
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    in_class = False
    class_lines = []
    indent_level = 0
    
    for line in lines:
        if f"class {class_name}" in line:
            in_class = True
            indent_level = len(line) - len(line.lstrip())
            class_lines.append(line)
        elif in_class:
            current_indent = len(line) - len(line.lstrip())
            if line.strip() and current_indent <= indent_level:
                break
            class_lines.append(line)
    
    if class_lines:
        with open(output_path, 'w') as f:
            f.writelines(class_lines)
        print(f"  ✓ Extracted {class_name} class")

def create_readme(output_dir):
    """創建 README 檔案"""
    readme_content = """# YOLOv7 MultiHead Code Review Package

## 專案摘要
- **目標**: 將 YOLOv7-tiny 改造為 4 頭檢測架構 (1B4H)
- **策略**: Strategy A - 共享 box/objectness，分離 classification
- **新增程式碼**: ~1000 行
- **修改原始碼**: <50 行

## 資料夾結構
- `1_配置檔案/`: YAML 配置
- `2_核心實作/`: 主要實作檔案
- `3_測試檔案/`: 單元測試
- `4_文檔/`: 設計文檔和指南
- `5_程式碼差異/`: diff 檔案
- `6_關鍵程式碼片段/`: 提取的關鍵類別

## 審查優先順序
1. **最高**: MultiHeadDetect.forward() 和 ComputeLossMultiHead
2. **高**: train.py 整合邏輯
3. **中**: 配置管理和測試覆蓋
4. **低**: 文檔和程式碼風格

## 關鍵問題
1. Strategy A 實作是否正確？
2. 梯度流是否正常？
3. 記憶體使用是否合理？
4. 與原始 YOLOv7 的相容性？

請先閱讀 CODE_REVIEW_GUIDE.md 了解詳細資訊。
"""
    
    with open(os.path.join(output_dir, "README.md"), 'w') as f:
        f.write(readme_content)
    print("  ✓ Created README.md")

if __name__ == "__main__":
    create_review_package()