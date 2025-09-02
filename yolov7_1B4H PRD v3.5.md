# YOLOv7 多頭檢測架構 (1B4H) 產品需求文檔 PRD v3.5

## 執行摘要

本文檔定義 YOLOv7-tiny 從單一檢測頭改造為四檢測頭（1B4H）架構的產品需求，基於 v3.0 版本修正關鍵問題並增加實施細節。專案預估需要 8-10 週開發時間，包含概念驗證階段，目標達成 mAP@0.5 提升 2-3%。

**v3.5 關鍵更新**：

- 修正類別分組錯誤，基於 COCO supercategory 重新設計
- 增加策略 A 實施細節規範
- 新增 320x320 解析度專項優化
- 調整四階段時間規劃，增加 PoC 階段
- 明確內存和計算開銷預算

------

## 1. 專案概述與目標

### 1.1 改造範圍定義

| 項目           | 規格                        | 備註            |
| -------------- | --------------------------- | --------------- |
| **基準模型**   | YOLOv7-tiny (6.0M 參數)     | GitHub 官方版本 |
| **目標架構**   | 1B4H (單一骨幹，四個檢測頭) | 策略 A 實現     |
| **輸入解析度** | 320×320                     | 需特殊優化      |
| **數據集**     | COCO 2017                   | 80 類別         |
| **性能目標**   | mAP@0.5 提升 2-3%           | 相對值          |
| **速度要求**   | FPS ≥ 基準的 85%            | RTX 3090 測試   |
| **模型大小**   | 增加 ≤ 35%                  | 考慮 4 頭開銷   |

### 1.2 核心改造策略

採用**策略 A（全局定位，專門分類）**：

- 每個頭輸出完整 85 維（80類 + 4框 + 1物件性）
- 所有頭共享相同的錨框配置
- 損失計算：Objectness 和 Box 全局，Classification 遮罩

------

## 2. 類別分組策略（v3.5 修正版）

### 2.1 基於 COCO Supercategory 的分組

```python
# 完整 80 類別分配（無重複，無遺漏）
head_assignments = {
    'head_0': {  # 人物與運動用品（20類）
        'classes': [0, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 76],
        'names': ['person', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 
                 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
                 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
                 'cup', 'fork', 'knife', 'teddy bear'],
        'supercategory': 'person & sports',
        'expected_samples': '~35%',
        'initial_weight': 1.0
    },
    
    'head_1': {  # 交通工具與戶外設施（20類）
        'classes': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 24, 25, 72, 73, 74, 75, 77],
        'names': ['bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
                 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter',
                 'bench', 'backpack', 'umbrella', 'book', 'clock', 'vase', 'scissors',
                 'hair drier'],
        'supercategory': 'vehicle & outdoor',
        'expected_samples': '~25%',
        'initial_weight': 1.2
    },
    
    'head_2': {  # 動物與食物（20類）
        'classes': [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53],
        'names': ['bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
                 'zebra', 'giraffe', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
                 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza'],
        'supercategory': 'animal & food',
        'expected_samples': '~20%',
        'initial_weight': 1.5
    },
    
    'head_3': {  # 家具與電子產品（20類）
        'classes': [54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 78, 79],
        'names': ['donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 
                 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                 'refrigerator', 'toothbrush'],
        'supercategory': 'furniture & appliance',
        'expected_samples': '~20%',
        'initial_weight': 1.3
    }
}
```

### 2.2 320×320 解析度專用錨框配置

```python
# 針對 320×320 優化的錨框（縮放自 640×640）
anchors_320 = [
    [5, 6, 8, 15, 16, 11],        # P3/8 → 40×40 網格
    [15, 30, 31, 22, 29, 59],     # P4/16 → 20×20 網格  
    [58, 45, 78, 99, 186, 163]    # P5/32 → 10×10 網格
]

# 超參數調整
hyperparameters_320 = {
    'anchor_t': 3.5,  # 從 4.0 降低，適應小物體
    'obj': 1.0,       # 提高物體損失權重
    'cls': 0.5,       # 標準分類權重
    'box': 0.05,      # 標準框回歸權重
    'mosaic': 0.5,    # 降低 mosaic 比例
    'mixup': 0.1      # 減少 mixup 強度
}
```

------

## 3. 策略 A 實施規範（v3.5 詳細版）

### 3.1 MultiHeadDetect 架構規範

```python
class MultiHeadDetect(nn.Module):
    """
    策略 A 多頭檢測層規範
    - 每個頭輸出完整 85 維
    - 共享錨框和 stride 計算
    - 兼容 YOLOv7 導出流程
    """
    def __init__(self, nc=80, anchors=(), ch=(), n_heads=4):
        super().__init__()
        self.nc = nc
        self.n_heads = n_heads
        self.no = nc + 5  # 85 維輸出
        self.nl = len(anchors)  # 3 個檢測層
        self.na = len(anchors[0]) // 2  # 3 個錨框
        self.grid = [torch.zeros(1)] * self.nl
        self.stride = None  # 在 Model.build_strides() 中計算
        
        # 註冊錨框（與原版 Detect 兼容）
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))
        
        # 創建 4 個完整檢測頭
        self.heads = nn.ModuleList()
        for i in range(n_heads):
            head = nn.ModuleList()
            for x in ch:
                # 關鍵：每個頭都輸出 85 維
                head.append(nn.Conv2d(x, self.no * self.na, 1))
            self.heads.append(head)
            
        # 兼容性：self.m 指向第一個頭
        self.m = self.heads[0]
```

### 3.2 損失計算規範

```python
class ComputeLossMultiHead:
    """
    策略 A 損失計算規範
    - Objectness: 全局計算（所有物體都是前景）
    - Box Regression: 全局計算（所有框都參與）
    - Classification: 遮罩計算（只負責的類別）
    """
    def compute_loss_per_head(self, pred, targets, head_id):
        # 1. 構建目標（所有頭使用相同目標）
        tcls, tbox, indices, anchors = self.build_targets(pred, targets)
        
        # 2. Objectness 損失（全局）
        obj_targets = torch.ones_like(pred[..., 4])  # 所有物體都是前景
        obj_loss = self.BCEobj(pred[..., 4], obj_targets)
        
        # 3. Box 回歸損失（全局）
        box_loss = self.compute_box_loss(pred[..., :4], tbox, indices, anchors)
        
        # 4. Classification 損失（遮罩）
        class_mask = self.get_class_mask(head_id)
        cls_pred = pred[..., 5:]
        
        # 關鍵：只對負責的類別計算損失
        responsible_mask = class_mask[tcls.long()]
        if responsible_mask.sum() > 0:
            cls_loss = self.BCEcls(
                cls_pred[responsible_mask],
                tcls[responsible_mask]
            )
        else:
            cls_loss = torch.zeros(1, device=pred.device)
            
        return obj_loss, box_loss, cls_loss
```

------

## 4. 五階段實施策略（v3.5 調整版）

### 階段 0：概念驗證 PoC（第 0 週，新增）

**目標**：用 10% 數據驗證策略 A 可行性

| 任務                   | 工作量 | 驗證點       |
| ---------------------- | ------ | ------------ |
| 簡化版 MultiHeadDetect | 8h     | 前向傳播成功 |
| 基礎損失計算           | 8h     | 損失下降     |
| Mini 訓練（1000 張圖） | 8h     | 無 NaN，收斂 |
| 320×320 適配測試       | 8h     | 小物體檢測   |

**Go/No-Go 決策點**：

- ✅ 損失正常下降
- ✅ 4 頭輸出維度正確
- ✅ 內存增長 < 40%

### 階段 1：核心架構實現（第 1-2 週）

**目標**：完整實現策略 A 架構

| 模組                 | 文件                               | 優先級 | 工作量 |
| -------------------- | ---------------------------------- | ------ | ------ |
| MultiHeadConfig      | utils/multihead_config.py          | P0     | 6h     |
| MultiHeadDetect      | models/yolo.py                     | P0     | 12h    |
| ComputeLossMultiHead | utils/loss.py                      | P0     | 12h    |
| 配置文件             | cfg/training/yolov7-tiny-1b4h.yaml | P0     | 4h     |
| 單元測試             | test/test_multihead.py             | P0     | 6h     |

### 階段 2：訓練穩定性（第 3-5 週，延長）

**目標**：確保 320×320 訓練穩定

| 任務                | 預期結果         | 時間  |
| ------------------- | ---------------- | ----- |
| 10 epochs 測試      | 損失穩定下降     | 1週   |
| 動態權重調整        | 4頭平衡(std<0.2) | 0.5週 |
| 50 epochs 訓練      | mAP > 基準 90%   | 1週   |
| 100 epochs 完整訓練 | mAP > 基準 95%   | 0.5週 |

### 階段 3：推理優化（第 6-7 週）

**目標**：優化推理性能

| 優化項     | 目標       | 方法       |
| ---------- | ---------- | ---------- |
| 兩階段 NMS | 重複率<5%  | 跨類別抑制 |
| 向量化運算 | FPS提升20% | torch.jit  |
| 內存優化   | 降低15%    | 輸出合併   |

### 階段 4：性能調優（第 8-10 週，延長）

**目標**：達到目標性能

| 調優維度 | 搜索空間 | 預期改進      |
| -------- | -------- | ------------- |
| 頭權重   | ±20%     | mAP +0.5%     |
| NMS 閾值 | 0.2-0.5  | 重複率 -2%    |
| 損失權重 | ±50%     | 收斂速度 +15% |
| 數據增強 | 5種組合  | mAP +0.3%     |

------

## 5. 風險管理（v3.5 更新）

### 5.1 技術風險矩陣

| 風險            | 概率 | 影響 | 緩解措施            | 責任人     |
| --------------- | ---- | ---- | ------------------- | ---------- |
| 策略A內存爆炸   | 中   | 高   | 梯度累積，減小batch | 架構師     |
| 320×320性能退化 | 高   | 中   | 專用錨框，FPN調整   | 算法工程師 |
| 4頭不平衡       | 中   | 中   | 動態權重，重採樣    | 訓練工程師 |
| 推理速度不達標  | 低   | 高   | TensorRT優化        | 部署工程師 |

### 5.2 回滾計劃

1. **快速回滾**：配置開關 `use_multihead: false`
2. **分支策略**：在 `feature/1b4h` 分支開發
3. **檢查點**：每階段結束設置 Go/No-Go 決策點

------

## 6. 驗收標準

### 6.1 功能驗收

- [ ] 4個檢測頭正常工作
- [ ] 80類別無重複完整覆蓋
- [ ] 策略A損失計算正確
- [ ] 兩階段NMS實現

### 6.2 性能驗收

| 指標         | 目標值        | 測試條件        |
| ------------ | ------------- | --------------- |
| mAP@0.5      | +2-3%         | COCO val2017    |
| mAP@0.5:0.95 | +1-2%         | COCO val2017    |
| FPS          | ≥85% baseline | RTX 3090, BS=1  |
| 模型大小     | ≤8.1M params  | FP32            |
| 顯存占用     | ≤2.7GB        | Training, BS=16 |

### 6.3 品質驗收

- [ ] 單元測試覆蓋率 >80%
- [ ] 無內存洩漏
- [ ] 訓練 200 epochs 穩定
- [ ] ONNX/TensorRT 可導出

------

## 7. 附錄

### 7.1 參考資料

- YOLOv7 官方實現：https://github.com/WongKinYiu/yolov7
- COCO 數據集：https://cocodataset.org
- 策略 A 論文：[待補充]

### 7.2 術語表

- **1B4H**: 1 Backbone 4 Heads
- **策略 A**: 全局定位，專門分類
- **PoC**: Proof of Concept
- **mAP**: mean Average Precision

------

*文檔版本：v3.5*
 *更新日期：2024-01-XX*
 *狀態：待審核*
 *下一版本：v4.0（實施後更新）*