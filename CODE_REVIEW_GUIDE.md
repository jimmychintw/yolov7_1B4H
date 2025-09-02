# YOLOv7 MultiHead (1B4H) Code Review Guide

## 專案概述
將 YOLOv7-tiny 從單一檢測頭改造為四檢測頭（1 Backbone 4 Heads）架構，採用 Strategy A（共享 box/objectness，分離 classification）。

## 核心設計原則
1. **最小侵入性**：盡量不修改原始 YOLOv7 程式碼
2. **模組化設計**：新功能放在獨立檔案中
3. **向後相容**：保持與原始 YOLOv7 的完全相容性
4. **策略 A 實現**：共享定位，專門分類

## 需要重點審查的模組

### 1. 配置管理模組
**檔案**: `utils/multihead_utils.py`
**行數**: ~164 行
**關鍵類別**: `MultiHeadConfig`

**審查重點**:
- [ ] 類別分配是否正確（80類分成4組，每組20類）
- [ ] Head mask 生成邏輯是否正確
- [ ] 權重正規化是否合理
- [ ] 記憶體使用是否高效

**關鍵程式碼片段**:
```python
def get_head_mask(self, head_id, device='cpu'):
    """獲取特定頭的類別遮罩"""
    # 檢查這裡的邏輯是否正確
    mask = torch.zeros(self.num_classes, dtype=torch.bool, device=device)
    for cls_id in self.head_assignments[f'head_{head_id}']['classes']:
        mask[cls_id] = True
    return mask
```

### 2. MultiHeadDetect 模型層
**檔案**: `models/yolo.py`
**修改行數**: ~200 行（新增）
**關鍵類別**: `MultiHeadDetect`

**審查重點**:
- [ ] 卷積層結構是否符合 Strategy A
- [ ] Forward 方法的訓練/推理模式切換
- [ ] Grid 生成和座標轉換是否正確
- [ ] 梯度流是否正常

**關鍵程式碼片段**:
```python
class MultiHeadDetect(nn.Module):
    def __init__(self, nc=80, anchors=(), ch=(), n_heads=4):
        # 重點檢查：
        # 1. 共享 reg_obj 卷積（3層）
        # 2. 分離 cls 卷積（4頭 × 3層 = 12層）
        
    def forward(self, x):
        # 重點檢查：
        # 1. 訓練模式輸出格式：(reg_obj_outputs, cls_outputs)
        # 2. 推理模式座標轉換
```

### 3. 損失函數模組
**檔案**: `utils/loss_multihead.py`
**行數**: ~215 行
**關鍵類別**: `ComputeLossMultiHead`

**審查重點**:
- [ ] Strategy A 損失計算邏輯
- [ ] Box/Objectness 全局計算
- [ ] Classification 遮罩過濾
- [ ] 梯度回傳是否正確

**關鍵程式碼片段**:
```python
def __call__(self, p, targets):
    # 重點檢查：
    # 1. reg_obj_outputs, cls_outputs 解包
    # 2. 共享 box/obj 損失
    # 3. 分離 cls 損失（只計算負責的類別）
```

### 4. 訓練整合
**檔案**: `train.py`
**修改行數**: ~20 行
**位置**: Lines 301-319

**審查重點**:
- [ ] 自動檢測邏輯是否正確
- [ ] 損失函數選擇是否恰當
- [ ] 對原始訓練流程的影響

## 測試覆蓋度檢查

### 單元測試
- `test_step1_config.py` - 配置載入
- `test_step2_multihead_config.py` - MultiHeadConfig 功能
- `test_step3_multihead_detect_init.py` - 模型初始化
- `test_step4_convolutions.py` - 卷積層結構
- `test_step5_forward.py` - 前向傳播
- `test_step6_loss.py` - 損失計算
- `test_step7_training.py` - 訓練整合

### 整合測試
```bash
# 執行所有測試
for i in 1 2 3 4 5 6 7; do
    python multihead/tests/test_step${i}*.py
done
```

## 性能考量

### 記憶體使用
- 原始 Detect: 229,245 參數
- MultiHeadDetect: 876,525 參數（增加 282.4%）
- 這個增加是否合理？是否有優化空間？

### 計算效率
- Forward pass 是否有不必要的運算？
- 是否可以使用 torch.jit 優化？

## 潛在問題區域

### 1. Import 路徑不一致
**問題**: `ComputeLossMultiHead` 在獨立檔案 vs 在 loss.py 末尾
```python
# 目前實作
from utils.loss_multihead import ComputeLossMultiHead

# Markdown 期望
from utils.loss import ComputeLossMultiHead
```
**影響**: 功能正常，但與文檔不一致

### 2. 類別分配驗證
需要確認 80 個 COCO 類別是否：
- 完整覆蓋（無遺漏）
- 無重複分配
- 語義分組合理

### 3. 梯度流驗證
特別注意多頭結構中的梯度是否正確回傳到 backbone。

## 建議的 Review 流程

### Phase 1: 靜態分析
1. 程式碼風格一致性
2. 命名規範
3. 註釋完整性
4. 型別提示

### Phase 2: 邏輯審查
1. Strategy A 實作正確性
2. 數學運算正確性
3. 邊界條件處理
4. 錯誤處理

### Phase 3: 性能審查
1. 記憶體使用效率
2. 計算複雜度
3. 可優化點識別

### Phase 4: 整合測試
1. 與原始 YOLOv7 相容性
2. 訓練收斂性
3. 推理正確性

## 具體審查問題

1. **架構設計**
   - Strategy A 的實作是否完全符合設計？
   - 是否有更好的模組化方式？

2. **程式碼品質**
   - 是否有重複程式碼可以重構？
   - 錯誤處理是否完善？

3. **性能優化**
   - 是否有不必要的張量複製？
   - 是否可以使用 inplace 操作？

4. **測試覆蓋**
   - 是否有遺漏的邊界條件？
   - 是否需要更多的整合測試？

## 提供給 Reviewer 的資料

1. **原始 YOLOv7 參考**：https://github.com/WongKinYiu/yolov7
2. **設計文檔**：PRD v3.5.md, SDD v2.3.md
3. **開發日誌**：multihead/DEVELOPMENT_LOG.md
4. **測試結果**：multihead/validation/

## Review 優先級

### 高優先級 🔴
- MultiHeadDetect.forward() 方法
- ComputeLossMultiHead.__call__() 方法
- train.py 的損失選擇邏輯

### 中優先級 🟡
- MultiHeadConfig 類別
- 卷積層初始化
- 測試案例完整性

### 低優先級 🟢
- 文檔一致性
- 程式碼風格
- 註釋完整性