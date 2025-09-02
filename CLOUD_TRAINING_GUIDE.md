# 雲端雙主機訓練指南

## 🚀 架構配置

### 主機 A - 訓練 Baseline
- **任務**: YOLOv7-Tiny Baseline
- **GPU**: 1x RTX 3090 或更高
- **RAM**: 32GB+
- **儲存**: 100GB+

### 主機 B - 訓練 MultiHead
- **任務**: YOLOv7-Tiny MultiHead (1B4H)
- **GPU**: 1x RTX 3090 或更高
- **RAM**: 32GB+
- **儲存**: 100GB+

---

## 📋 執行步驟

### Step 1: 兩台主機都 Clone 專案

```bash
# 在兩台主機上都執行
git clone https://github.com/jimmychintw/yolov7_1B4H.git
cd yolov7_1B4H

# 安裝依賴
pip install -r requirements.txt
```

### Step 2: 準備 COCO 數據集

```bash
# 兩台主機都需要下載 COCO
# 選項 1: 使用腳本下載
bash scripts/get_coco.sh

# 選項 2: 從您的雲端儲存複製
# gsutil cp -r gs://your-bucket/coco ./
# 或
# aws s3 sync s3://your-bucket/coco ./coco
```

### Step 3: 配置數據路徑

**主機 A & B 都要修改**：

編輯 `data/coco.yaml` 和 `data/coco-multihead.yaml`：
```yaml
# 修改為實際路徑
train: /path/to/coco/train2017.txt
val: /path/to/coco/val2017.txt
test: /path/to/coco/test2017.txt
```

---

## 🖥️ 主機 A - Baseline 訓練

### 啟動訓練腳本

```bash
# 創建訓練腳本
cat > train_baseline_cloud.sh << 'EOF'
#!/bin/bash

# YOLOv7-Tiny Baseline Training on Cloud
echo "Starting Baseline Training on $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Time: $(date)"

python train.py \
  --img-size 320 320 \
  --batch-size 384 \
  --epochs 100 \
  --data data/coco.yaml \
  --cfg cfg/training/yolov7-tiny.yaml \
  --weights '' \
  --hyp data/hyp.scratch.tiny.yaml \
  --device 0 \
  --workers 8 \
  --save_period 25 \
  --project runs/baseline \
  --name yolov7_tiny_baseline_320 \
  --exist-ok \
  --noautoanchor \
  --cache-images

echo "Training completed at $(date)"
EOF

chmod +x train_baseline_cloud.sh

# 使用 nohup 背景執行
nohup ./train_baseline_cloud.sh > baseline_training.log 2>&1 &

# 或使用 screen/tmux
screen -S baseline
./train_baseline_cloud.sh
# Ctrl+A+D 分離
```

### 監控訓練

```bash
# 查看即時日誌
tail -f baseline_training.log

# 查看 GPU 使用
watch -n 1 nvidia-smi

# 查看 TensorBoard
tensorboard --logdir runs/baseline --host 0.0.0.0 --port 6006
```

---

## 🖥️ 主機 B - MultiHead 訓練

### 啟動訓練腳本

```bash
# 創建訓練腳本
cat > train_multihead_cloud.sh << 'EOF'
#!/bin/bash

# YOLOv7-Tiny MultiHead (1B4H) Training on Cloud
echo "Starting MultiHead Training on $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Time: $(date)"

python train.py \
  --img-size 320 320 \
  --batch-size 384 \
  --epochs 100 \
  --data data/coco-multihead.yaml \
  --cfg cfg/training/yolov7-tiny-multihead-proper.yaml \
  --weights '' \
  --hyp data/hyp.scratch.tiny.multihead.320.yaml \
  --device 0 \
  --workers 8 \
  --save_period 25 \
  --project runs/multihead \
  --name yolov7_tiny_1b4h_320 \
  --exist-ok \
  --noautoanchor \
  --cache-images

echo "Training completed at $(date)"
EOF

chmod +x train_multihead_cloud.sh

# 使用 nohup 背景執行
nohup ./train_multihead_cloud.sh > multihead_training.log 2>&1 &

# 或使用 screen/tmux
screen -S multihead
./train_multihead_cloud.sh
# Ctrl+A+D 分離
```

### 監控訓練

```bash
# 查看即時日誌
tail -f multihead_training.log

# 查看 GPU 使用
watch -n 1 nvidia-smi

# 查看 TensorBoard
tensorboard --logdir runs/multihead --host 0.0.0.0 --port 6006
```

---

## 📊 Step 4: 收集結果

### 兩個訓練都完成後

**主機 A - 下載 Baseline 結果**：
```bash
# 打包結果
tar -czf baseline_results.tar.gz runs/baseline/

# 上傳到雲端儲存
# Google Cloud
gsutil cp baseline_results.tar.gz gs://your-bucket/

# AWS
aws s3 cp baseline_results.tar.gz s3://your-bucket/

# 或直接 SCP 到本地
scp baseline_results.tar.gz your_local_machine:/path/to/save/
```

**主機 B - 下載 MultiHead 結果**：
```bash
# 打包結果
tar -czf multihead_results.tar.gz runs/multihead/

# 上傳到雲端儲存
gsutil cp multihead_results.tar.gz gs://your-bucket/
aws s3 cp multihead_results.tar.gz s3://your-bucket/

# 或直接 SCP 到本地
scp multihead_results.tar.gz your_local_machine:/path/to/save/
```

---

## 🔄 Step 5: 合併結果並對比

### 在本地或第三台機器上

```bash
# 下載兩個結果
wget https://your-storage/baseline_results.tar.gz
wget https://your-storage/multihead_results.tar.gz

# 解壓
tar -xzf baseline_results.tar.gz
tar -xzf multihead_results.tar.gz

# Clone 專案（如果還沒有）
git clone https://github.com/jimmychintw/yolov7_1B4H.git
cd yolov7_1B4H

# 執行對比
python compare_training_results.py \
    --baseline runs/baseline/yolov7_tiny_baseline_320 \
    --multihead runs/multihead/yolov7_tiny_1b4h_320

# 生成完整報告
./run_comparison.sh
```

---

## 💰 成本優化建議

### 1. 使用 Spot/Preemptible 實例
```bash
# 設置檢查點自動保存
--save_period 10  # 每 10 epochs 保存

# 支援斷點續訓
--resume runs/xxx/weights/last.pt
```

### 2. 選擇合適的實例

| 雲端平台 | 推薦實例 | GPU | 成本 (約) |
|---------|---------|-----|-----------|
| **AWS** | p3.2xlarge | V100 16GB | $3.06/hr |
| **GCP** | n1-standard-8 + V100 | V100 16GB | $2.48/hr |
| **Azure** | NC6s_v3 | V100 16GB | $3.06/hr |
| **Vast.ai** | RTX 3090 | RTX 3090 24GB | $0.5-1/hr |
| **Lambda Labs** | RTX 3090 | RTX 3090 24GB | $1.10/hr |

### 3. 時間估算

- **100 epochs @ batch 384**: 約 3-4 小時
- **總成本**: $6-24（兩台主機並行）

---

## 🔧 故障排除

### 問題 1: CUDA Out of Memory
```bash
# 降低 batch size
--batch-size 256  # 或 128
```

### 問題 2: 訓練中斷
```bash
# 從檢查點恢復
python train.py --resume runs/xxx/weights/last.pt ...
```

### 問題 3: 網路傳輸慢
```bash
# 使用雲端內部傳輸
# 例如：兩台主機在同一區域，使用內網 IP
```

---

## 📈 實時監控儀表板

### 設置遠端 TensorBoard

**主機 A**:
```bash
tensorboard --logdir runs/baseline --host 0.0.0.0 --port 6006
# 開放防火牆 port 6006
```

**主機 B**:
```bash
tensorboard --logdir runs/multihead --host 0.0.0.0 --port 6007
# 開放防火牆 port 6007
```

**本地訪問**:
- http://主機A-IP:6006 - Baseline
- http://主機B-IP:6007 - MultiHead

### 使用 Weights & Biases (可選)

```bash
# 安裝
pip install wandb

# 登入
wandb login

# 在訓練指令加入
--wandb
```

---

## ✅ 完成檢查清單

- [ ] 兩台主機都已安裝依賴
- [ ] COCO 數據集已準備
- [ ] 數據路徑已配置
- [ ] 訓練腳本已創建
- [ ] 背景訓練已啟動
- [ ] TensorBoard 監控正常
- [ ] 定期檢查訓練進度
- [ ] 結果已下載備份
- [ ] 對比分析已完成

---

## 🎯 預期結果

### 訓練時間（並行）
- **總時間**: 3-4 小時（而非 6-8 小時串行）
- **成本**: 約 $12-24

### 性能對比
- **Baseline mAP@0.5**: X%
- **MultiHead mAP@0.5**: X+2-3%
- **改進**: +2-3%

### 輸出文件
```
baseline_results/
├── weights/
│   ├── best.pt
│   └── last.pt
├── results.txt
└── results.png

multihead_results/
├── weights/
│   ├── best.pt
│   └── last.pt
├── results.txt
└── results.png

comparison_results/
├── comparison_report.md
├── training_curves_comparison.png
└── comparison_results.json
```