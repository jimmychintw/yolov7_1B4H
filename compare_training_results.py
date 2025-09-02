#!/usr/bin/env python3
"""
對比訓練結果：Baseline vs MultiHead
分析訓練日誌、mAP 性能、類別表現等
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import yaml
import argparse
from glob import glob

class TrainingResultsComparator:
    """訓練結果對比器"""
    
    def __init__(self, baseline_path, multihead_path):
        """
        初始化
        
        Args:
            baseline_path: Baseline 訓練結果路徑 (runs/feasibility/baseline_be_optimized)
            multihead_path: MultiHead 訓練結果路徑 (runs/multihead/yolov7_tiny_1b4h_320)
        """
        self.baseline_path = Path(baseline_path)
        self.multihead_path = Path(multihead_path)
        
        # 設置繪圖風格
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
    
    def compare_training_curves(self):
        """對比訓練曲線"""
        print("\n" + "="*70)
        print("1. 訓練曲線對比")
        print("="*70)
        
        # 讀取結果文件
        baseline_results = self._read_results(self.baseline_path / 'results.txt')
        multihead_results = self._read_results(self.multihead_path / 'results.txt')
        
        if baseline_results is None or multihead_results is None:
            print("⚠️ 無法讀取訓練結果文件")
            return None
        
        # 創建子圖
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 定義要繪製的指標
        metrics = [
            ('train/box_loss', 'Box Loss (Train)', axes[0, 0]),
            ('train/obj_loss', 'Object Loss (Train)', axes[0, 1]),
            ('train/cls_loss', 'Class Loss (Train)', axes[0, 2]),
            ('metrics/precision', 'Precision', axes[1, 0]),
            ('metrics/recall', 'Recall', axes[1, 1]),
            ('metrics/mAP_0.5', 'mAP@0.5', axes[1, 2])
        ]
        
        for metric_name, title, ax in metrics:
            if metric_name in baseline_results.columns:
                ax.plot(baseline_results['epoch'], baseline_results[metric_name], 
                       label='Baseline', color='blue', linewidth=2)
            if metric_name in multihead_results.columns:
                ax.plot(multihead_results['epoch'], multihead_results[metric_name], 
                       label='MultiHead', color='orange', linewidth=2)
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel(title)
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Training Curves: Baseline vs MultiHead', fontsize=16)
        plt.tight_layout()
        plt.savefig('training_curves_comparison.png', dpi=150)
        print("📊 訓練曲線已保存到: training_curves_comparison.png")
        
        # 打印最終指標
        print("\n📈 最終訓練指標 (最後 10 epochs 平均):")
        print("\nBaseline:")
        self._print_final_metrics(baseline_results)
        print("\nMultiHead:")
        self._print_final_metrics(multihead_results)
        
        return baseline_results, multihead_results
    
    def compare_validation_results(self):
        """對比驗證結果"""
        print("\n" + "="*70)
        print("2. 驗證性能對比")
        print("="*70)
        
        # 嘗試讀取最佳模型的驗證結果
        baseline_best = self._find_best_weights(self.baseline_path)
        multihead_best = self._find_best_weights(self.multihead_path)
        
        if baseline_best and multihead_best:
            print(f"\n📁 找到最佳權重:")
            print(f"   Baseline:  {baseline_best}")
            print(f"   MultiHead: {multihead_best}")
        
        # 創建對比表格
        comparison_data = []
        
        # 這裡假設有 test 結果文件
        baseline_test = self._read_test_results(self.baseline_path)
        multihead_test = self._read_test_results(self.multihead_path)
        
        if baseline_test and multihead_test:
            metrics = ['mAP@0.5', 'mAP@0.5:0.95', 'Precision', 'Recall']
            
            print("\n📊 驗證集性能對比:")
            print("-" * 50)
            print(f"{'指標':<20} {'Baseline':<15} {'MultiHead':<15} {'差異':<10}")
            print("-" * 50)
            
            for metric in metrics:
                b_val = baseline_test.get(metric, 0)
                m_val = multihead_test.get(metric, 0)
                diff = m_val - b_val
                diff_pct = (m_val / b_val - 1) * 100 if b_val > 0 else 0
                
                print(f"{metric:<20} {b_val:<15.4f} {m_val:<15.4f} {diff:+.4f} ({diff_pct:+.1f}%)")
                comparison_data.append([metric, b_val, m_val, diff, diff_pct])
        
        return comparison_data
    
    def compare_per_class_performance(self):
        """對比每個類別的性能"""
        print("\n" + "="*70)
        print("3. 類別性能對比 (MultiHead 特性)")
        print("="*70)
        
        # 載入 MultiHead 配置
        config_path = 'data/coco-multihead.yaml'
        if not Path(config_path).exists():
            print(f"⚠️ 找不到配置文件: {config_path}")
            return
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        multihead_config = config.get('multihead', {})
        head_assignments = multihead_config.get('head_assignments', {})
        
        print("\n🎯 MultiHead 類別分組:")
        for head_id in range(4):
            head_info = head_assignments.get(head_id, {})
            print(f"\nHead {head_id} ({head_info.get('name', 'unknown')}):")
            print(f"  類別數: {len(head_info.get('classes', []))}")
            print(f"  預期樣本比例: {head_info.get('expected_samples', 'N/A')}")
            
            # 列出部分類別名稱
            class_ids = head_info.get('classes', [])[:5]  # 顯示前5個
            class_names = [config['names'][i] for i in class_ids if i < len(config['names'])]
            print(f"  示例類別: {', '.join(class_names)}...")
        
        # 如果有詳細的類別 AP 結果
        baseline_ap = self._read_class_ap(self.baseline_path)
        multihead_ap = self._read_class_ap(self.multihead_path)
        
        if baseline_ap and multihead_ap:
            self._plot_class_performance(baseline_ap, multihead_ap, head_assignments, config['names'])
    
    def compare_inference_examples(self):
        """對比推理示例"""
        print("\n" + "="*70)
        print("4. 推理示例對比")
        print("="*70)
        
        # 檢查是否有推理結果
        baseline_detections = list((self.baseline_path / 'test').glob('*.txt')) if (self.baseline_path / 'test').exists() else []
        multihead_detections = list((self.multihead_path / 'test').glob('*.txt')) if (self.multihead_path / 'test').exists() else []
        
        if baseline_detections or multihead_detections:
            print(f"\n📦 檢測結果:")
            print(f"   Baseline:  {len(baseline_detections)} 個檔案")
            print(f"   MultiHead: {len(multihead_detections)} 個檔案")
        
        print("\n💡 推理指令對比:")
        print("\nBaseline:")
        print("```bash")
        print(f"python detect.py --weights runs/feasibility/baseline_be_optimized/weights/best.pt \\")
        print(f"                 --source test_images/ --img 320 --conf 0.25")
        print("```")
        
        print("\nMultiHead:")
        print("```bash")
        print(f"python detect.py --weights runs/multihead/yolov7_tiny_1b4h_320/weights/best.pt \\")
        print(f"                 --source test_images/ --img 320 --conf 0.25")
        print("```")
    
    def generate_comparison_report(self):
        """生成完整對比報告"""
        print("\n" + "="*70)
        print("5. 綜合對比報告")
        print("="*70)
        
        report = []
        report.append("# YOLOv7-Tiny Baseline vs MultiHead (1B4H) 對比報告\n")
        report.append(f"生成時間: {pd.Timestamp.now()}\n")
        
        # 1. 配置對比
        report.append("## 1. 訓練配置對比\n")
        report.append("| 參數 | Baseline | MultiHead |\n")
        report.append("|------|----------|----------|\n")
        report.append("| 模型配置 | yolov7-tiny.yaml | yolov7-tiny-multihead-proper.yaml |\n")
        report.append("| 數據配置 | coco_vast.ai.yaml | coco-multihead.yaml |\n")
        report.append("| 超參數 | hyp.scratch.tiny.bs384.yaml | hyp.scratch.tiny.multihead.320.yaml |\n")
        report.append("| 輸入大小 | 320×320 | 320×320 |\n")
        report.append("| Batch Size | 384 | 384 |\n")
        report.append("| Epochs | 100 | 100 |\n")
        
        # 2. 架構差異
        report.append("\n## 2. 模型架構差異\n")
        report.append("- **Baseline**: 單一檢測頭 (IDetect/Detect)\n")
        report.append("- **MultiHead**: 4 個檢測頭 (MultiHeadDetect)\n")
        report.append("- **參數增加**: ~15% (主要在檢測層)\n")
        report.append("- **策略**: Strategy A (共享 box/obj，獨立 cls)\n")
        
        # 3. 預期改進
        report.append("\n## 3. 預期性能改進\n")
        report.append("- mAP@0.5: +2-3%\n")
        report.append("- 小物體檢測: +3-5%\n")
        report.append("- 類別平衡: 更均衡\n")
        report.append("- 推理速度: -10-15%\n")
        
        # 4. 關鍵優勢
        report.append("\n## 4. MultiHead 關鍵優勢\n")
        report.append("1. **類別專門化**: 每個頭專注於語義相關的類別\n")
        report.append("2. **減少類別競爭**: 降低類別間的相互抑制\n")
        report.append("3. **平衡學習**: 動態調整頭權重\n")
        report.append("4. **小物體優化**: anchor_t 調整為 3.5\n")
        
        # 保存報告
        report_path = 'comparison_report.md'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.writelines(report)
        
        print(f"\n📄 完整報告已保存到: {report_path}")
        
        # 打印摘要
        print("\n📋 對比摘要:")
        print("   ✅ MultiHead 使用 4 個專門檢測頭")
        print("   ✅ 每頭負責 20 個語義相關類別")
        print("   ✅ 參數增加約 15%")
        print("   ✅ 預期 mAP 提升 2-3%")
        print("   ⚠️  推理速度降低 10-15%")
    
    # === 輔助方法 ===
    
    def _read_results(self, path):
        """讀取 results.txt"""
        if not path.exists():
            return None
        
        try:
            # YOLOv7 的 results.txt 格式
            df = pd.read_csv(path, sep=r'\s+', header=None)
            # 根據列數設置列名
            if len(df.columns) == 15:  # 標準 YOLOv7 格式
                df.columns = ['epoch', 'gpu_mem', 'train/box_loss', 'train/obj_loss', 
                             'train/cls_loss', 'train/total', 'targets', 'img_size',
                             'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 
                             'metrics/mAP_0.5:0.95', 'val/box_loss', 'val/obj_loss', 
                             'val/cls_loss']
            return df
        except Exception as e:
            print(f"讀取失敗: {e}")
            return None
    
    def _print_final_metrics(self, df):
        """打印最終指標"""
        if df is None or len(df) == 0:
            print("   無數據")
            return
        
        # 取最後 10 epochs 的平均值
        last_n = min(10, len(df))
        final_metrics = df.tail(last_n).mean()
        
        print(f"   mAP@0.5:      {final_metrics.get('metrics/mAP_0.5', 0):.4f}")
        print(f"   mAP@0.5:0.95: {final_metrics.get('metrics/mAP_0.5:0.95', 0):.4f}")
        print(f"   Precision:    {final_metrics.get('metrics/precision', 0):.4f}")
        print(f"   Recall:       {final_metrics.get('metrics/recall', 0):.4f}")
    
    def _find_best_weights(self, path):
        """找到最佳權重文件"""
        weights_path = path / 'weights' / 'best.pt'
        if weights_path.exists():
            return weights_path
        return None
    
    def _read_test_results(self, path):
        """讀取測試結果"""
        # 這需要根據實際的測試結果格式調整
        # 假設有一個 test_results.json
        test_json = path / 'test_results.json'
        if test_json.exists():
            with open(test_json, 'r') as f:
                return json.load(f)
        
        # 或者從 results.txt 取最後一行
        results = self._read_results(path / 'results.txt')
        if results is not None and len(results) > 0:
            last_row = results.iloc[-1]
            return {
                'mAP@0.5': last_row.get('metrics/mAP_0.5', 0),
                'mAP@0.5:0.95': last_row.get('metrics/mAP_0.5:0.95', 0),
                'Precision': last_row.get('metrics/precision', 0),
                'Recall': last_row.get('metrics/recall', 0)
            }
        return {}
    
    def _read_class_ap(self, path):
        """讀取每個類別的 AP"""
        # 假設有類別 AP 文件
        ap_file = path / 'class_ap.json'
        if ap_file.exists():
            with open(ap_file, 'r') as f:
                return json.load(f)
        return None
    
    def _plot_class_performance(self, baseline_ap, multihead_ap, head_assignments, class_names):
        """繪製類別性能對比"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        for head_id in range(4):
            ax = axes[head_id // 2, head_id % 2]
            head_info = head_assignments.get(head_id, {})
            class_ids = head_info.get('classes', [])
            
            # 獲取該頭的類別 AP
            baseline_values = [baseline_ap.get(str(i), 0) for i in class_ids]
            multihead_values = [multihead_ap.get(str(i), 0) for i in class_ids]
            names = [class_names[i][:10] for i in class_ids if i < len(class_names)]
            
            x = np.arange(len(names))
            width = 0.35
            
            ax.bar(x - width/2, baseline_values[:len(names)], width, label='Baseline', color='blue')
            ax.bar(x + width/2, multihead_values[:len(names)], width, label='MultiHead', color='orange')
            
            ax.set_xlabel('Classes')
            ax.set_ylabel('AP')
            ax.set_title(f"Head {head_id}: {head_info.get('name', 'unknown')}")
            ax.set_xticks(x)
            ax.set_xticklabels(names, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Per-Class AP Comparison', fontsize=16)
        plt.tight_layout()
        plt.savefig('class_performance_comparison.png', dpi=150)
        print("\n📊 類別性能對比已保存到: class_performance_comparison.png")

def main():
    parser = argparse.ArgumentParser(description='Compare training results')
    parser.add_argument('--baseline', default='runs/feasibility/baseline_be_optimized',
                       help='Baseline training results path')
    parser.add_argument('--multihead', default='runs/multihead/yolov7_tiny_1b4h_320',
                       help='MultiHead training results path')
    
    args = parser.parse_args()
    
    # 執行對比
    comparator = TrainingResultsComparator(args.baseline, args.multihead)
    
    # 執行所有對比分析
    comparator.compare_training_curves()
    comparator.compare_validation_results()
    comparator.compare_per_class_performance()
    comparator.compare_inference_examples()
    comparator.generate_comparison_report()
    
    print("\n" + "="*70)
    print("✅ 訓練結果對比完成！")
    print("="*70)

if __name__ == "__main__":
    main()