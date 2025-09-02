#!/usr/bin/env python3
"""
完整對比 YOLOv7-Tiny Baseline vs MultiHead (1B4H)
包含模型結構、性能指標、推理速度等全方位比較
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
import time
import json
from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns
from models.yolo import Model
import argparse
import yaml

class ModelComparator:
    """模型對比器"""
    
    def __init__(self, baseline_cfg, multihead_cfg, img_size=320):
        """
        初始化對比器
        
        Args:
            baseline_cfg: Baseline 模型配置路徑
            multihead_cfg: MultiHead 模型配置路徑
            img_size: 輸入圖片大小
        """
        self.img_size = img_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 載入模型
        print("載入模型中...")
        self.baseline_model = Model(baseline_cfg).to(self.device)
        self.multihead_model = Model(multihead_cfg).to(self.device)
        
        # 設置為評估模式
        self.baseline_model.eval()
        self.multihead_model.eval()
        
        self.results = {}
    
    def compare_architecture(self):
        """比較模型架構"""
        print("\n" + "="*70)
        print("1. 模型架構對比")
        print("="*70)
        
        results = {}
        
        # 1.1 參數量比較
        baseline_params = sum(p.numel() for p in self.baseline_model.parameters())
        multihead_params = sum(p.numel() for p in self.multihead_model.parameters())
        
        results['parameters'] = {
            'baseline': baseline_params,
            'multihead': multihead_params,
            'difference': multihead_params - baseline_params,
            'increase_pct': (multihead_params / baseline_params - 1) * 100
        }
        
        print(f"\n📊 參數量對比:")
        print(f"   Baseline:  {baseline_params:,} 參數")
        print(f"   MultiHead: {multihead_params:,} 參數")
        print(f"   增加:      {multihead_params - baseline_params:,} ({results['parameters']['increase_pct']:.1f}%)")
        
        # 1.2 檢測層對比
        baseline_det = self.baseline_model.model[-1]
        multihead_det = self.multihead_model.model[-1]
        
        print(f"\n🔍 檢測層對比:")
        print(f"   Baseline:  {type(baseline_det).__name__}")
        print(f"   MultiHead: {type(multihead_det).__name__}")
        
        if hasattr(multihead_det, 'n_heads'):
            print(f"   檢測頭數量: {multihead_det.n_heads}")
        
        # 1.3 層級結構對比
        baseline_layers = len(self.baseline_model.model)
        multihead_layers = len(self.multihead_model.model)
        
        print(f"\n📐 網絡深度:")
        print(f"   Baseline:  {baseline_layers} 層")
        print(f"   MultiHead: {multihead_layers} 層")
        
        # 1.4 檢測層參數詳細對比
        baseline_det_params = sum(p.numel() for p in baseline_det.parameters())
        multihead_det_params = sum(p.numel() for p in multihead_det.parameters())
        
        results['detection_layer'] = {
            'baseline': baseline_det_params,
            'multihead': multihead_det_params,
            'increase_pct': (multihead_det_params / baseline_det_params - 1) * 100
        }
        
        print(f"\n🎯 檢測層參數:")
        print(f"   Baseline:  {baseline_det_params:,}")
        print(f"   MultiHead: {multihead_det_params:,}")
        print(f"   增加:      {(multihead_det_params / baseline_det_params - 1) * 100:.1f}%")
        
        self.results['architecture'] = results
        return results
    
    def compare_inference_speed(self, num_runs=100, batch_sizes=[1, 8, 16, 32]):
        """比較推理速度"""
        print("\n" + "="*70)
        print("2. 推理速度對比")
        print("="*70)
        
        results = {}
        
        for batch_size in batch_sizes:
            print(f"\n⏱️  Batch Size = {batch_size}:")
            
            # 創建假輸入
            x = torch.randn(batch_size, 3, self.img_size, self.img_size).to(self.device)
            
            # Warmup
            for _ in range(10):
                with torch.no_grad():
                    _ = self.baseline_model(x)
                    _ = self.multihead_model(x)
            
            # Baseline 測試
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start = time.time()
            for _ in range(num_runs):
                with torch.no_grad():
                    _ = self.baseline_model(x)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            baseline_time = (time.time() - start) / num_runs
            baseline_fps = batch_size / baseline_time
            
            # MultiHead 測試
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start = time.time()
            for _ in range(num_runs):
                with torch.no_grad():
                    _ = self.multihead_model(x)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            multihead_time = (time.time() - start) / num_runs
            multihead_fps = batch_size / multihead_time
            
            results[f'batch_{batch_size}'] = {
                'baseline_ms': baseline_time * 1000,
                'multihead_ms': multihead_time * 1000,
                'baseline_fps': baseline_fps,
                'multihead_fps': multihead_fps,
                'slowdown_pct': (multihead_time / baseline_time - 1) * 100
            }
            
            print(f"   Baseline:  {baseline_time*1000:.2f} ms ({baseline_fps:.1f} FPS)")
            print(f"   MultiHead: {multihead_time*1000:.2f} ms ({multihead_fps:.1f} FPS)")
            print(f"   速度差異:  {results[f'batch_{batch_size}']['slowdown_pct']:.1f}%")
        
        self.results['inference_speed'] = results
        return results
    
    def compare_memory_usage(self):
        """比較記憶體使用"""
        print("\n" + "="*70)
        print("3. 記憶體使用對比")
        print("="*70)
        
        results = {}
        
        # 模型大小（MB）
        baseline_size = sum(p.numel() * p.element_size() for p in self.baseline_model.parameters()) / 1024 / 1024
        multihead_size = sum(p.numel() * p.element_size() for p in self.multihead_model.parameters()) / 1024 / 1024
        
        results['model_size_mb'] = {
            'baseline': baseline_size,
            'multihead': multihead_size,
            'increase_pct': (multihead_size / baseline_size - 1) * 100
        }
        
        print(f"\n💾 模型大小:")
        print(f"   Baseline:  {baseline_size:.2f} MB")
        print(f"   MultiHead: {multihead_size:.2f} MB")
        print(f"   增加:      {(multihead_size / baseline_size - 1) * 100:.1f}%")
        
        if torch.cuda.is_available():
            # GPU 記憶體使用（訓練時）
            batch_size = 16
            x = torch.randn(batch_size, 3, self.img_size, self.img_size).to(self.device)
            
            # Baseline
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            self.baseline_model.train()
            _ = self.baseline_model(x)
            baseline_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
            
            # MultiHead
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            self.multihead_model.train()
            _ = self.multihead_model(x)
            multihead_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
            
            results['gpu_memory_mb'] = {
                'baseline': baseline_memory,
                'multihead': multihead_memory,
                'increase_pct': (multihead_memory / baseline_memory - 1) * 100
            }
            
            print(f"\n🎮 GPU 記憶體 (BS=16, 訓練模式):")
            print(f"   Baseline:  {baseline_memory:.2f} MB")
            print(f"   MultiHead: {multihead_memory:.2f} MB")
            print(f"   增加:      {(multihead_memory / baseline_memory - 1) * 100:.1f}%")
            
            # 恢復評估模式
            self.baseline_model.eval()
            self.multihead_model.eval()
        
        self.results['memory'] = results
        return results
    
    def compare_output_dimensions(self):
        """比較輸出維度"""
        print("\n" + "="*70)
        print("4. 輸出維度對比")
        print("="*70)
        
        batch_size = 2
        x = torch.randn(batch_size, 3, self.img_size, self.img_size).to(self.device)
        
        with torch.no_grad():
            # Baseline 輸出
            self.baseline_model.eval()
            baseline_out, _ = self.baseline_model(x)
            
            # MultiHead 輸出
            self.multihead_model.eval()
            multihead_out, _ = self.multihead_model(x)
        
        print(f"\n📏 推理模式輸出形狀:")
        print(f"   Baseline:  {baseline_out.shape}")
        print(f"   MultiHead: {multihead_out.shape}")
        
        # 計算錨框數量
        total_anchors = baseline_out.shape[1]
        print(f"\n📦 總錨框數: {total_anchors:,}")
        
        # 不同尺度的錨框分佈
        strides = [8, 16, 32]
        print(f"\n🔢 錨框分佈 ({self.img_size}×{self.img_size}):")
        for stride in strides:
            grid_size = self.img_size // stride
            anchors = grid_size * grid_size * 3
            print(f"   P{int(np.log2(stride))}/{stride}: {grid_size}×{grid_size}×3 = {anchors:,} 錨框")
        
        return {
            'baseline_shape': list(baseline_out.shape),
            'multihead_shape': list(multihead_out.shape),
            'total_anchors': total_anchors
        }
    
    def generate_comparison_table(self):
        """生成對比表格"""
        print("\n" + "="*70)
        print("5. 綜合對比表格")
        print("="*70)
        
        # 準備表格數據
        data = []
        
        # 架構對比
        if 'architecture' in self.results:
            arch = self.results['architecture']
            data.append(['總參數量', 
                        f"{arch['parameters']['baseline']:,}", 
                        f"{arch['parameters']['multihead']:,}",
                        f"+{arch['parameters']['increase_pct']:.1f}%"])
            data.append(['檢測層參數', 
                        f"{arch['detection_layer']['baseline']:,}", 
                        f"{arch['detection_layer']['multihead']:,}",
                        f"+{arch['detection_layer']['increase_pct']:.1f}%"])
        
        # 推理速度對比 (BS=1)
        if 'inference_speed' in self.results and 'batch_1' in self.results['inference_speed']:
            speed = self.results['inference_speed']['batch_1']
            data.append(['推理時間 (ms)', 
                        f"{speed['baseline_ms']:.2f}", 
                        f"{speed['multihead_ms']:.2f}",
                        f"+{speed['slowdown_pct']:.1f}%"])
            data.append(['FPS (BS=1)', 
                        f"{speed['baseline_fps']:.1f}", 
                        f"{speed['multihead_fps']:.1f}",
                        f"-{(1 - speed['multihead_fps']/speed['baseline_fps'])*100:.1f}%"])
        
        # 記憶體對比
        if 'memory' in self.results:
            mem = self.results['memory']
            data.append(['模型大小 (MB)', 
                        f"{mem['model_size_mb']['baseline']:.2f}", 
                        f"{mem['model_size_mb']['multihead']:.2f}",
                        f"+{mem['model_size_mb']['increase_pct']:.1f}%"])
            if 'gpu_memory_mb' in mem:
                data.append(['GPU記憶體 (MB)', 
                            f"{mem['gpu_memory_mb']['baseline']:.2f}", 
                            f"{mem['gpu_memory_mb']['multihead']:.2f}",
                            f"+{mem['gpu_memory_mb']['increase_pct']:.1f}%"])
        
        # 打印表格
        headers = ['指標', 'Baseline', 'MultiHead (1B4H)', '差異']
        print("\n" + tabulate(data, headers=headers, tablefmt='grid'))
        
        return data
    
    def save_results(self, output_path='comparison_results.json'):
        """保存對比結果"""
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"\n💾 結果已保存到: {output_path}")
    
    def plot_comparison(self):
        """繪製對比圖表"""
        if not self.results:
            print("無結果可繪製")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. 參數量對比
        if 'architecture' in self.results:
            ax = axes[0, 0]
            params = self.results['architecture']['parameters']
            models = ['Baseline', 'MultiHead']
            values = [params['baseline'], params['multihead']]
            bars = ax.bar(models, values, color=['blue', 'orange'])
            ax.set_ylabel('Parameters')
            ax.set_title('Model Parameters Comparison')
            ax.bar_label(bars, fmt='%d')
        
        # 2. 推理速度對比
        if 'inference_speed' in self.results:
            ax = axes[0, 1]
            batch_sizes = []
            baseline_fps = []
            multihead_fps = []
            
            for key in sorted(self.results['inference_speed'].keys()):
                bs = int(key.split('_')[1])
                batch_sizes.append(bs)
                baseline_fps.append(self.results['inference_speed'][key]['baseline_fps'])
                multihead_fps.append(self.results['inference_speed'][key]['multihead_fps'])
            
            ax.plot(batch_sizes, baseline_fps, 'o-', label='Baseline', color='blue')
            ax.plot(batch_sizes, multihead_fps, 's-', label='MultiHead', color='orange')
            ax.set_xlabel('Batch Size')
            ax.set_ylabel('FPS')
            ax.set_title('Inference Speed Comparison')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 3. 記憶體使用對比
        if 'memory' in self.results:
            ax = axes[1, 0]
            mem = self.results['memory']['model_size_mb']
            models = ['Baseline', 'MultiHead']
            values = [mem['baseline'], mem['multihead']]
            bars = ax.bar(models, values, color=['blue', 'orange'])
            ax.set_ylabel('Size (MB)')
            ax.set_title('Model Size Comparison')
            ax.bar_label(bars, fmt='%.2f')
        
        # 4. 相對性能圖
        ax = axes[1, 1]
        metrics = []
        percentages = []
        
        if 'architecture' in self.results:
            metrics.append('Parameters')
            percentages.append(self.results['architecture']['parameters']['increase_pct'])
        
        if 'inference_speed' in self.results and 'batch_1' in self.results['inference_speed']:
            metrics.append('Inference Time')
            percentages.append(self.results['inference_speed']['batch_1']['slowdown_pct'])
        
        if 'memory' in self.results:
            metrics.append('Model Size')
            percentages.append(self.results['memory']['model_size_mb']['increase_pct'])
        
        if metrics:
            colors = ['red' if p > 0 else 'green' for p in percentages]
            bars = ax.barh(metrics, percentages, color=colors)
            ax.set_xlabel('Change (%)')
            ax.set_title('Relative Performance (MultiHead vs Baseline)')
            ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
            
            # 添加數值標籤
            for bar, val in zip(bars, percentages):
                ax.text(val, bar.get_y() + bar.get_height()/2, 
                       f'{val:+.1f}%', 
                       ha='left' if val > 0 else 'right',
                       va='center')
        
        plt.tight_layout()
        plt.savefig('comparison_plots.png', dpi=150)
        print("\n📊 對比圖表已保存到: comparison_plots.png")
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Compare Baseline vs MultiHead YOLOv7-Tiny')
    parser.add_argument('--baseline-cfg', default='cfg/training/yolov7-tiny.yaml',
                       help='Baseline model config')
    parser.add_argument('--multihead-cfg', default='cfg/training/yolov7-tiny-multihead-proper.yaml',
                       help='MultiHead model config')
    parser.add_argument('--img-size', type=int, default=320,
                       help='Input image size')
    parser.add_argument('--batch-sizes', nargs='+', type=int, default=[1, 8, 16, 32],
                       help='Batch sizes for speed test')
    parser.add_argument('--num-runs', type=int, default=100,
                       help='Number of runs for speed test')
    parser.add_argument('--output', default='comparison_results.json',
                       help='Output JSON file')
    parser.add_argument('--plot', action='store_true',
                       help='Generate comparison plots')
    
    args = parser.parse_args()
    
    # 執行對比
    comparator = ModelComparator(
        baseline_cfg=args.baseline_cfg,
        multihead_cfg=args.multihead_cfg,
        img_size=args.img_size
    )
    
    # 執行所有對比
    comparator.compare_architecture()
    comparator.compare_inference_speed(
        num_runs=args.num_runs,
        batch_sizes=args.batch_sizes
    )
    comparator.compare_memory_usage()
    comparator.compare_output_dimensions()
    
    # 生成表格
    comparator.generate_comparison_table()
    
    # 保存結果
    comparator.save_results(args.output)
    
    # 繪製圖表
    if args.plot:
        comparator.plot_comparison()
    
    print("\n" + "="*70)
    print("✅ 對比完成！")
    print("="*70)

if __name__ == "__main__":
    main()