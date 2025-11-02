# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : plot_training_data.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-11-02 00:00
@Update Date    :
@Description    : 训练数据可视化脚本
从保存的JSON/CSV文件中加载训练指标并绘制图表
"""
# ------------------------------------------------------------

import argparse
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from common.plot import Plotter
from common.utils.data_manager import TrainingDataManager


def main():
    parser = argparse.ArgumentParser(description="绘制训练数据图表")
    parser.add_argument(
        "--data-file",
        type=str,
        required=True,
        help="训练数据文件路径（JSON或CSV）",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="plots",
        help="图表输出目录（默认：plots）",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=None,
        help="要绘制的指标列表（默认：绘制所有指标）",
    )
    parser.add_argument(
        "--smooth",
        action="store_true",
        help="是否使用平滑曲线",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=10,
        help="平滑窗口大小（默认：10）",
    )

    args = parser.parse_args()

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载数据
    data_file = Path(args.data_file)
    if not data_file.exists():
        print(f"❌ 文件不存在: {data_file}")
        return

    print(f"📊 加载训练数据: {data_file}")

    data_manager = TrainingDataManager()

    if data_file.suffix == ".json":
        data_manager.load_metrics(str(data_file))
    else:
        print(f"⚠️  暂不支持 {data_file.suffix} 格式，请使用JSON文件")
        return

    # 获取DataFrame
    df = data_manager.get_metrics_dataframe()
    if df is None or len(df) == 0:
        print("❌ 没有可用的训练数据")
        return

    print(f"✅ 加载了 {len(df)} 条记录")
    print(f"📈 可用指标: {', '.join([c for c in df.columns if c != 'step'])}")

    # 准备绘图数据
    metrics_to_plot = args.metrics if args.metrics else [c for c in df.columns if c != "step"]

    plot_data = {}
    for metric in metrics_to_plot:
        if metric not in df.columns:
            print(f"⚠️  跳过不存在的指标: {metric}")
            continue
        plot_data[metric] = df[metric].tolist()

    if len(plot_data) == 0:
        print("❌ 没有可绘制的指标")
        return

    # 创建绘图器
    plotter = Plotter()

    # 绘制图表
    output_path = output_dir / f"training_curves_{data_file.stem}.png"

    print(f"🎨 绘制图表...")
    plotter.plot_training_metrics(
        metrics=plot_data,
        save_path=str(output_path),
        show=False,
        smooth=args.smooth,
        window_size=args.window_size,
    )

    print(f"✅ 图表已保存: {output_path}")


if __name__ == "__main__":
    main()
