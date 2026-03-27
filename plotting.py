"""
可视化模块
绘制收敛曲线对比图和汇总结果表格图
"""

import numpy as np
import os

# 各算法配色方案 (12种算法)
ALGO_COLORS = {
    "WSOA": "#e74c3c",  # 红色 — 战国优化
    "PSO": "#3498db",   # 蓝色
    "DE":  "#2ecc71",   # 绿色
    "GA":  "#9b59b6",   # 紫色
    "ABC": "#f39c12",   # 橙色
    "GWO": "#1abc9c",   # 青色
    "WOA": "#e67e22",   # 深橙色
    "SCA": "#34495e",   # 深灰蓝
    "HHO": "#d35400",   # 南瓜色
    "MPA": "#8e44ad",   # 紫罗兰
    "AO":  "#16a085",   # 深绿
    "DBO": "#c0392b",   # 暗红
}

ALGO_MARKERS = {
    "WSOA": "o", "PSO": "s", "DE": "^", "GA": "D",
    "ABC": "v", "GWO": "p", "WOA": "*", "SCA": "h",
    "HHO": "<", "MPA": ">", "AO": "P", "DBO": "X",
}


def _setup_matplotlib():
    """配置matplotlib后端和字体"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["DejaVu Sans", "SimHei", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def plot_convergence(results, func_name, save_dir="results"):
    """
    绘制单个基准函数的收敛曲线对比图

    参数:
        results:   dict, {算法名: {"convergence": [...]}}
        func_name: 函数名称
        save_dir:  保存目录
    """
    try:
        plt = _setup_matplotlib()
    except ImportError:
        print("  [跳过绘图] 未安装matplotlib")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    for algo_name, data in results.items():
        conv = data["convergence"]
        color = ALGO_COLORS.get(algo_name, None)
        marker = ALGO_MARKERS.get(algo_name, None)
        # 每隔一定间距标记点，避免过于密集
        mark_every = max(1, len(conv) // 10)
        lw = 2.5 if algo_name == "WSOA" else 1.5
        ax.plot(conv, label=algo_name, color=color, linewidth=lw,
                marker=marker, markevery=mark_every, markersize=5)

    ax.set_xlabel("Iteration", fontsize=13)
    ax.set_ylabel("Best Fitness (log scale)", fontsize=13)
    ax.set_title(f"Convergence Comparison — {func_name}", fontsize=15)
    ax.set_yscale("log")
    ax.legend(fontsize=11, loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"convergence_{func_name}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  收敛曲线已保存: {path}")


def plot_summary_bar(summary, save_dir="results"):
    """
    绘制所有函数的算法平均适应度柱状对比图

    参数:
        summary: dict, {func_name: {algo_name: mean_fitness, ...}, ...}
        save_dir: 保存目录
    """
    try:
        plt = _setup_matplotlib()
    except ImportError:
        print("  [跳过绘图] 未安装matplotlib")
        return

    func_names = list(summary.keys())
    algo_names = list(next(iter(summary.values())).keys())
    n_func = len(func_names)
    n_algo = len(algo_names)

    x = np.arange(n_func)
    width = 0.8 / n_algo

    fig, ax = plt.subplots(figsize=(14, 6))

    for i, algo in enumerate(algo_names):
        vals = [summary[fn][algo] for fn in func_names]
        # 对数尺度下用log值绘图
        log_vals = [np.log10(v + 1e-30) for v in vals]
        color = ALGO_COLORS.get(algo, None)
        ax.bar(x + i * width - 0.4 + width / 2, log_vals, width,
               label=algo, color=color, edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Benchmark Function", fontsize=13)
    ax.set_ylabel("log10(Mean Fitness)", fontsize=13)
    ax.set_title("Algorithm Comparison Summary", fontsize=15)
    ax.set_xticks(x)
    ax.set_xticklabels(func_names, fontsize=11)
    ax.legend(fontsize=10, loc="upper left", ncol=n_algo)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "summary_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  汇总对比图已保存: {path}")


def plot_radar(summary, save_dir="results"):
    """
    绘制算法在各函数上的排名雷达图

    参数:
        summary: dict, {func_name: {algo_name: mean_fitness, ...}, ...}
        save_dir: 保存目录
    """
    try:
        plt = _setup_matplotlib()
    except ImportError:
        print("  [跳过绘图] 未安装matplotlib")
        return

    func_names = list(summary.keys())
    algo_names = list(next(iter(summary.values())).keys())
    n_func = len(func_names)

    # 计算每个函数上各算法的排名 (1=最好)
    rankings = {algo: [] for algo in algo_names}
    for fn in func_names:
        sorted_algos = sorted(algo_names, key=lambda a: summary[fn][a])
        for rank, algo in enumerate(sorted_algos, 1):
            rankings[algo].append(rank)

    # 雷达图
    angles = np.linspace(0, 2 * np.pi, n_func, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for algo in algo_names:
        vals = rankings[algo] + rankings[algo][:1]
        color = ALGO_COLORS.get(algo, None)
        lw = 2.5 if algo == "WSOA" else 1.5
        ax.plot(angles, vals, label=algo, color=color, linewidth=lw)
        ax.fill(angles, vals, alpha=0.05, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(func_names, fontsize=10)
    ax.set_yticks(range(1, len(algo_names) + 1))
    ax.set_yticklabels([str(i) for i in range(1, len(algo_names) + 1)], fontsize=8)
    ax.set_ylim(0, len(algo_names) + 0.5)
    ax.set_title("Algorithm Ranking Radar (lower = better)", fontsize=14, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10)
    fig.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "ranking_radar.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  排名雷达图已保存: {path}")
