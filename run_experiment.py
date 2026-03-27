"""
战国七雄算法 — 实验运行模块
基于 CEC2022 测试套件，对比 WSOA 与 PSO, DE, GA, ABC, GWO, WOA 的性能
输出: CSV结果表 + 收敛曲线图 + 汇总图表 + 日志
"""

import numpy as np
import csv
import time
import sys
import os
import logging
from datetime import datetime

from warring_states_algorithm import WarringStatesOptimizationAlgorithm
from benchmark_functions import get_all_benchmarks
from comparison_algorithms import ALGORITHMS
from plotting import plot_convergence, plot_summary_bar, plot_radar


def setup_logger(output_dir="results"):
    """配置日志系统，同时输出到终端和文件"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"experiment_{timestamp}.log")

    logger = logging.getLogger("WSOA_Experiment")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S")

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    logger.info(f"日志文件: {log_file}")
    return logger, log_file


def save_csv_results(all_stats, all_times, algo_names, benchmarks, output_dir, dim, runs):
    """
    保存实验结果到CSV文件，生成论文所需的标准格式表格

    输出文件:
        1. results_raw.csv         — 每次独立运行的原始数据
        2. results_summary.csv     — 各函数的Best/Mean/Worst/Std/Median汇总
        3. results_ranking.csv     — 各函数上各算法的排名
        4. results_convergence.csv — 收敛曲线数据
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── 1. 原始数据表 (每次运行的适应度值) ──
    raw_path = os.path.join(output_dir, "results_raw.csv")
    with open(raw_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Function", "f_bias", "Dimension", "Algorithm",
                         "Run", "Fitness", "Time(s)"])
        for func in benchmarks:
            for algo in algo_names:
                for run_i, (fit, t) in enumerate(
                        zip(all_stats[func.name][algo], all_times[func.name][algo])):
                    writer.writerow([func.name, func.optimal_value, dim,
                                     algo, run_i + 1, f"{fit:.15e}", f"{t:.4f}"])
    print(f"  原始数据已保存: {raw_path}")

    # ── 2. 汇总统计表 (论文Table标准格式) ──
    summary_path = os.path.join(output_dir, "results_summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Function", "f_bias", "Dimension", "Algorithm",
                         "Best", "Worst", "Mean", "Median", "Std", "Avg_Time(s)"])
        for func in benchmarks:
            for algo in algo_names:
                vals = np.array(all_stats[func.name][algo])
                ts = np.array(all_times[func.name][algo])
                writer.writerow([
                    func.name, func.optimal_value, dim, algo,
                    f"{np.min(vals):.15e}",
                    f"{np.max(vals):.15e}",
                    f"{np.mean(vals):.15e}",
                    f"{np.median(vals):.15e}",
                    f"{np.std(vals):.15e}",
                    f"{np.mean(ts):.4f}",
                ])
    print(f"  汇总统计已保存: {summary_path}")

    # ── 3. 排名表 ──
    ranking_path = os.path.join(output_dir, "results_ranking.csv")
    with open(ranking_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Function"] + algo_names)

        rank_totals = {algo: 0 for algo in algo_names}
        for func in benchmarks:
            means = {algo: np.mean(all_stats[func.name][algo]) for algo in algo_names}
            sorted_algos = sorted(algo_names, key=lambda a: means[a])
            ranks = {algo: sorted_algos.index(algo) + 1 for algo in algo_names}
            writer.writerow([func.name] + [ranks[a] for a in algo_names])
            for algo in algo_names:
                rank_totals[algo] += ranks[algo]

        # 平均排名行
        n_funcs = len(benchmarks)
        writer.writerow(["Avg_Rank"] +
                         [f"{rank_totals[a]/n_funcs:.2f}" for a in algo_names])
        # 总排名行
        writer.writerow(["Total_Rank"] + [rank_totals[a] for a in algo_names])
    print(f"  排名表已保存: {ranking_path}")

    # ── 4. 收敛曲线数据 (方便用其他工具重绘) ──
    return raw_path, summary_path, ranking_path


def save_convergence_csv(all_convs, algo_names, benchmarks, output_dir):
    """保存收敛曲线数据到CSV"""
    conv_path = os.path.join(output_dir, "results_convergence.csv")
    with open(conv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for func in benchmarks:
            # 每个函数的收敛曲线
            max_len = max(len(all_convs[func.name][a]) for a in algo_names
                          if all_convs[func.name][a] is not None)
            header = ["Function", "Iteration"] + algo_names
            writer.writerow(header)
            for i in range(max_len):
                row = [func.name, i]
                for algo in algo_names:
                    conv = all_convs[func.name][algo]
                    if conv is not None and i < len(conv):
                        row.append(f"{conv[i]:.15e}")
                    else:
                        row.append("")
                writer.writerow(row)
            writer.writerow([])  # 空行分隔
    print(f"  收敛曲线数据已保存: {conv_path}")


def run_experiment(dim=10, runs=5, pop_size=140, max_iter=1000,
                   output_dir="results", suite="CEC2022"):
    """
    运行完整对比实验

    参数:
        dim:        搜索空间维度
        runs:       独立运行次数
        pop_size:   总种群大小
        max_iter:   最大迭代次数
        output_dir: 结果输出目录
        suite:      CEC测试套件 ("CEC2005","CEC2014","CEC2017","CEC2022")
    """
    logger, log_file = setup_logger(output_dir)
    benchmarks = get_all_benchmarks(dim, suite=suite)
    algo_names = ["WSOA"] + list(ALGORITHMS.keys())
    n_funcs = len(benchmarks)
    n_algos = len(algo_names)
    total_tasks = n_funcs * runs * n_algos
    completed = 0
    exp_start = time.time()

    logger.info("╔" + "═" * 72 + "╗")
    logger.info(f"║  战国优化算法 (WSOA) — {suite} 对比实验")
    logger.info(f"║  维度={dim}, 种群={pop_size}, 迭代={max_iter}, 独立运行={runs}次")
    logger.info(f"║  对比算法: {', '.join(ALGORITHMS.keys())}")
    logger.info(f"║  测试函数: {suite} ({n_funcs}个函数)")
    logger.info(f"║  总任务数: {total_tasks} (={n_funcs}函数 x {runs}次 x {n_algos}算法)")
    logger.info("╚" + "═" * 72 + "╝")

    # 存储所有数据（用于CSV导出）
    all_stats = {}   # {func_name: {algo: [fit1, fit2, ...]}}
    all_times = {}   # {func_name: {algo: [t1, t2, ...]}}
    all_convs = {}   # {func_name: {algo: [conv_curve]}}
    summary_mean = {}

    for fi, func in enumerate(benchmarks):
        func_start = time.time()
        logger.info("")
        logger.info("─" * 72)
        logger.info(f"  [{fi+1}/{n_funcs}] {func.name}: {func.full_name}")
        logger.info(f"  维度={func.dim}, f*={func.optimal_value}")
        logger.info("─" * 72)

        stats = {name: [] for name in algo_names}
        times = {name: [] for name in algo_names}
        convs = {name: None for name in algo_names}

        for run in range(runs):
            run_start = time.time()

            # ----- WSOA -----
            np.random.seed(run * 42 + 7)
            t0 = time.time()
            wsa = WarringStatesOptimizationAlgorithm(
                pop_size=pop_size, max_iter=max_iter, verbose=False)
            _, wsa_fit, wsa_conv = wsa.optimize(func)
            wsa_time = time.time() - t0
            stats["WSOA"].append(wsa_fit)
            times["WSOA"].append(wsa_time)
            if run == 0:
                convs["WSOA"] = wsa_conv
            completed += 1
            logger.info(f"    Run {run+1}/{runs} | WSOA={wsa_fit:.6e} ({wsa_time:.1f}s) "
                        f"| 进度: {completed}/{total_tasks} ({100*completed/total_tasks:.0f}%)")

            # ----- 对比算法 -----
            for algo_name, algo_func in ALGORITHMS.items():
                np.random.seed(run * 42 + 7)
                t0 = time.time()
                _, fit, conv = algo_func(func, pop_size, max_iter)
                algo_time = time.time() - t0
                stats[algo_name].append(fit)
                times[algo_name].append(algo_time)
                if run == 0:
                    convs[algo_name] = conv
                completed += 1

            run_elapsed = time.time() - run_start
            total_elapsed = time.time() - exp_start
            est_remaining = (total_elapsed / completed) * (total_tasks - completed)
            logger.info(f"    Run {run+1}/{runs} 完成 | 本轮耗时: {run_elapsed:.1f}s | "
                        f"预计剩余: {est_remaining/60:.1f}min")

        # 保存到全局数据
        all_stats[func.name] = stats
        all_times[func.name] = times
        all_convs[func.name] = convs

        # 打印结果表
        logger.info("")
        logger.info(f"  {'算法':<8} {'最优值':<16} {'平均值':<16} "
                    f"{'标准差':<16} {'平均耗时':<10}")
        logger.info(f"  {'─' * 68}")

        best_algo = min(algo_names, key=lambda k: np.mean(stats[k]))

        func_summary = {}
        for algo in algo_names:
            vals = np.array(stats[algo])
            best_v = np.min(vals)
            mean_v = np.mean(vals)
            std_v = np.std(vals)
            mean_t = np.mean(times[algo])
            marker = " ★" if algo == best_algo else ""
            logger.info(f"  {algo:<8} {best_v:<16.6e} {mean_v:<16.6e} "
                        f"{std_v:<16.6e} {mean_t:<8.2f}s{marker}")
            func_summary[algo] = mean_v

        summary_mean[func.name] = func_summary

        func_elapsed = time.time() - func_start
        logger.info(f"  {func.name} 总耗时: {func_elapsed:.1f}s")

        # 绘制收敛曲线
        conv_data = {algo: {"convergence": convs[algo]} for algo in algo_names}
        plot_convergence(conv_data, func.name, save_dir=output_dir)

    # ===== 保存CSV =====
    logger.info("")
    logger.info("═" * 72)
    logger.info("  保存CSV结果文件...")
    save_csv_results(all_stats, all_times, algo_names, benchmarks,
                     output_dir, dim, runs)
    save_convergence_csv(all_convs, algo_names, benchmarks, output_dir)

    # ===== 汇总图表 =====
    logger.info("")
    logger.info("  绘制汇总图表...")
    plot_summary_bar(summary_mean, save_dir=output_dir)
    plot_radar(summary_mean, save_dir=output_dir)

    # ===== 排名汇总 =====
    logger.info("")
    logger.info("═" * 72)
    logger.info("  各函数平均排名汇总:")
    logger.info(f"  {'─' * 50}")
    rank_totals = {algo: 0 for algo in algo_names}
    for fn in summary_mean:
        sorted_algos = sorted(algo_names, key=lambda a: summary_mean[fn][a])
        for rank, algo in enumerate(sorted_algos, 1):
            rank_totals[algo] += rank

    avg_ranks = sorted(rank_totals.items(), key=lambda x: x[1])
    for algo, total_rank in avg_ranks:
        avg_r = total_rank / len(summary_mean)
        marker = " ★" if algo == avg_ranks[0][0] else ""
        logger.info(f"  {algo:<8} 平均排名: {avg_r:.2f}{marker}")

    total_time = time.time() - exp_start
    logger.info("")
    logger.info("═" * 72)
    logger.info(f"  实验完成! ★ = 最优算法")
    logger.info(f"  总耗时: {total_time/60:.1f} 分钟")
    logger.info(f"  结果保存在: {output_dir}/")
    logger.info(f"  日志文件: {log_file}")
    logger.info("  CSV文件:")
    logger.info(f"    - results_raw.csv         (每次运行原始数据)")
    logger.info(f"    - results_summary.csv     (Best/Worst/Mean/Median/Std)")
    logger.info(f"    - results_ranking.csv     (各函数排名 + 平均排名)")
    logger.info(f"    - results_convergence.csv (收敛曲线数据)")
    logger.info("═" * 72)

    return summary_mean
