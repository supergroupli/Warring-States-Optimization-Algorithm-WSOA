#!/usr/bin/env python3
"""
战国优化算法 (Warring States Optimization Algorithm) — 启动脚本
支持 CEC2005, CEC2014, CEC2017, CEC2022 测试套件
"""

import argparse
import os
from run_experiment import run_experiment


def main():
    parser = argparse.ArgumentParser(
        description="战国优化算法 (WSOA) 对比实验",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python run.py                                     默认 (CEC2022, dim=10)
  python run.py --suite CEC2017 --dim 30            CEC2017, 30维
  python run.py --suite CEC2005 --dim 10 --runs 10  CEC2005, 10次运行
  python run.py --quick                             快速测试
  python run.py --all_suites                        运行全部CEC套件
        """,
    )
    parser.add_argument("--suite", type=str, default="CEC2022",
                        choices=["CEC2005", "CEC2014", "CEC2017", "CEC2022"],
                        help="CEC测试套件 (默认: CEC2022)")
    parser.add_argument("--dim", type=int, default=10,
                        help="搜索空间维度 (默认: 10)")
    parser.add_argument("--runs", type=int, default=5,
                        help="独立运行次数 (默认: 5)")
    parser.add_argument("--pop_size", type=int, default=140,
                        help="总种群大小 (默认: 140)")
    parser.add_argument("--max_iter", type=int, default=200,
                        help="最大迭代次数 (默认: 200)")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="结果输出目录 (默认: results)")
    parser.add_argument("--quick", action="store_true",
                        help="快速测试模式 (dim=10, runs=3, iter=100)")
    parser.add_argument("--all_suites", action="store_true",
                        help="运行全部CEC套件 (CEC2005/2014/2017/2022)")

    args = parser.parse_args()

    if args.quick:
        args.dim = 10
        args.runs = 3
        args.max_iter = 100
        args.pop_size = 70

    if args.all_suites:
        for suite in ["CEC2005", "CEC2014", "CEC2017", "CEC2022"]:
            out_dir = os.path.join(args.output_dir, suite)
            os.makedirs(out_dir, exist_ok=True)
            print(f"\n{'#' * 80}")
            print(f"  运行测试套件: {suite}")
            print(f"{'#' * 80}\n")
            run_experiment(
                dim=args.dim,
                runs=args.runs,
                pop_size=args.pop_size,
                max_iter=args.max_iter,
                output_dir=out_dir,
                suite=suite,
            )
    else:
        out_dir = os.path.join(args.output_dir, args.suite)
        os.makedirs(out_dir, exist_ok=True)
        run_experiment(
            dim=args.dim,
            runs=args.runs,
            pop_size=args.pop_size,
            max_iter=args.max_iter,
            output_dir=out_dir,
            suite=args.suite,
        )


if __name__ == "__main__":
    main()
