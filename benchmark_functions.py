"""
基准测试函数集 — 支持多个CEC测试套件
基于 opfunu 库，支持 CEC2005, CEC2014, CEC2017, CEC2022

依赖: pip install opfunu
"""

import numpy as np
import opfunu


class BenchmarkFunction:
    """CEC基准测试函数包装类，统一接口"""

    def __init__(self, suite, func_id, dim=10):
        """
        参数:
            suite:   测试套件名称 ("CEC2005","CEC2014","CEC2017","CEC2022")
            func_id: 函数编号
            dim:     维度
        """
        class_name = f"F{func_id}{suite[-4:]}"
        func_classes = opfunu.get_functions_by_classname(class_name)
        self._func = func_classes[0](ndim=dim)

        self.suite = suite
        self.func_id = func_id
        self.name = f"F{func_id}"
        self.full_name = self._func.name
        self.dim = dim
        self.lb = np.full(dim, self._func.bounds[0][0])
        self.ub = np.full(dim, self._func.bounds[0][1])
        self.optimal_value = self._func.f_bias
        self.f_bias = self._func.f_bias

    def evaluate(self, x):
        return self._func.evaluate(x)

    def __repr__(self):
        return f"{self.suite}-{self.name}({self.full_name}, dim={self.dim})"


# ============================================================
#  各测试套件配置
# ============================================================
SUITE_CONFIG = {
    "CEC2005": {
        "func_ids": list(range(1, 26)),   # F1-F25
        "dims": [10, 30],
        "description": "IEEE CEC2005 (25 functions)",
    },
    "CEC2014": {
        "func_ids": list(range(1, 31)),   # F1-F30
        "dims": [10, 30, 50],
        "description": "IEEE CEC2014 (30 functions)",
    },
    "CEC2017": {
        "func_ids": list(range(1, 30)),   # F1-F29
        "dims": [10, 30, 50],
        "description": "IEEE CEC2017 (29 functions)",
    },
    "CEC2022": {
        "func_ids": list(range(1, 13)),   # F1-F12
        "dims": [10, 20],
        "description": "IEEE CEC2022 (12 functions)",
    },
}


def get_all_benchmarks(dim=10, suite="CEC2022"):
    """
    返回指定CEC套件的全部基准测试函数

    参数:
        dim:   维度
        suite: 测试套件 ("CEC2005","CEC2014","CEC2017","CEC2022")
    """
    if suite not in SUITE_CONFIG:
        raise ValueError(f"未知测试套件: {suite}, 可选: {list(SUITE_CONFIG.keys())}")

    config = SUITE_CONFIG[suite]
    functions = []
    for fid in config["func_ids"]:
        try:
            f = BenchmarkFunction(suite, fid, dim)
            # 验证可用
            x = np.random.uniform(f.lb, f.ub)
            f.evaluate(x)
            functions.append(f)
        except Exception:
            pass  # 跳过不支持当前维度的函数
    return functions


def get_available_suites():
    """返回所有可用的测试套件信息"""
    return SUITE_CONFIG
