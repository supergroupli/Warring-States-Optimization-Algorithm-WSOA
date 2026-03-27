"""
战国优化算法 (Warring States Optimization Algorithm, WSOA)
一种基于中国战国历史的全新启发式优化算法

核心思想：
    7个子种群（代表战国七雄）在搜索空间中既竞争又合作，
    通过变法图强、合纵连横、征伐吞并等历史机制驱动寻优，
    最终"大一统"收敛到全局最优解。

算法阶段：
    1. 七国初立 — 初始化7个子种群，各有不同搜索策略
    2. 变法争雄 — 各国独立演化，周期性评估国力排名
    3. 合纵连横 — 根据国力格局动态调整合作/竞争关系
    4. 征伐吞并 — 弱国逐步被灭，资源重新分配
    5. 大一统   — 合并所有资源，集中精细搜索，输出最优解

v2.0 改进:
    - 增强七种变法策略的搜索效能
    - 新增策略自适应机制（弱国学习强国变法）
    - 合纵连横影响更多个体，增加间谍机制
    - 动态种群资源再分配
    - 大一统多算子混合精细搜索
    - 反向学习增强初始化多样性
"""

import numpy as np
from math import gamma as math_gamma


# ============================================================
#  个体（士兵/人才）
# ============================================================
class Individual:
    """搜索空间中的一个候选解"""

    __slots__ = ("position", "fitness")

    def __init__(self, position, fitness=np.inf):
        self.position = position.copy()
        self.fitness = fitness


# ============================================================
#  诸侯国
# ============================================================
class State:
    """一个子种群，代表一个诸侯国"""

    def __init__(self, name, strategy_id, individuals, lb, ub):
        self.name = name
        self.strategy_id = strategy_id
        self.individuals = individuals
        self.lb = lb
        self.ub = ub
        self.alive = True
        self.stagnation = 0          # 停滞计数器
        self.prev_best_fit = np.inf  # 上一轮最优

        # 国力指标
        self.best_individual = None
        self.avg_fitness = np.inf
        self.national_power = 0.0
        # 策略成功率追踪
        self.success_count = 0
        self.total_count = 0

    @property
    def size(self):
        return len(self.individuals)

    @property
    def success_rate(self):
        return self.success_count / max(self.total_count, 1)

    def update_stats(self):
        """更新国力统计"""
        if not self.individuals:
            self.alive = False
            return
        fitnesses = np.array([ind.fitness for ind in self.individuals])
        best_idx = np.argmin(fitnesses)
        self.best_individual = self.individuals[best_idx]
        self.avg_fitness = np.mean(fitnesses)
        # 综合国力 = 60%最优适应度 + 40%平均适应度（越小越强）
        self.national_power = 0.6 * self.best_individual.fitness + 0.4 * self.avg_fitness

        # 停滞检测
        if self.best_individual.fitness < self.prev_best_fit * 0.9999:
            self.stagnation = 0
        else:
            self.stagnation += 1
        self.prev_best_fit = self.best_individual.fitness


# ============================================================
#  七种变法策略（搜索算子） — 增强版
# ============================================================
class ReformStrategies:
    """
    七国各自的变法策略，对应不同的搜索算子：
        0-秦: 商鞅变法 — Lévy飞行 + 精英引导
        1-齐: 稷下学宫 — 多教师协同学习
        2-楚: 地大物博 — 柯西变异大范围探索
        3-燕: 坚守北疆 — 高斯局部搜索 + 本国精英引导
        4-赵: 胡服骑射 — 自适应差分变异
        5-魏: 李悝变法 — 带自适应参数的差分进化
        6-韩: 术治之道 — 正弦余弦引导搜索
    """

    _levy_sigma_cache = {}

    @staticmethod
    def levy_flight(dim, beta=1.5):
        """Lévy飞行步长"""
        if beta not in ReformStrategies._levy_sigma_cache:
            sigma_u = (math_gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                       (math_gamma((1 + beta) / 2) * beta *
                        2 ** ((beta - 1) / 2))) ** (1 / beta)
            ReformStrategies._levy_sigma_cache[beta] = sigma_u
        sigma_u = ReformStrategies._levy_sigma_cache[beta]
        u = np.random.randn(dim) * sigma_u
        v = np.random.randn(dim)
        step = u / (np.abs(v) ** (1 / beta))
        return step

    @staticmethod
    def qin_strategy(ind, global_best, lb, ub, t_ratio):
        """秦·商鞅变法: Lévy飞行 + 精英引导混合"""
        dim = len(ind.position)
        step = ReformStrategies.levy_flight(dim)
        # 早期偏Lévy探索，后期偏精英引导
        alpha = 0.01 + (1 - t_ratio) * 0.5
        if np.random.rand() < 0.5:
            # Lévy飞行探索
            new_pos = ind.position + alpha * step * np.abs(global_best.position - ind.position)
        else:
            # 精英引导 + Lévy扰动
            r = np.random.rand(dim)
            new_pos = global_best.position + 0.1 * step * (ub - lb) * (1 - t_ratio)
            new_pos = ind.position + r * (new_pos - ind.position)
        return np.clip(new_pos, lb, ub)

    @staticmethod
    def qi_strategy(ind, state_individuals, global_best, lb, ub, t_ratio):
        """齐·稷下学宫: 多教师协同学习"""
        dim = len(ind.position)
        n = len(state_individuals)
        # 根据适应度选择教师（越优概率越高）
        fits = np.array([s.fitness for s in state_individuals])
        if np.max(fits) - np.min(fits) > 1e-30:
            weights = (np.max(fits) - fits) + 1e-10
            probs = weights / np.sum(weights)
        else:
            probs = np.ones(n) / n

        # 选两个教师
        t_ids = np.random.choice(n, size=min(2, n), replace=False, p=probs)
        teacher1 = state_individuals[t_ids[0]]
        teacher2 = state_individuals[t_ids[1]] if len(t_ids) > 1 else global_best

        r1, r2 = np.random.rand(dim), np.random.rand(dim)
        # 向优秀教师学习 + 差分探索
        new_pos = ind.position + \
                  r1 * (teacher1.position - ind.position) + \
                  r2 * 0.5 * (teacher2.position - ind.position)
        # 有概率向全局最优进一步学习
        if np.random.rand() < 0.3 + 0.3 * t_ratio:
            mask = np.random.rand(dim) < 0.3
            new_pos[mask] = global_best.position[mask] + 0.05 * np.random.randn(np.sum(mask)) * (ub[mask] - lb[mask])
        return np.clip(new_pos, lb, ub)

    @staticmethod
    def chu_strategy(ind, global_best, lb, ub, t_ratio):
        """楚·地大物博: 柯西变异大范围探索"""
        dim = len(ind.position)
        # 柯西分布提供重尾探索，比均匀随机更有效
        cauchy = np.random.standard_cauchy(dim) * 0.1
        if np.random.rand() < 0.5:
            # 围绕自身柯西探索
            scale = (1 - t_ratio) * 0.3 + 0.02
            new_pos = ind.position + cauchy * (ub - lb) * scale
        else:
            # 围绕全局最优柯西探索
            new_pos = global_best.position + cauchy * (ub - lb) * 0.1 * (1 - t_ratio)
        return np.clip(new_pos, lb, ub)

    @staticmethod
    def yan_strategy(ind, state_best, global_best, lb, ub, t_ratio):
        """燕·坚守北疆: 高斯局部搜索 + 双精英引导"""
        dim = len(ind.position)
        sigma = (ub - lb) * (0.05 * (1 - t_ratio) + 0.002)
        perturbation = np.random.randn(dim) * sigma
        # 同时参考本国最优和全局最优
        w1 = 0.3 + 0.4 * t_ratio  # 后期更多跟随全局最优
        direction = w1 * (global_best.position - ind.position) + \
                    (1 - w1) * (state_best.position - ind.position)
        r = np.random.rand()
        new_pos = ind.position + r * 0.5 * direction + perturbation
        return np.clip(new_pos, lb, ub)

    @staticmethod
    def zhao_strategy(ind, global_best, state_individuals, lb, ub, t_ratio):
        """赵·胡服骑射: 自适应差分变异"""
        dim = len(ind.position)
        n = len(state_individuals)
        # 自适应缩放因子
        F = 0.5 + 0.3 * np.random.rand() * (1 - t_ratio)

        if n >= 3:
            idxs = np.random.choice(n, 3, replace=False)
            a, b, c = [state_individuals[i] for i in idxs]
            # DE/current-to-best/1
            new_pos = ind.position + \
                      F * (global_best.position - ind.position) + \
                      F * (a.position - b.position)
        else:
            r = np.random.rand(dim)
            new_pos = ind.position + F * r * (global_best.position - ind.position) + \
                      0.1 * np.random.randn(dim) * (ub - lb) * (1 - t_ratio)
        return np.clip(new_pos, lb, ub)

    @staticmethod
    def wei_strategy(ind, state_individuals, global_best, lb, ub, t_ratio):
        """魏·李悝变法: 自适应参数差分进化"""
        dim = len(ind.position)
        n = len(state_individuals)

        if n >= 4:
            # 自适应F和CR
            F = 0.4 + 0.5 * np.random.rand()
            CR = 0.9 - 0.4 * t_ratio

            # DE/rand-to-best/1
            idxs = np.random.choice(n, 3, replace=False)
            r1, r2, r3 = [state_individuals[i] for i in idxs]
            mutant = r1.position + F * (global_best.position - r1.position) + \
                     F * (r2.position - r3.position)

            new_pos = ind.position.copy()
            j_rand = np.random.randint(dim)
            mask = np.random.rand(dim) < CR
            mask[j_rand] = True
            new_pos[mask] = mutant[mask]
        else:
            new_pos = ind.position + 0.1 * np.random.randn(dim) * (ub - lb)
        return np.clip(new_pos, lb, ub)

    @staticmethod
    def han_strategy(ind, global_best, state_best, lb, ub, t_ratio):
        """韩·术治之道: 正弦余弦引导搜索 (SCA-inspired)"""
        dim = len(ind.position)
        a = 2.0 * (1 - t_ratio)  # 线性递减
        r1 = a * np.random.rand(dim)
        r2 = 2 * np.pi * np.random.rand(dim)
        r3 = np.random.rand(dim)

        if np.random.rand() < 0.5:
            # 正弦更新
            new_pos = ind.position + r1 * np.sin(r2) * np.abs(r3 * global_best.position - ind.position)
        else:
            # 余弦更新
            new_pos = ind.position + r1 * np.cos(r2) * np.abs(r3 * state_best.position - ind.position)
        return np.clip(new_pos, lb, ub)


# ============================================================
#  战国七雄算法主类
# ============================================================
class WarringStatesOptimizationAlgorithm:
    """
    战国优化算法 (Warring States Optimization Algorithm, WSOA) v2.0

    参数:
        pop_size    : 总种群大小（默认140，每国20个个体）
        max_iter    : 最大迭代次数
        annex_cycle : 每隔多少代进行一次征伐吞并
        unify_ratio : 当迭代进度达到此比例时进入大一统阶段
    """

    STATE_NAMES = ["秦(Qin)", "齐(Qi)", "楚(Chu)", "燕(Yan)",
                   "赵(Zhao)", "魏(Wei)", "韩(Han)"]

    def __init__(self, pop_size=140, max_iter=1000,
                 annex_cycle=50, unify_ratio=0.80, verbose=True):
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.annex_cycle = annex_cycle
        self.unify_ratio = unify_ratio
        self.verbose = verbose

        self.states = []
        self.global_best = None
        self.convergence_curve = []
        self.func_evals = 0

    # ----------------------------------------------------------
    #  辅助：评估并更新全局最优
    # ----------------------------------------------------------
    def _evaluate(self, pos, func):
        """评估适应度并更新全局最优"""
        fit = func.evaluate(pos)
        self.func_evals += 1
        if fit < self.global_best.fitness:
            self.global_best = Individual(pos, fit)
        return fit

    # ----------------------------------------------------------
    #  阶段一: 七国初立（含反向学习增强）
    # ----------------------------------------------------------
    def _initialize(self, func):
        """初始化七个诸侯国，使用反向学习增强多样性"""
        dim = func.dim
        lb, ub = func.lb, func.ub
        per_state = self.pop_size // 7
        remainder = self.pop_size % 7

        self.states = []
        self.global_best = Individual(np.zeros(dim), np.inf)

        for i in range(7):
            n = per_state + (1 if i < remainder else 0)
            individuals = []
            for _ in range(n):
                pos = lb + np.random.rand(dim) * (ub - lb)
                fit = self._evaluate(pos, func)
                ind = Individual(pos, fit)

                # 反向学习: 生成反向解，保留更优的
                opp_pos = lb + ub - pos
                opp_fit = self._evaluate(opp_pos, func)
                if opp_fit < fit:
                    ind = Individual(opp_pos, opp_fit)

                individuals.append(ind)

            state = State(self.STATE_NAMES[i], i, individuals, lb, ub)
            state.update_stats()
            self.states.append(state)

        if self.verbose:
            print("=" * 60)
            print("  阶段一：七国初立 — 初始化完成（含反向学习）")
            print("=" * 60)
            for s in self.states:
                print(f"  {s.name}: 人口={s.size}, 最优={s.best_individual.fitness:.6e}")
            print(f"  全局最优: {self.global_best.fitness:.6e}")
            print()

    # ----------------------------------------------------------
    #  阶段二: 变法争雄 — 各国独立演化
    # ----------------------------------------------------------
    def _reform_evolve(self, state, func, t_ratio):
        """对单个国家执行变法策略，带策略成功率追踪"""
        lb, ub = func.lb, func.ub
        strategy = ReformStrategies

        for ind in state.individuals:
            sid = state.strategy_id

            if sid == 0:
                new_pos = strategy.qin_strategy(ind, self.global_best, lb, ub, t_ratio)
            elif sid == 1:
                new_pos = strategy.qi_strategy(ind, state.individuals, self.global_best,
                                               lb, ub, t_ratio)
            elif sid == 2:
                new_pos = strategy.chu_strategy(ind, self.global_best, lb, ub, t_ratio)
            elif sid == 3:
                new_pos = strategy.yan_strategy(ind, state.best_individual,
                                                self.global_best, lb, ub, t_ratio)
            elif sid == 4:
                new_pos = strategy.zhao_strategy(ind, self.global_best,
                                                 state.individuals, lb, ub, t_ratio)
            elif sid == 5:
                new_pos = strategy.wei_strategy(ind, state.individuals,
                                                self.global_best, lb, ub, t_ratio)
            elif sid == 6:
                new_pos = strategy.han_strategy(ind, self.global_best,
                                                state.best_individual, lb, ub, t_ratio)
            else:
                new_pos = ind.position.copy()

            new_fit = self._evaluate(new_pos, func)
            state.total_count += 1

            if new_fit < ind.fitness:
                ind.position = new_pos
                ind.fitness = new_fit
                state.success_count += 1

        state.update_stats()

    # ----------------------------------------------------------
    #  策略自适应: 弱国学习强国变法
    # ----------------------------------------------------------
    def _strategy_adaptation(self):
        """
        策略自适应：停滞过久的国家学习最强国的变法策略。
        模拟历史上各国互相学习变法（如赵学胡服骑射）。
        """
        alive_states = [s for s in self.states if s.alive]
        if len(alive_states) <= 1:
            return

        # 找出成功率最高的策略
        best_state = max(alive_states, key=lambda s: s.success_rate)

        for s in alive_states:
            if s is best_state:
                continue
            # 停滞超过20代，或成功率过低时，学习强国策略
            if s.stagnation > 20 or (s.success_rate < 0.05 and s.total_count > 50):
                old_sid = s.strategy_id
                s.strategy_id = best_state.strategy_id
                s.stagnation = 0
                s.success_count = 0
                s.total_count = 0
                if self.verbose:
                    print(f"    策略自适应: {s.name} 学习 {best_state.name} 的变法 "
                          f"(策略 {old_sid} → {s.strategy_id})")

    # ----------------------------------------------------------
    #  阶段三: 合纵连横（增强版）
    # ----------------------------------------------------------
    def _hezong_lianheng(self, func, t_ratio):
        """
        合纵连横机制（增强版）:
          - 合纵：多个弱国共享信息，影响多个个体
          - 连横：强国吸引弱国人才
          - 间谍：从其他国家窃取维度信息
        """
        alive_states = [s for s in self.states if s.alive]
        if len(alive_states) <= 1:
            return

        alive_states.sort(key=lambda s: s.national_power)
        strongest = alive_states[0]
        weaker = alive_states[1:]
        dim = func.dim
        lb, ub = func.lb, func.ub

        # === 合纵：弱国联盟共享信息（影响多个个体） ===
        if len(weaker) >= 2:
            alliance_bests = [s.best_individual for s in weaker]
            # 合纵联盟最优解（所有弱国中的最优）
            alliance_best = min(alliance_bests, key=lambda ind: ind.fitness)

            for s in weaker:
                # 影响本国前30%差的个体
                sorted_inds = sorted(s.individuals, key=lambda ind: ind.fitness, reverse=True)
                n_affect = max(1, int(0.3 * len(sorted_inds)))

                for ind in sorted_inds[:n_affect]:
                    r = np.random.rand(dim)
                    new_pos = ind.position + r * (alliance_best.position - ind.position)
                    # 加入扰动避免早熟
                    new_pos += 0.02 * np.random.randn(dim) * (ub - lb) * (1 - t_ratio)
                    new_pos = np.clip(new_pos, lb, ub)
                    new_fit = self._evaluate(new_pos, func)

                    if new_fit < ind.fitness:
                        ind.position = new_pos
                        ind.fitness = new_fit

                s.update_stats()

        # === 间谍机制：窃取其他国家的维度信息 ===
        for s in alive_states:
            if len(s.individuals) < 2:
                continue
            # 随机选一个个体作为间谍
            spy = s.individuals[np.random.randint(len(s.individuals))]
            # 从另一个国家的最优解中窃取部分维度
            other_states = [os for os in alive_states if os is not s]
            if other_states:
                target = other_states[np.random.randint(len(other_states))]
                new_pos = spy.position.copy()
                # 窃取约30%的维度
                spy_dims = np.random.rand(dim) < 0.3
                new_pos[spy_dims] = target.best_individual.position[spy_dims]
                new_pos = np.clip(new_pos, lb, ub)
                new_fit = self._evaluate(new_pos, func)

                if new_fit < spy.fitness:
                    spy.position = new_pos
                    spy.fitness = new_fit

            s.update_stats()

        # === 连横：强国从弱国吸引人才 ===
        for s in weaker:
            if len(s.individuals) > 3:
                best_idx = np.argmin([ind.fitness for ind in s.individuals])
                if np.random.rand() < 0.15 + 0.2 * t_ratio:
                    talent = s.individuals.pop(best_idx)
                    strongest.individuals.append(talent)
                    s.update_stats()

        strongest.update_stats()

    # ----------------------------------------------------------
    #  远交近攻机制（增强版）
    # ----------------------------------------------------------
    def _distant_ally_near_attack(self, func, t_ratio):
        """
        远交近攻（增强版）：
          近攻：与最近国家竞争学习（多个个体参与）
          远交：与最远国家交换信息
        """
        alive_states = [s for s in self.states if s.alive]
        if len(alive_states) <= 2:
            return

        dim = func.dim
        lb, ub = func.lb, func.ub
        centers = np.array([s.best_individual.position for s in alive_states])

        for i, state_i in enumerate(alive_states):
            distances = np.array([np.linalg.norm(centers[i] - centers[j])
                                  if i != j else np.inf
                                  for j in range(len(alive_states))])

            nearest_idx = np.argmin(distances)
            farthest_idx = np.argmax(distances[distances < np.inf]) if np.any(distances < np.inf) else nearest_idx
            nearest_state = alive_states[nearest_idx]

            # 近攻：多个个体向最近国家的最优学习
            n_attackers = max(1, int(0.2 * len(state_i.individuals)))
            attack_indices = np.random.choice(len(state_i.individuals),
                                              n_attackers, replace=False)
            for idx in attack_indices:
                attacker = state_i.individuals[idx]
                rival_best = nearest_state.best_individual
                r = np.random.rand(dim)
                new_pos = attacker.position + r * (rival_best.position - attacker.position)
                new_pos += 0.01 * np.random.randn(dim) * (ub - lb) * (1 - t_ratio)
                new_pos = np.clip(new_pos, lb, ub)
                new_fit = self._evaluate(new_pos, func)

                if new_fit < attacker.fitness:
                    attacker.position = new_pos
                    attacker.fitness = new_fit

            state_i.update_stats()

    # ----------------------------------------------------------
    #  动态种群资源再分配
    # ----------------------------------------------------------
    def _resource_rebalance(self, func, t_ratio):
        """
        动态资源再分配：强国获得更多个体，弱国缩减。
        模拟国力消长导致的人口流动。
        """
        alive_states = [s for s in self.states if s.alive]
        if len(alive_states) <= 2:
            return

        total_pop = sum(s.size for s in alive_states)
        dim = func.dim
        lb, ub = func.lb, func.ub

        # 按国力计算理想种群大小（国力强的分配更多）
        powers = np.array([s.national_power for s in alive_states])
        # 反转：国力值越小越强 -> 用 max - power 作为权重
        if np.max(powers) - np.min(powers) > 1e-30:
            weights = np.max(powers) - powers + 1e-10
        else:
            return  # 国力相同，无需调整

        weights = weights / np.sum(weights)
        target_sizes = np.round(weights * total_pop).astype(int)
        # 确保每国至少3个个体
        target_sizes = np.maximum(target_sizes, 3)
        # 调整总量
        diff = total_pop - np.sum(target_sizes)
        if diff > 0:
            target_sizes[np.argmax(weights)] += diff
        elif diff < 0:
            for _ in range(-diff):
                idx = np.argmax(target_sizes)
                if target_sizes[idx] > 3:
                    target_sizes[idx] -= 1

        for s, target in zip(alive_states, target_sizes):
            current = s.size
            if current > target and current > 3:
                # 缩减：淘汰最差个体
                s.individuals.sort(key=lambda ind: ind.fitness)
                s.individuals = s.individuals[:max(target, 3)]
            elif current < target:
                # 扩充：在本国最优解附近生成新个体
                for _ in range(target - current):
                    sigma = 0.05 * (ub - lb) * (1 - t_ratio) + 0.005 * (ub - lb)
                    new_pos = s.best_individual.position + np.random.randn(dim) * sigma
                    new_pos = np.clip(new_pos, lb, ub)
                    new_fit = self._evaluate(new_pos, func)
                    s.individuals.append(Individual(new_pos, new_fit))
            s.update_stats()

    # ----------------------------------------------------------
    #  阶段四: 征伐吞并
    # ----------------------------------------------------------
    def _annex_weakest(self, func):
        """征伐吞并：最弱国被灭，人口分配给最强国和次强国"""
        alive_states = [s for s in self.states if s.alive]
        if len(alive_states) <= 1:
            return

        alive_states.sort(key=lambda s: s.national_power)
        weakest = alive_states[-1]
        strongest = alive_states[0]
        second = alive_states[1] if len(alive_states) > 2 else strongest

        if self.verbose:
            print(f"    征伐吞并: {strongest.name} 灭 {weakest.name} "
                  f"(获得{weakest.size}个体)")

        refugees = sorted(weakest.individuals, key=lambda ind: ind.fitness)
        n_total = len(refugees)
        n_to_strongest = int(0.6 * n_total)
        n_to_second = n_total - n_to_strongest

        # 分配给最强国
        strongest.individuals.extend(refugees[:n_to_strongest])
        # 分配给次强国
        if second is not strongest:
            second.individuals.extend(refugees[n_to_strongest:])
            second.update_stats()

        weakest.individuals = []
        weakest.alive = False
        strongest.update_stats()

    # ----------------------------------------------------------
    #  阶段五: 大一统（多算子混合精细搜索）
    # ----------------------------------------------------------
    def _unification(self, func):
        """
        大一统阶段：合并所有存活国家，使用多算子混合精细搜索。
        - 精英引导搜索
        - Lévy飞行微调
        - 维度逐个优化
        """
        if self.verbose:
            print("\n" + "=" * 60)
            print("  阶段五：大一统 — 多算子混合精细搜索")
            print("=" * 60)

        all_individuals = []
        for s in self.states:
            if s.alive:
                all_individuals.extend(s.individuals)
                s.alive = False

        if not all_individuals:
            return

        dim = func.dim
        lb, ub = func.lb, func.ub
        remaining_iters = int(self.max_iter * (1 - self.unify_ratio))

        for t in range(remaining_iters):
            t_ratio = t / max(remaining_iters - 1, 1)
            sigma = (ub - lb) * (0.03 * (1 - t_ratio) + 0.001)

            for ind in all_individuals:
                r = np.random.rand()

                if r < 0.4:
                    # 精英引导 + 高斯扰动
                    w = np.random.rand(dim)
                    new_pos = ind.position + w * (self.global_best.position - ind.position) + \
                              np.random.randn(dim) * sigma
                elif r < 0.7:
                    # Lévy飞行微调
                    step = ReformStrategies.levy_flight(dim)
                    new_pos = self.global_best.position + 0.01 * step * (ub - lb) * (1 - t_ratio)
                elif r < 0.9:
                    # 维度学习（逐维优化思想）
                    new_pos = ind.position.copy()
                    d = np.random.randint(dim)
                    new_pos[d] = self.global_best.position[d] + np.random.randn() * sigma[d]
                else:
                    # DE/best/1 微调
                    if len(all_individuals) >= 3:
                        idxs = np.random.choice(len(all_individuals), 2, replace=False)
                        a, b = all_individuals[idxs[0]], all_individuals[idxs[1]]
                        F = 0.3 * np.random.rand()
                        new_pos = self.global_best.position + F * (a.position - b.position)
                    else:
                        new_pos = self.global_best.position + np.random.randn(dim) * sigma

                new_pos = np.clip(new_pos, lb, ub)
                new_fit = self._evaluate(new_pos, func)

                if new_fit < ind.fitness:
                    ind.position = new_pos
                    ind.fitness = new_fit

            self.convergence_curve.append(self.global_best.fitness)

    # ----------------------------------------------------------
    #  主优化流程
    # ----------------------------------------------------------
    def optimize(self, func):
        """
        执行战国七雄算法优化

        参数:
            func: 基准测试函数对象

        返回:
            best_position, best_fitness, convergence_curve
        """
        self.func_evals = 0
        self.convergence_curve = []

        # ===== 阶段一: 七国初立 =====
        self._initialize(func)
        self.convergence_curve.append(self.global_best.fitness)

        # ===== 阶段二~四 =====
        main_iters = int(self.max_iter * self.unify_ratio)

        if self.verbose:
            print("=" * 60)
            print("  阶段二~四：变法争雄 · 合纵连横 · 征伐吞并")
            print("=" * 60)

        for t in range(main_iters):
            t_ratio = t / max(main_iters - 1, 1)
            alive_states = [s for s in self.states if s.alive]

            # 变法争雄
            for state in alive_states:
                self._reform_evolve(state, func, t_ratio)

            # 合纵连横 (每3代触发)
            if t % 3 == 0 and len(alive_states) > 1:
                self._hezong_lianheng(func, t_ratio)

            # 远交近攻 (每7代触发)
            if t % 7 == 0 and len(alive_states) > 2:
                self._distant_ally_near_attack(func, t_ratio)

            # 策略自适应 (每25代检查)
            if t % 25 == 0 and t > 0 and len(alive_states) > 1:
                self._strategy_adaptation()

            # 动态资源再分配 (每30代)
            if t % 30 == 0 and t > 0 and len(alive_states) > 2:
                self._resource_rebalance(func, t_ratio)

            # 征伐吞并
            if (t + 1) % self.annex_cycle == 0 and len(alive_states) > 1:
                self._annex_weakest(func)

            self.convergence_curve.append(self.global_best.fitness)

            if self.verbose and (t + 1) % 100 == 0:
                n_alive = sum(1 for s in self.states if s.alive)
                print(f"  迭代 {t+1}/{main_iters} | 存活国家: {n_alive} | "
                      f"全局最优: {self.global_best.fitness:.6e}")

        # ===== 阶段五: 大一统 =====
        self._unification(func)

        if self.verbose:
            print(f"\n  最终结果: {self.global_best.fitness:.6e}")
            print(f"  函数评估次数: {self.func_evals}")
            print("=" * 60)

        return (self.global_best.position.copy(),
                self.global_best.fitness,
                self.convergence_curve)
