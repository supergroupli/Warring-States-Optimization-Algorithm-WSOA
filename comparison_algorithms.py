"""
对比算法集合
包含 10 种经典及最新元启发式算法
所有算法统一接口: (func, pop_size, max_iter) -> (best_pos, best_fit, convergence)
"""

import numpy as np


# ============================================================
#  粒子群优化 (Particle Swarm Optimization, PSO) — 1995
# ============================================================
def pso_optimize(func, pop_size=140, max_iter=1000):
    dim = func.dim
    lb, ub = func.lb, func.ub

    positions = lb + np.random.rand(pop_size, dim) * (ub - lb)
    velocities = np.random.randn(pop_size, dim) * 0.1 * (ub - lb)
    fitnesses = np.array([func.evaluate(positions[i]) for i in range(pop_size)])

    pbest_pos = positions.copy()
    pbest_fit = fitnesses.copy()
    gbest_idx = np.argmin(fitnesses)
    gbest_pos = positions[gbest_idx].copy()
    gbest_fit = fitnesses[gbest_idx]
    convergence = [gbest_fit]

    w_max, w_min = 0.9, 0.4
    c1, c2 = 2.0, 2.0

    for t in range(max_iter):
        w = w_max - (w_max - w_min) * t / max_iter
        r1 = np.random.rand(pop_size, dim)
        r2 = np.random.rand(pop_size, dim)
        velocities = (w * velocities +
                      c1 * r1 * (pbest_pos - positions) +
                      c2 * r2 * (gbest_pos - positions))
        positions = np.clip(positions + velocities, lb, ub)

        for i in range(pop_size):
            fitnesses[i] = func.evaluate(positions[i])
            if fitnesses[i] < pbest_fit[i]:
                pbest_fit[i] = fitnesses[i]
                pbest_pos[i] = positions[i].copy()
                if fitnesses[i] < gbest_fit:
                    gbest_fit = fitnesses[i]
                    gbest_pos = positions[i].copy()
        convergence.append(gbest_fit)

    return gbest_pos, gbest_fit, convergence


# ============================================================
#  差分进化 (Differential Evolution, DE) — 1997
# ============================================================
def de_optimize(func, pop_size=140, max_iter=1000):
    dim = func.dim
    lb, ub = func.lb, func.ub
    F, CR = 0.5, 0.9

    population = lb + np.random.rand(pop_size, dim) * (ub - lb)
    fitnesses = np.array([func.evaluate(population[i]) for i in range(pop_size)])

    gbest_idx = np.argmin(fitnesses)
    gbest_pos = population[gbest_idx].copy()
    gbest_fit = fitnesses[gbest_idx]
    convergence = [gbest_fit]

    for t in range(max_iter):
        for i in range(pop_size):
            idxs = np.random.choice([j for j in range(pop_size) if j != i], 3, replace=False)
            a, b, c = population[idxs[0]], population[idxs[1]], population[idxs[2]]
            mutant = np.clip(a + F * (b - c), lb, ub)

            trial = population[i].copy()
            j_rand = np.random.randint(dim)
            for d in range(dim):
                if np.random.rand() < CR or d == j_rand:
                    trial[d] = mutant[d]

            trial_fit = func.evaluate(trial)
            if trial_fit < fitnesses[i]:
                population[i] = trial
                fitnesses[i] = trial_fit
                if trial_fit < gbest_fit:
                    gbest_fit = trial_fit
                    gbest_pos = trial.copy()
        convergence.append(gbest_fit)

    return gbest_pos, gbest_fit, convergence


# ============================================================
#  遗传算法 (Genetic Algorithm, GA) — 1975
# ============================================================
def ga_optimize(func, pop_size=140, max_iter=1000):
    dim = func.dim
    lb, ub = func.lb, func.ub
    pc, pm = 0.9, 0.1
    eta_c, eta_m = 20, 20
    tournament_size = 3

    population = lb + np.random.rand(pop_size, dim) * (ub - lb)
    fitnesses = np.array([func.evaluate(population[i]) for i in range(pop_size)])

    gbest_idx = np.argmin(fitnesses)
    gbest_pos = population[gbest_idx].copy()
    gbest_fit = fitnesses[gbest_idx]
    convergence = [gbest_fit]

    for t in range(max_iter):
        def tournament():
            idxs = np.random.choice(pop_size, tournament_size, replace=False)
            return idxs[np.argmin(fitnesses[idxs])]

        offspring = np.empty_like(population)
        for i in range(0, pop_size, 2):
            p1, p2 = tournament(), tournament()
            c1, c2 = population[p1].copy(), population[p2].copy()

            if np.random.rand() < pc:
                for d in range(dim):
                    if np.random.rand() < 0.5 and abs(c1[d] - c2[d]) > 1e-14:
                        beta = 1.0 + 2.0 * min(c1[d] - lb[d], ub[d] - c1[d]) / abs(c1[d] - c2[d])
                        alpha = 2.0 - beta ** (-(eta_c + 1))
                        u = np.random.rand()
                        betaq = ((u * alpha) ** (1.0 / (eta_c + 1)) if u <= 1.0 / alpha
                                 else (1.0 / (2.0 - u * alpha)) ** (1.0 / (eta_c + 1)))
                        c1[d], c2[d] = (0.5 * ((1 + betaq) * c1[d] + (1 - betaq) * c2[d]),
                                         0.5 * ((1 - betaq) * c1[d] + (1 + betaq) * c2[d]))

            for child in [c1, c2]:
                for d in range(dim):
                    if np.random.rand() < pm:
                        delta1 = (child[d] - lb[d]) / (ub[d] - lb[d])
                        delta2 = (ub[d] - child[d]) / (ub[d] - lb[d])
                        u = np.random.rand()
                        if u < 0.5:
                            deltaq = (2 * u + (1 - 2 * u) * (1 - delta1) ** (eta_m + 1)) ** (1 / (eta_m + 1)) - 1
                        else:
                            deltaq = 1 - (2 * (1 - u) + 2 * (u - 0.5) * (1 - delta2) ** (eta_m + 1)) ** (1 / (eta_m + 1))
                        child[d] += deltaq * (ub[d] - lb[d])

            offspring[i] = np.clip(c1, lb, ub)
            if i + 1 < pop_size:
                offspring[i + 1] = np.clip(c2, lb, ub)

        offspring_fit = np.array([func.evaluate(offspring[i]) for i in range(pop_size)])
        worst_idx = np.argmax(offspring_fit)
        best_idx_old = np.argmin(fitnesses)
        if fitnesses[best_idx_old] < offspring_fit[worst_idx]:
            offspring[worst_idx] = population[best_idx_old].copy()
            offspring_fit[worst_idx] = fitnesses[best_idx_old]

        population = offspring
        fitnesses = offspring_fit

        cur_best_idx = np.argmin(fitnesses)
        if fitnesses[cur_best_idx] < gbest_fit:
            gbest_fit = fitnesses[cur_best_idx]
            gbest_pos = population[cur_best_idx].copy()
        convergence.append(gbest_fit)

    return gbest_pos, gbest_fit, convergence


# ============================================================
#  人工蜂群算法 (Artificial Bee Colony, ABC) — 2007
# ============================================================
def abc_optimize(func, pop_size=140, max_iter=1000):
    dim = func.dim
    lb, ub = func.lb, func.ub
    n_food = pop_size // 2
    limit = n_food * dim

    food = lb + np.random.rand(n_food, dim) * (ub - lb)
    fitnesses = np.array([func.evaluate(food[i]) for i in range(n_food)])
    trials = np.zeros(n_food)

    gbest_idx = np.argmin(fitnesses)
    gbest_pos = food[gbest_idx].copy()
    gbest_fit = fitnesses[gbest_idx]
    convergence = [gbest_fit]

    def abc_fitness(val):
        return 1.0 / (1.0 + val) if val >= 0 else 1.0 + abs(val)

    for t in range(max_iter):
        for i in range(n_food):
            k = np.random.choice([j for j in range(n_food) if j != i])
            d = np.random.randint(dim)
            new_food = food[i].copy()
            new_food[d] += np.random.uniform(-1, 1) * (food[i][d] - food[k][d])
            new_food = np.clip(new_food, lb, ub)
            new_fit = func.evaluate(new_food)
            if new_fit < fitnesses[i]:
                food[i] = new_food
                fitnesses[i] = new_fit
                trials[i] = 0
            else:
                trials[i] += 1

        fit_vals = np.array([abc_fitness(f) for f in fitnesses])
        probs = fit_vals / np.sum(fit_vals)
        for _ in range(n_food):
            i = np.random.choice(n_food, p=probs)
            k = np.random.choice([j for j in range(n_food) if j != i])
            d = np.random.randint(dim)
            new_food = food[i].copy()
            new_food[d] += np.random.uniform(-1, 1) * (food[i][d] - food[k][d])
            new_food = np.clip(new_food, lb, ub)
            new_fit = func.evaluate(new_food)
            if new_fit < fitnesses[i]:
                food[i] = new_food
                fitnesses[i] = new_fit
                trials[i] = 0
            else:
                trials[i] += 1

        for i in range(n_food):
            if trials[i] > limit:
                food[i] = lb + np.random.rand(dim) * (ub - lb)
                fitnesses[i] = func.evaluate(food[i])
                trials[i] = 0

        cur_best = np.argmin(fitnesses)
        if fitnesses[cur_best] < gbest_fit:
            gbest_fit = fitnesses[cur_best]
            gbest_pos = food[cur_best].copy()
        convergence.append(gbest_fit)

    return gbest_pos, gbest_fit, convergence


# ============================================================
#  灰狼优化算法 (Grey Wolf Optimizer, GWO) — 2014
# ============================================================
def gwo_optimize(func, pop_size=140, max_iter=1000):
    dim = func.dim
    lb, ub = func.lb, func.ub

    positions = lb + np.random.rand(pop_size, dim) * (ub - lb)
    fitnesses = np.array([func.evaluate(positions[i]) for i in range(pop_size)])

    sorted_idx = np.argsort(fitnesses)
    alpha_pos, alpha_fit = positions[sorted_idx[0]].copy(), fitnesses[sorted_idx[0]]
    beta_pos, beta_fit = positions[sorted_idx[1]].copy(), fitnesses[sorted_idx[1]]
    delta_pos, delta_fit = positions[sorted_idx[2]].copy(), fitnesses[sorted_idx[2]]
    convergence = [alpha_fit]

    for t in range(max_iter):
        a = 2.0 - 2.0 * t / max_iter

        for i in range(pop_size):
            r1, r2 = np.random.rand(dim), np.random.rand(dim)
            A1, C1 = 2 * a * r1 - a, 2 * r2
            X1 = alpha_pos - A1 * np.abs(C1 * alpha_pos - positions[i])

            r1, r2 = np.random.rand(dim), np.random.rand(dim)
            A2, C2 = 2 * a * r1 - a, 2 * r2
            X2 = beta_pos - A2 * np.abs(C2 * beta_pos - positions[i])

            r1, r2 = np.random.rand(dim), np.random.rand(dim)
            A3, C3 = 2 * a * r1 - a, 2 * r2
            X3 = delta_pos - A3 * np.abs(C3 * delta_pos - positions[i])

            positions[i] = np.clip((X1 + X2 + X3) / 3, lb, ub)
            fitnesses[i] = func.evaluate(positions[i])

            if fitnesses[i] < alpha_fit:
                delta_pos, delta_fit = beta_pos.copy(), beta_fit
                beta_pos, beta_fit = alpha_pos.copy(), alpha_fit
                alpha_pos, alpha_fit = positions[i].copy(), fitnesses[i]
            elif fitnesses[i] < beta_fit:
                delta_pos, delta_fit = beta_pos.copy(), beta_fit
                beta_pos, beta_fit = positions[i].copy(), fitnesses[i]
            elif fitnesses[i] < delta_fit:
                delta_pos, delta_fit = positions[i].copy(), fitnesses[i]

        convergence.append(alpha_fit)

    return alpha_pos, alpha_fit, convergence


# ============================================================
#  鲸鱼优化算法 (Whale Optimization Algorithm, WOA) — 2016
# ============================================================
def woa_optimize(func, pop_size=140, max_iter=1000):
    dim = func.dim
    lb, ub = func.lb, func.ub
    b = 1

    positions = lb + np.random.rand(pop_size, dim) * (ub - lb)
    fitnesses = np.array([func.evaluate(positions[i]) for i in range(pop_size)])

    gbest_idx = np.argmin(fitnesses)
    gbest_pos = positions[gbest_idx].copy()
    gbest_fit = fitnesses[gbest_idx]
    convergence = [gbest_fit]

    for t in range(max_iter):
        a = 2.0 - 2.0 * t / max_iter

        for i in range(pop_size):
            A = 2 * a * np.random.rand(dim) - a
            C = 2 * np.random.rand(dim)
            p = np.random.rand()

            if p < 0.5:
                if np.linalg.norm(A) < 1:
                    D = np.abs(C * gbest_pos - positions[i])
                    positions[i] = gbest_pos - A * D
                else:
                    rand_idx = np.random.randint(pop_size)
                    X_rand = positions[rand_idx]
                    D = np.abs(C * X_rand - positions[i])
                    positions[i] = X_rand - A * D
            else:
                D_prime = np.abs(gbest_pos - positions[i])
                l = np.random.uniform(-1, 1, dim)
                positions[i] = D_prime * np.exp(b * l) * np.cos(2 * np.pi * l) + gbest_pos

            positions[i] = np.clip(positions[i], lb, ub)
            fitnesses[i] = func.evaluate(positions[i])

            if fitnesses[i] < gbest_fit:
                gbest_fit = fitnesses[i]
                gbest_pos = positions[i].copy()

        convergence.append(gbest_fit)

    return gbest_pos, gbest_fit, convergence


# ============================================================
#  正弦余弦算法 (Sine Cosine Algorithm, SCA) — 2016
# ============================================================
def sca_optimize(func, pop_size=140, max_iter=1000):
    dim = func.dim
    lb, ub = func.lb, func.ub

    positions = lb + np.random.rand(pop_size, dim) * (ub - lb)
    fitnesses = np.array([func.evaluate(positions[i]) for i in range(pop_size)])

    gbest_idx = np.argmin(fitnesses)
    gbest_pos = positions[gbest_idx].copy()
    gbest_fit = fitnesses[gbest_idx]
    convergence = [gbest_fit]

    for t in range(max_iter):
        a = 2.0 - 2.0 * t / max_iter  # 线性递减

        for i in range(pop_size):
            r1 = a * np.random.rand(dim)
            r2 = 2 * np.pi * np.random.rand(dim)
            r3 = 2 * np.random.rand(dim)
            r4 = np.random.rand()

            if r4 < 0.5:
                positions[i] = positions[i] + r1 * np.sin(r2) * np.abs(r3 * gbest_pos - positions[i])
            else:
                positions[i] = positions[i] + r1 * np.cos(r2) * np.abs(r3 * gbest_pos - positions[i])

            positions[i] = np.clip(positions[i], lb, ub)
            fitnesses[i] = func.evaluate(positions[i])

            if fitnesses[i] < gbest_fit:
                gbest_fit = fitnesses[i]
                gbest_pos = positions[i].copy()

        convergence.append(gbest_fit)

    return gbest_pos, gbest_fit, convergence


# ============================================================
#  哈里斯鹰优化 (Harris Hawks Optimization, HHO) — 2019
# ============================================================
def hho_optimize(func, pop_size=140, max_iter=1000):
    dim = func.dim
    lb, ub = func.lb, func.ub

    positions = lb + np.random.rand(pop_size, dim) * (ub - lb)
    fitnesses = np.array([func.evaluate(positions[i]) for i in range(pop_size)])

    gbest_idx = np.argmin(fitnesses)
    gbest_pos = positions[gbest_idx].copy()
    gbest_fit = fitnesses[gbest_idx]
    convergence = [gbest_fit]

    def levy_flight(d, beta=1.5):
        from math import gamma
        sigma = (gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                 (gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.randn(d) * sigma
        v = np.random.randn(d)
        return u / (np.abs(v) ** (1 / beta))

    for t in range(max_iter):
        E0 = 2 * np.random.rand() - 1  # 初始能量
        E = 2 * E0 * (1 - t / max_iter)  # 逃逸能量

        for i in range(pop_size):
            q = np.random.rand()
            r = np.random.rand()

            if abs(E) >= 1:
                # 探索阶段
                if q >= 0.5:
                    rand_idx = np.random.randint(pop_size)
                    X_rand = positions[rand_idx]
                    positions[i] = X_rand - np.random.rand(dim) * np.abs(
                        X_rand - 2 * np.random.rand(dim) * positions[i])
                else:
                    positions[i] = (gbest_pos - np.mean(positions, axis=0)) - \
                                   np.random.rand(dim) * (lb + np.random.rand(dim) * (ub - lb))
            else:
                # 开发阶段
                J = 2 * (1 - np.random.rand())
                if r >= 0.5 and abs(E) >= 0.5:
                    # 软围攻
                    positions[i] = gbest_pos - E * np.abs(J * gbest_pos - positions[i])
                elif r >= 0.5 and abs(E) < 0.5:
                    # 硬围攻
                    positions[i] = gbest_pos - E * np.abs(gbest_pos - positions[i])
                elif r < 0.5 and abs(E) >= 0.5:
                    # 带Lévy的软围攻
                    Y = gbest_pos - E * np.abs(J * gbest_pos - positions[i])
                    Y = np.clip(Y, lb, ub)
                    Z = Y + np.random.rand(dim) * levy_flight(dim)
                    Z = np.clip(Z, lb, ub)
                    fY = func.evaluate(Y)
                    fZ = func.evaluate(Z)
                    if fY < fitnesses[i]:
                        positions[i] = Y
                    if fZ < fitnesses[i]:
                        positions[i] = Z
                else:
                    # 带Lévy的硬围攻
                    Y = gbest_pos - E * np.abs(J * gbest_pos - np.mean(positions, axis=0))
                    Y = np.clip(Y, lb, ub)
                    Z = Y + np.random.rand(dim) * levy_flight(dim)
                    Z = np.clip(Z, lb, ub)
                    fY = func.evaluate(Y)
                    fZ = func.evaluate(Z)
                    if fY < fitnesses[i]:
                        positions[i] = Y
                    if fZ < fitnesses[i]:
                        positions[i] = Z

            positions[i] = np.clip(positions[i], lb, ub)
            fitnesses[i] = func.evaluate(positions[i])

            if fitnesses[i] < gbest_fit:
                gbest_fit = fitnesses[i]
                gbest_pos = positions[i].copy()

        convergence.append(gbest_fit)

    return gbest_pos, gbest_fit, convergence


# ============================================================
#  海洋捕食者算法 (Marine Predators Algorithm, MPA) — 2020
# ============================================================
def mpa_optimize(func, pop_size=140, max_iter=1000):
    dim = func.dim
    lb, ub = func.lb, func.ub
    FADs = 0.2   # Fish Aggregating Devices效应
    P = 0.5      # 概率参数

    positions = lb + np.random.rand(pop_size, dim) * (ub - lb)
    fitnesses = np.array([func.evaluate(positions[i]) for i in range(pop_size)])

    gbest_idx = np.argmin(fitnesses)
    gbest_pos = positions[gbest_idx].copy()
    gbest_fit = fitnesses[gbest_idx]
    convergence = [gbest_fit]

    # 构造精英矩阵
    Elite = np.tile(gbest_pos, (pop_size, 1))

    def levy_step(d, beta=1.5):
        from math import gamma
        sigma = (gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                 (gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.randn(d) * sigma
        v = np.random.randn(d)
        return u / (np.abs(v) ** (1 / beta))

    for t in range(max_iter):
        CF = (1 - t / max_iter) ** (2 * t / max_iter)
        Elite = np.tile(gbest_pos, (pop_size, 1))

        for i in range(pop_size):
            R = np.random.rand(dim)

            if t < max_iter / 3:
                # 阶段1: 高速比率(探索)
                stepsize = R * (Elite[i] - R * positions[i])
                positions[i] = positions[i] + P * R * stepsize
            elif t < 2 * max_iter / 3:
                # 阶段2: 中间过渡
                if i < pop_size / 2:
                    stepsize = levy_step(dim) * (Elite[i] - R * positions[i])
                    positions[i] = positions[i] + P * R * stepsize
                else:
                    stepsize = R * (R * Elite[i] - positions[i])
                    positions[i] = Elite[i] + P * CF * stepsize
            else:
                # 阶段3: 低速比率(开发)
                stepsize = levy_step(dim) * (R * Elite[i] - positions[i])
                positions[i] = Elite[i] + P * CF * stepsize

            positions[i] = np.clip(positions[i], lb, ub)

            # FADs效应
            if np.random.rand() < FADs:
                u = np.random.rand(dim) < FADs
                positions[i] = positions[i] + CF * (lb + np.random.rand(dim) * (ub - lb)) * u
                positions[i] = np.clip(positions[i], lb, ub)

            fitnesses[i] = func.evaluate(positions[i])
            if fitnesses[i] < gbest_fit:
                gbest_fit = fitnesses[i]
                gbest_pos = positions[i].copy()

        convergence.append(gbest_fit)

    return gbest_pos, gbest_fit, convergence


# ============================================================
#  天鹰优化算法 (Aquila Optimizer, AO) — 2021
# ============================================================
def ao_optimize(func, pop_size=140, max_iter=1000):
    dim = func.dim
    lb, ub = func.lb, func.ub

    positions = lb + np.random.rand(pop_size, dim) * (ub - lb)
    fitnesses = np.array([func.evaluate(positions[i]) for i in range(pop_size)])

    gbest_idx = np.argmin(fitnesses)
    gbest_pos = positions[gbest_idx].copy()
    gbest_fit = fitnesses[gbest_idx]
    convergence = [gbest_fit]

    def levy_flight(d, beta=1.5):
        from math import gamma
        sigma = (gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                 (gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.randn(d) * sigma
        v = np.random.randn(d)
        return u / (np.abs(v) ** (1 / beta))

    for t in range(max_iter):
        t_ratio = t / max_iter
        mean_pos = np.mean(positions, axis=0)

        for i in range(pop_size):
            r = np.random.rand()

            if t_ratio <= 1/3:
                # 高空翱翔 (探索)
                positions[i] = gbest_pos * (1 - t_ratio) + \
                               (mean_pos - gbest_pos) * np.random.rand()
            elif t_ratio <= 2/3:
                if r < 0.5:
                    # 等高线飞行
                    theta = np.random.rand(dim) * 2 * np.pi
                    r_spiral = (t_ratio - 1/3) * np.random.rand(dim)
                    positions[i] = gbest_pos + r_spiral * np.cos(theta)
                else:
                    # Lévy飞行捕猎
                    positions[i] = gbest_pos + levy_flight(dim) * \
                                   (gbest_pos - positions[i]) * np.random.rand()
            else:
                if r < 0.5:
                    # 低空飞行攻击
                    positions[i] = (gbest_pos - mean_pos) * \
                                   np.random.rand() * 0.1 + \
                                   np.random.rand(dim) * (ub - lb) * \
                                   np.random.rand() * 0.01 + gbest_pos
                else:
                    # 步行攻击
                    QF = t_ratio ** 2 * np.random.rand()
                    levy = levy_flight(dim)
                    positions[i] = gbest_pos + QF * \
                                   (positions[i] - gbest_pos) * np.random.rand() - \
                                   levy * np.random.rand()

            positions[i] = np.clip(positions[i], lb, ub)
            fitnesses[i] = func.evaluate(positions[i])

            if fitnesses[i] < gbest_fit:
                gbest_fit = fitnesses[i]
                gbest_pos = positions[i].copy()

        convergence.append(gbest_fit)

    return gbest_pos, gbest_fit, convergence


# ============================================================
#  蜣螂优化算法 (Dung Beetle Optimizer, DBO) — 2023
# ============================================================
def dbo_optimize(func, pop_size=140, max_iter=1000):
    dim = func.dim
    lb, ub = func.lb, func.ub

    positions = lb + np.random.rand(pop_size, dim) * (ub - lb)
    fitnesses = np.array([func.evaluate(positions[i]) for i in range(pop_size)])

    gbest_idx = np.argmin(fitnesses)
    gbest_pos = positions[gbest_idx].copy()
    gbest_fit = fitnesses[gbest_idx]
    gworst_idx = np.argmax(fitnesses)
    gworst_pos = positions[gworst_idx].copy()
    convergence = [gbest_fit]

    # 分组比例: 滚球/繁殖/觅食/偷窃
    n_roll = int(pop_size * 0.3)
    n_breed = int(pop_size * 0.2)
    n_forage = int(pop_size * 0.3)
    n_steal = pop_size - n_roll - n_breed - n_forage

    for t in range(max_iter):
        t_ratio = t / max_iter

        # 更新最差解
        gworst_idx = np.argmax(fitnesses)
        gworst_pos = positions[gworst_idx].copy()

        for i in range(pop_size):
            if i < n_roll:
                # 滚球行为
                k = np.random.rand()
                b_val = 0.3
                alpha_val = np.random.rand(dim)
                if k < 0.9:
                    positions[i] = positions[i] + b_val * np.abs(positions[i] - gworst_pos) + \
                                   alpha_val * (positions[i] - gbest_pos)
                else:
                    positions[i] = positions[i] + np.tan(np.random.rand() * np.pi - np.pi/2) * \
                                   np.abs(positions[i] - positions[np.random.randint(pop_size)])

            elif i < n_roll + n_breed:
                # 繁殖行为
                R = np.random.rand(dim)
                positions[i] = gbest_pos + R * (positions[i] - lb) * (1 - t_ratio)

            elif i < n_roll + n_breed + n_forage:
                # 觅食行为
                R = np.random.rand(dim)
                positions[i] = positions[i] + np.random.randn(dim) * \
                               (positions[i] - gbest_pos) * (1 - t_ratio)

            else:
                # 偷窃行为
                positions[i] = gbest_pos + np.random.randn(dim) * \
                               (np.abs(positions[i] - gbest_pos) +
                                np.abs(positions[i] - gworst_pos)) / 2

            positions[i] = np.clip(positions[i], lb, ub)
            fitnesses[i] = func.evaluate(positions[i])

            if fitnesses[i] < gbest_fit:
                gbest_fit = fitnesses[i]
                gbest_pos = positions[i].copy()

        convergence.append(gbest_fit)

    return gbest_pos, gbest_fit, convergence


# ============================================================
#  算法注册表
# ============================================================
ALGORITHMS = {
    "PSO":  pso_optimize,
    "DE":   de_optimize,
    "GA":   ga_optimize,
    "ABC":  abc_optimize,
    "GWO":  gwo_optimize,
    "WOA":  woa_optimize,
    "SCA":  sca_optimize,
    "HHO":  hho_optimize,
    "MPA":  mpa_optimize,
    "AO":   ao_optimize,
    "DBO":  dbo_optimize,
}
