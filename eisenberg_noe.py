from shocks import *
from ortools.linear_solver import pywraplp
import collections
import multiprocessing
import matplotlib.pyplot as plt
import pickle
import tqdm
import utils
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')


def generate_financial_network(n, p):
    G = nx.gnp_random_graph(n, p).to_directed()
    return process_financial_network(G)


def process_financial_network(G):
    # Adjacency matrix of G
    adj = nx.to_numpy_array(G).astype(np.float64)

    # Liability matrix. The entry P[i, j] denotes a liability of i to j
    P = adj * np.random.randint(2, size=adj.shape).astype(np.float64)

    # External assets of each node
    C = np.random.lognormal(mean=10, sigma=1, size=(adj.shape[0], 1))
    # np.random.randint(low=1, high=100, size=(adj.shape[0], 1)).astype(np.float64)

    B = 0.25 * C

    # Total external liabilities of i
    P_bar = P.sum(0)
    P_bar = P_bar.reshape(len(P_bar), 1) + B

    # Relative liability matrix
    A = np.copy(P)

    for i in range(P.shape[0]):
        if P_bar[i].sum() == 0:
            A[i] = 0 * A[i]
        else:
            A[i] /= P_bar[i]

    # Wealth of each node (external assets + liabilities of others - liabilities of node)
    w = C + P.sum(-1).reshape(len(P_bar), 1) - P_bar

    return G, adj, B, P, P_bar, C, A, w

def eisenberg_noe_bailout_randomized_rounding_given_shock(args):

    P_bar, A, C, X, L, b, k, gini, p_minority, v, tol = args

    beta = A.sum(-1)
    network_based = True

    n = A.shape[0]

    # Create solver
    solver = pywraplp.Solver.CreateSolver('GLOP')

    # Create variables p_i

    payment_variables = [solver.NumVar(0, P_bar[i, 0], 'p{}'.format(i)) for i in range(n)]
    stimuli_variables = [solver.NumVar(0, 1, 'z{}'.format(i)) for i in range(n)]

    # Create constraints
    for i in range(n):
        if isinstance(L, int):
            solver.Add(sum([(int(i == j) - A[j, i]) * payment_variables[j]
                            for j in range(n)]) <= C[i, 0] - X[i, 0] + L * stimuli_variables[i])
        elif isinstance(L, np.ndarray):
            solver.Add(sum([(int(i == j) - A[j, i]) * payment_variables[j]
                            for j in range(n)]) <= C[i, 0] - X[i, 0] + L[i, 0] * stimuli_variables[i])

    if isinstance(L, int):
        solver.Add(sum(stimuli_variables) == k)
    elif isinstance(L, np.ndarray):
        solver.Add(sum([stimuli_variables[i] * L[i, 0] for i in range(len(stimuli_variables))]) <= k * b)

    if gini:
        gini_helper_variables = {}
        for i in range(n):
            for j in range(n):
                gini_helper_variables[i, j] = solver.NumVar(0, np.inf, 'pi{}{}'.format(i, j))
                if isinstance(L, int):
                    if not(network_based) or (network_based and A[i, j] > 0):
                        solver.Add(-gini_helper_variables[i, j] <= stimuli_variables[i] - stimuli_variables[j])
                        solver.Add(stimuli_variables[i] - stimuli_variables[j] <= gini_helper_variables[i, j])

                elif isinstance(L, np.ndarray):
                    if not(network_based) or (network_based and A[i, j] > 0):
                        solver.Add(-gini_helper_variables[i, j] <= L[i, 0] * stimuli_variables[i] - L[j, 0] * stimuli_variables[j])
                        solver.Add(stimuli_variables[i] - stimuli_variables[j] <= gini_helper_variables[i, j])

        if isinstance(L, int):
            if p_minority is None and not network_based:
                solver.Add(sum(gini_helper_variables.values()) <= 2 * n * gini * k)
            elif not(p_minority is None):
                solver.Add(sum([gini_helper_variables[i, j] * p_minority[i, 0] * (1 - p_minority[j, 0]) for (i, j) in gini_helper_variables]) <= 2 * np.sum(1 - p_minority) * gini * sum([p_minority[i, 0] * stimuli_variables[i] for i in range(n)]))
            elif network_based:
                solver.Add(sum([gini_helper_variables[i, j] * A[i, j] for (i, j) in gini_helper_variables]) <= gini * sum([beta[i] * stimuli_variables[i] for i in range(n)]))

        elif isinstance(L, np.ndarray):
            if p_minority is None and not network_based:
                solver.Add(sum(gini_helper_variables.values()) <= 2 * n * gini * k * b)
            elif not(p_minority is None):
                solver.Add(sum([gini_helper_variables[i, j] * p_minority[i, 0] * (1 - p_minority[j, 0]) for (i, j) in gini_helper_variables]) <= 2 * np.sum(1- p_minority) * gini * sum([p_minority[i, 0] * L[i, 0] * stimuli_variables[i] for i in range(n)]))
            elif network_based:
                solver.Add(sum([gini_helper_variables[i, j] * A[i, j] for (i, j) in gini_helper_variables]) <= gini * sum([beta[i] * stimuli_variables[i] * L[i, 0] for i in range(n)]))

    # Objective
    solver.Maximize(sum([v[i, 0] * payment_variables[i] for i in range(n)]))

    # Solve LP
    status = solver.Solve()

    fractional_stimuli = np.array([z.solution_value() for z in stimuli_variables])


    while True:
        uniform_variables = np.random.uniform(size=fractional_stimuli.shape)
        realized_stimuli = (uniform_variables <= fractional_stimuli).astype(np.float64)

        if isinstance(L, int) and realized_stimuli.sum() <= k + tol:  # Tol should be something like O(sqrt(k))
            S = set(np.where(realized_stimuli == 1)[0].tolist())
            break
        elif isinstance(L, np.ndarray) and np.dot(realized_stimuli, L.flatten()) <= k * b + tol:
            S = set(np.where(realized_stimuli == 1)[0].tolist())
            break

    sol = eisenberg_noe_bailout_given_shock((P_bar, A, C, X, L, S, None, v))
    opt_lp = solver.Objective().Value()

    return S, fractional_stimuli, sol, opt_lp

def eisenberg_noe_bailout_randomized_rounding_min_default_given_shock(args):

    P_bar, A, C, X, L, b, k, gini, p_minority, eps, tol = args

    n = A.shape[0]

    # Create solver
    solver = pywraplp.Solver.CreateSolver('GLOP')

    # Create variables p_i
    min_var = solver.NumVar(0, 1, 's')
    payment_variables = [solver.NumVar(0, P_bar[i, 0], 'p{}'.format(i)) for i in range(n)]
    stimuli_variables = [solver.NumVar(0, 1, 'z{}'.format(i)) for i in range(n)]

    # Create constraints
    for i in range(n):
        if isinstance(L, int):
            solver.Add(sum([(int(i == j) - A[j, i]) * payment_variables[j]
                            for j in range(n)]) <= C[i, 0] - X[i, 0] + L * stimuli_variables[i])
        elif isinstance(L, np.ndarray):
            solver.Add(sum([(int(i == j) - A[j, i]) * payment_variables[j]
                            for j in range(n)]) <= C[i, 0] - X[i, 0] + L[i, 0] * stimuli_variables[i])

        solver.Add(payment_variables[i] >= min_var * P_bar[i, 0])

    if isinstance(L, int):
        solver.Add(sum(stimuli_variables) == k)
    elif isinstance(L, np.ndarray):
        solver.Add(sum([stimuli_variables[i] * L[i, 0]]) <= k * b)

    # Objective
    solver.Maximize(min_var + eps / (2 * k * L) * sum(payment_variables))

    # Solve LP
    status = solver.Solve()

    fractional_stimuli = np.array([z.solution_value() for z in stimuli_variables])

    while True:
        uniform_variables = np.random.uniform(size=fractional_stimuli.shape)
        realized_stimuli = (uniform_variables <= fractional_stimuli).astype(np.float64)

        if isinstance(L, int) and np.isclose(realized_stimuli.sum(), k, atol=tol):  # Tol should be something like O(sqrt(k))
            S = set(np.where(realized_stimuli == 1)[0].tolist())
            break
        elif isinstance(L, np.ndarray) and np.dot(realized_stimuli, L.flatten()) <= k * b + tol:
            S = set(np.where(realized_stimuli == 1)[0].tolist())
            break

    sol = eisenberg_noe_bailout_min_default_given_shock((P_bar, A, C, X, L, S, None, eps))
    opt_lp = solver.Objective().Value()

    return S, fractional_stimuli, sol, opt_lp

def eisenberg_noe(P_bar, A, C, X):
    P_eq = P_bar.copy()
    P_eq_prev = P_eq.copy()
    while True:
        P_eq = np.minimum(P_bar, np.maximum(0, A.T @ P_eq + C - X))

        if np.allclose(P_eq, P_eq_prev):
            break
        else:
            P_eq_prev = P_eq.copy()

    return P_eq

def eisenberg_noe_bailout_given_shock(args):
    P_bar, A, C, X, L, S, u, v = args
    n = len(C)
    C_temp = np.copy(C)
    for z in S:
        if isinstance(L, int):
            C_temp[z] += L
        elif isinstance(L, np.ndarray):
            C_temp[z] += L[z, 0]

    P_eq = eisenberg_noe(P_bar, A, C_temp, X)

    return (v.T @ P_eq)[0, 0]

def eisenberg_noe_bailout_min_default_given_shock(args):
    P_bar, A, C, X, L, S, u, eps = args
    n = len(C)
    C_temp = np.copy(C)
    for z in S:
        if isinstance(L, int):
            C_temp[z] += L
        elif isinstance(L, np.ndarray):
            C_temp[z] += L[z, 0]

    k = len(S)
    P_eq = eisenberg_noe(P_bar, A, C_temp, X)

    return (P_eq / P_bar).min() + eps / (2 * k * L) * P_eq.sum()

def eisenberg_noe_bailout(P_bar, A, C, L, S, u, v, num_iters, workers):
    shocks = []

    for i in range(num_iters):
        # X = generate_uniform_iid_shocks(C)
        X = generate_beta_iid_shocks(C)
        shocks.append(X)

    args = [(P_bar, A, C, X, L, S, u, v) for X in shocks]

    with multiprocessing.pool.ThreadPool(workers) as pool:
        marginal_gains_objectives = pool.map(
            eisenberg_noe_bailout_given_shock, args)

    marginal_gains_objective_mean = np.mean(marginal_gains_objectives)
    marginal_gains_objective_stdev = np.std(marginal_gains_objectives, ddof=1)

    return marginal_gains_objective_mean, marginal_gains_objective_stdev

def eisenberg_noe_bailout_greedy(P_bar, A, C, L, b, k, V, S, v, num_iters, workers):

    best, best_arg = (-1, -1), None
    for u in V - S:
        if isinstance(L, int) or (isinstance(L, np.ndarray) and L[u, 0] + sum([L[i, 0] for i in S]) <= k * b):
            value = eisenberg_noe_bailout(
                P_bar, A, C, L, S | {u}, None, v, num_iters=num_iters, workers=workers)
            if value[0] >= best[0]:
                best = value
                best_arg = u

    if not (best_arg is None):
        S |= {best_arg}
        if isinstance(L, int):
            return S, best
        elif isinstance(L, np.ndarray):
            return eisenberg_noe_bailout_greedy(P_bar, A, C, L, b, k, S, v, num_iters, workers)
    else:
        best = eisenberg_noe_bailout(
            P_bar, A, C, L, S, None, v, num_iters=num_iters, workers=workers)

    return S, best

def eisenberg_noe_bailout_greedy_min_default(P_bar, A, C, L, b, k, V, S, eps, num_iters, workers):

    best, best_arg = (-1, -1), None
    for u in V - S:
        if isinstance(L, int) or (isinstance(L, np.ndarray) and L[u, 0] + sum([L[i, 0] for i in S]) <= k * b):
            value = eisenberg_noe_bailout_min_default(
                P_bar, A, C, L, S | {u}, None, eps, num_iters=num_iters, workers=workers)
            if value[0] >= best[0]:
                best = value
                best_arg = u

    if not(best_arg is None):
        S |= {best_arg}
        if isinstance(L, int):
            return S, best
        elif isinstance(L, np.ndarray):
            return eisenberg_noe_bailout_greedy_min_default(P_bar, A, C, L, b, k, V, S, eps, num_iters, workers)
    else:
        best = eisenberg_noe_bailout_min_default(
            P_bar, A, C, L, S, None, eps, num_iters=num_iters, workers=workers)

        return S, best

def eisenberg_noe_bailout_min_default(P_bar, A, C, L, S, u, eps, num_iters, workers):
    shocks = []

    for i in range(num_iters):
        # X = generate_uniform_iid_shocks(C)
        X = generate_beta_iid_shocks(C)
        shocks.append(X)

    args = [(P_bar, A, C, X, L, S, u, eps) for X in shocks]

    with multiprocessing.pool.ThreadPool(workers) as pool:
        marginal_gains_objectives = pool.map(
            eisenberg_noe_bailout_min_default_given_shock, args)

    marginal_gains_objective_mean = np.mean(marginal_gains_objectives)
    marginal_gains_objective_stdev = np.std(marginal_gains_objectives, ddof=1)

    return marginal_gains_objective_mean, marginal_gains_objective_stdev

def eisenberg_noe_bailout_randomized_rounding(P_bar, A, C, L, b, k, gini, p_minority, v, tol, num_iters, workers):
    shocks = []

    for i in range(num_iters):
        # X = generate_uniform_iid_shocks(C)
        X = generate_beta_iid_shocks(C)
        shocks.append(X)

    args = [(P_bar, A, C, X, L, b, k, gini, p_minority, v, tol) for X in shocks]

    with multiprocessing.pool.ThreadPool(workers) as pool:
        marginal_gains = pool.map(
            eisenberg_noe_bailout_randomized_rounding_given_shock, args)

    marginal_gains_sets = [x[0] for x in marginal_gains]
    marginal_gains_fractional_stimuli = np.vstack([x[1] for x in marginal_gains])
    marginal_gains_sol = [x[2] for x in marginal_gains]
    marginal_gains_opt_lp = [x[3] for x in marginal_gains]

    marginal_gains_sol_mean = np.mean(marginal_gains_sol)
    marginal_gains_sol_stdev = np.std(marginal_gains_sol, ddof=1)

    marginal_gains_opt_lp_mean = np.mean(marginal_gains_opt_lp)
    marginal_gains_opt_lp_stdev = np.std(marginal_gains_opt_lp, ddof=1)

    marginal_gains_fractional_stimuli_mean = np.mean(marginal_gains_fractional_stimuli, axis=0)
    marginal_gains_fractional_stimuli_stdev = np.std(
        marginal_gains_fractional_stimuli, axis=0, ddof=1)

    return marginal_gains_sol_mean, marginal_gains_sol_stdev, marginal_gains_opt_lp_mean, marginal_gains_opt_lp_stdev, marginal_gains_fractional_stimuli_mean, marginal_gains_fractional_stimuli_stdev

def eisenberg_noe_bailout_randomized_rounding_min_default(P_bar, A, C, L, b, k, gini, p_minority, eps, tol, num_iters, workers):
    shocks = []

    for i in range(num_iters):
        # X = generate_uniform_iid_shocks(C)
        X = generate_beta_iid_shocks(C)
        shocks.append(X)

    args = [(P_bar, A, C, X, L, b, k, gini, p_minority, eps, tol) for X in shocks]

    with multiprocessing.pool.ThreadPool(workers) as pool:
        marginal_gains = pool.map(
            eisenberg_noe_bailout_randomized_rounding_min_default_given_shock, args)

    marginal_gains_sets = [x[0] for x in marginal_gains]
    marginal_gains_fractional_stimuli = np.vstack([x[1] for x in marginal_gains])
    marginal_gains_sol = [x[2] for x in marginal_gains]
    marginal_gains_opt_lp = [x[3] for x in marginal_gains]

    marginal_gains_sol_mean = np.mean(marginal_gains_sol)
    marginal_gains_sol_stdev = np.std(marginal_gains_sol, ddof=1)

    marginal_gains_opt_lp_mean = np.mean(marginal_gains_opt_lp)
    marginal_gains_opt_lp_stdev = np.std(marginal_gains_opt_lp, ddof=1)

    marginal_gains_fractional_stimuli_mean = np.mean(marginal_gains_fractional_stimuli, axis=0)
    marginal_gains_fractional_stimuli_stdev = np.std(
        marginal_gains_fractional_stimuli, axis=0, ddof=1)

    return marginal_gains_sol_mean, marginal_gains_sol_stdev, marginal_gains_opt_lp_mean, marginal_gains_opt_lp_stdev, marginal_gains_fractional_stimuli_mean, marginal_gains_fractional_stimuli_stdev

def min_budget_for_solvency(P_bar, A, C, k, v, delta, num_iters, workers):

    L_range = np.linspace(0, C.max(), int(1 / delta))

    lo = 0
    hi = len(L_range) - 1
    ideal_objective = np.dot(v, P_bar)

    while lo < hi:
        mid = (lo + hi) // 2
        marginal_gains_objective_mean, _, _, _ = eisenberg_noe_bailout_randomized_rounding(P_bar, A, C, L_range[mid], 1, k, v, num_iters, workers)

        if marginal_gains_objective_mean < ideal_objective:
            lo = mid + 1
        else:
            hi = mid - 1

    return L[mid]
