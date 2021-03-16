from shocks import *
from ortools.linear_solver import pywraplp
import collections
import multiprocessing
import matplotlib.pyplot as plt
import pickle
import tqdm
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

    P_bar, A, C, X, L, k, v, tol = args

    n = A.shape[0]

    # Create solver
    solver = pywraplp.Solver.CreateSolver('GLOP')

    # Create variables p_i

    payment_variables = [solver.NumVar(0, P_bar[i, 0], 'p{}'.format(i)) for i in range(n)]
    stimuli_variables = [solver.NumVar(0, 1, 'z{}'.format(i)) for i in range(n)]

    # Create constraints
    for i in range(n):
        solver.Add(sum([(int(i == j) - A[j, i]) * payment_variables[j]
                        for j in range(n)]) <= C[i, 0] - X[i, 0] + L * stimuli_variables[i])

    solver.Add(sum(stimuli_variables) == k)

    # Objective
    solver.Maximize(sum([v[i, 0] * payment_variables[i] for i in range(n)]))

    # Solve LP
    status = solver.Solve()

    fractional_stimuli = np.array([z.solution_value() for z in stimuli_variables])

    while True:
        uniform_variables = np.random.uniform(size=fractional_stimuli.shape)
        realized_stimuli = (uniform_variables <= fractional_stimuli).astype(np.float64)

        if np.isclose(realized_stimuli.sum(), k, atol=tol):  # Tol should be something like O(sqrt(k))
            S = set(np.where(realized_stimuli == 1)[0].tolist())
            break

    sol = eisenberg_noe_bailout_given_shock((P_bar, A, C, X, L, S, None, v))
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

def eisenberg_noe_bailout_given_shock_lp(args):

    P_bar, A, C, X, L, S, u, v = args

    n = A.shape[0]

    # Create solver
    solver = pywraplp.Solver.CreateSolver('GLOP')

    # Create variables p_i
    payment_variables = [solver.NumVar(0, P_bar[i, 0], 'p{}'.format(i)) for i in range(n)]

    # Indicator of S
    ind_S = np.zeros(n)
    ind_S[np.array(list(S), dtype=np.int64)] = 1

    # Create constraints
    for i in range(n):
        solver.Add(sum([(int(i == j) - A[j, i]) * payment_variables[j]
                        for j in range(n)]) <= C[i, 0] - X[i, 0] + L * ind_S[i])

    # Objective
    solver.Maximize(sum([v[i, 0] * payment_variables[i] for i in range(n)]))

    # Solve LP
    status = solver.Solve()

    return solver.Objective().Value()

def eisenberg_noe_bailout_given_shock(args):
    P_bar, A, C, X, L, S, u, v = args
    n = len(C)
    C_temp = np.copy(C)
    for z in S:
        C_temp[z] += L

    P_eq = eisenberg_noe(P_bar, A, C_temp, X)

    return (v.T @ P_eq)[0, 0]

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


def eisenberg_noe_bailout_randomized_rounding(P_bar, A, C, L, k, v, tol, num_iters, workers):
    shocks = []

    for i in range(num_iters):
        # X = generate_uniform_iid_shocks(C)
        X = generate_beta_iid_shocks(C)
        shocks.append(X)

    args = [(P_bar, A, C, X, L, k, v, tol) for X in shocks]

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

def min_budget_for_solvency(P_bar, A, C, k, v, delta, num_iters, workers):

    L_range = np.linspace(0, C.max(), int(1 / delta))

    lo = 0
    hi = len(L_range) - 1
    ideal_objective = np.dot(v, P_bar)

    while lo < hi:
        mid = (lo + hi) // 2
        marginal_gains_objective_mean, _, _, _ = eisenberg_noe_bailout_randomized_rounding(P_bar, A, C, L_range[mid], k, v, num_iters, workers)

        if marginal_gains_objective_mean < ideal_objective:
            lo = mid + 1
        else:
            hi = mid - 1

    return L[mid]
