import pickle
import tqdm
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import multiprocessing
import collections
from ortools.linear_solver import pywraplp
from shocks import *

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

def eisenberg_noe_bailout_total_budget_given_shock(args):

    P_bar, A, C, X, B = args

    n = A.shape[0]

    # Create solver
    solver = pywraplp.Solver.CreateSolver('GLOP')

    # Create variables p_i
    variables = [solver.NumVar(0, P_bar[i, 0], 'p{}'.format(i)) for i in range(n)]
    subsidies = [solver.NumVar(0, solver.infinity(), 'y{}'.format(i)) for i in range(n)]

    # Create constraints
    for i in range(n):
        solver.Add(sum([(int(i == j) - A[j, i]) * variables[j] for j in range(n)]) <= C[i, 0] - X[i, 0] + subsidies[i])

    solver.Add(sum(subsidies) <= B)

    # Objective
    solver.Maximize(sum([variables[i] / P_bar[i, 0] for i in range(n)]))

    # Solve LP
    status = solver.Solve()

    return solver.Objective().Value()

def eisenberg_noe_bailout_threshold_model_given_shock(args):

    P_bar, A, C, X, B, w, threshold = args

    threshold_ind = (w <= threshold).astype(np.int64)
    k = threshold_ind.sum()

    n = A.shape[0]

    # Create solver
    solver = pywraplp.Solver.CreateSolver('GLOP')

    # Create variables p_i
    variables = [solver.NumVar(0, P_bar[i, 0], 'p{}'.format(i)) for i in range(n)]

    # Stimulus variable
    L =  solver.NumVar(0, solver.infinity(), 'L')

    # Create constraints
    for i in range(n):
        solver.Add(sum([(int(i == j) - A[j, i]) * variables[j] for j in range(n)]) <= C[i, 0] - X[i, 0] + threshold_ind[i, 0] * L)

    # Objective
    solver.Maximize(sum([B[i, 0] / P_bar[i, 0] * variables[i] for i in range(n)]) + (-k) * L)

    # Solve LP
    status = solver.Solve()

    if (L.solution_value() > 0):
        import pdb; pdb.set_trace()

    return L.solution_value(), solver.Objective().Value()

def eisenberg_noe(P_bar, A, C, X, n_iters=15):
    P_eq = P_bar
    for i in range(n_iters):
        P_eq = np.minimum(P_bar, np.maximum(0, A.T @ P_eq + C - X))

    return P_eq

def eisenberg_noe_bailout_given_shock(args):
    P_bar, A, C, X, L, S, u = args
    n = len(C)
    C_temp = np.copy(C)
    for v in S:
        C_temp[v] += L

    P_eq = eisenberg_noe(P_bar, A, C_temp, X)

    num_saved_without_u = np.isclose(P_eq, P_bar).astype(np.int64).sum()

    return num_saved_without_u

def eisenberg_noe_bailout(P_bar, A, C, L, S, u, num_iters=10):
    marginal_gain_total = 0
    shocks = []

    for i in range(num_iters):
        # X = generate_uniform_iid_shocks(C)
        X = generate_beta_iid_shocks(C)
        shocks.append(X)

    args = [(P_bar, A, C, X, L, S, u) for X in shocks]

    for arg in args:
        marginal_gain_total += eisenberg_noe_bailout_given_shock(arg)

    return marginal_gain_total / num_iters

def eisenberg_noe_bailout_total_budget(P_bar, A, C, B, num_iters=10):
    marginal_gain_total = 0
    shocks = []

    for i in range(num_iters):
        X = generate_uniform_iid_shocks(C)
        shocks.append(X)

    pool = multiprocessing.Pool(6)
    args = [(P_bar, A, C, X, B) for X in shocks]
    marginal_gains = pool.map(eisenberg_noe_bailout_total_budget_given_shock, args)

    marginal_gain_total = sum(marginal_gains)

    return marginal_gain_total / num_iters

def eisenberg_noe_bailout_threshold_model(P_bar, A, C, B, w, epsilon=0.1):

    n = A.shape[0]
    # num_iters = int(n * B.sum()**2 * np.log(n) / epsilon**2)
    num_iters = 40

    w_sorted = np.sort(np.unique(w))
    results = np.zeros_like(w_sorted)

    L_values = np.zeros(shape=(len(w_sorted), num_iters))
    objective_values = np.zeros(shape=(len(w_sorted), num_iters))

    for k, threshold in enumerate(w_sorted):
        for i in range(num_iters):
            X = generate_uniform_iid_shocks(C)
            args = (P_bar, A, C, X, B, w, threshold)

            L_opt, objective_opt = eisenberg_noe_bailout_threshold_model_given_shock(args)

            L_values[k, i] = L_opt
            objective_values[k, i] = objective_opt
            print(objective_opt)
            print(B.sum())

    mean_objective_values = objective_values.mean(-1)
    argmax = np.argmax(mean_objective_values, axis=0)

    optimal_value = mean_objective_values[argmax]
    optimal_L = L_values.mean(-1)[argmax]

    plt.figure()
    plt.hist(L_values[argmax, :])
    plt.savefig('L.png')

    plt.figure()
    plt.hist(objective_values[argmax, :])
    plt.savefig('opt.png')
