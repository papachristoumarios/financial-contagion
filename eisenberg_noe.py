import pickle
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import multiprocessing
from ortools.linear_solver import pywraplp

def generate_financial_network(n, p):
    G = nx.gnp_random_graph(n, p)
    return process_financial_network(G)

def process_financial_network(G):
    adj = nx.to_numpy_array(G).astype(np.float64)
    B = np.random.randint(100, size=(adj.shape[0], 1)).astype(np.float64)
    P = adj * np.random.randint(100, size=adj.shape).astype(np.float64)
    P_bar = P.sum(-1)
    P_bar = P_bar.reshape(len(P_bar), 1) + B
    C = np.random.randint(100, size=(adj.shape[0], 1)).astype(np.float64)
    A = P

    for i in range(P.shape[0]):
        if P_bar[i].sum() == 0:
            A[i] = 0 * A[i]
        else:
            A[i] /= P_bar[i]
    return G, adj, B, P, P_bar, C, A

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

def eisenberg_noe(P_bar, A, C, X, n_iters=15):
    P_eq = P_bar
    for i in range(n_iters):
        P_eq = np.minimum(P_bar, np.maximum(0, A.T @ P_eq + C - X))

    return P_eq

def eisenberg_noe_bailout_given_shock(args):
    P_bar, A, C, X, L, S, u = args
    n = len(C)
    C_temp = C
    for v in S:
        C_temp[v] += L

    P_eq = eisenberg_noe(P_bar, A, C_temp, X)

    num_saved_without_u = n - (P_eq < P_bar).astype(np.int64).sum()

    if not u:
        return num_saved_without_u

    C_temp[u] += L
    P_eq = eisenberg_noe(P_bar, A, C_temp, X)

    num_saved_with_u = n - (P_eq < P_bar).astype(np.int64).sum()

    marginal_gain = num_saved_with_u - num_saved_without_u
    return marginal_gain

def eisenberg_noe_bailout(P_bar, A, C, L, S, u, num_iters=10):
    marginal_gain_total = 0
    shocks = []

    for i in range(num_iters):
        X = np.zeros_like(C)
        for i in range(len(X)):
            X[i] = np.random.randint(low=0, high=1+int(C[i]))
        shocks.append(X)

    pool = multiprocessing.Pool(6)
    args = [(P_bar, A, C, X, L, S, u) for X in shocks]
    marginal_gains = pool.map(eisenberg_noe_bailout_given_shock, args)

    marginal_gain_total = sum(marginal_gains)

    return marginal_gain_total / num_iters

def eisenberg_noe_bailout_total_budget(P_bar, A, C, B, num_iters=10):
    marginal_gain_total = 0
    shocks = []

    for i in range(num_iters):
        X = np.zeros_like(C)
        for i in range(len(X)):
            X[i] = np.random.randint(low=0, high=1+int(C[i]))
        shocks.append(X)

    pool = multiprocessing.Pool(6)
    args = [(P_bar, A, C, X, B) for X in shocks]
    marginal_gains = pool.map(eisenberg_noe_bailout_total_budget_given_shock, args)

    marginal_gain_total = sum(marginal_gains)

    return marginal_gain_total / num_iters

if __name__ == '__main__':
    with open('cb.pickle', 'rb') as f:
        G = pickle.load(f)

    G = nx.gnp_random_graph(40, 0.7).to_directed()

    n = len(G)

    G, adj, B, P, P_bar, C, A = process_financial_network(G)

    print(eisenberg_noe_bailout_total_budget(P_bar, A, C, 100, num_iters=10))


    expected_numer_of_saved_nodes_no_intervention = eisenberg_noe_bailout(P_bar, A, C, 0, set(), None)

    L = 100
    min_num_default, min_num_default_arg = n, None

    k = 10
    k_range = np.arange(k)
    V = set(range(n))
    S = set()

    centralities = nx.algorithms.centrality.betweenness_centrality(G)
    centralities = list(centralities.items())
    centralities = list(sorted(centralities, key=lambda x: -x[-1]))

    out_degrees = list(sorted([(v, G.out_degree(v)) for v in G], key=lambda x: -x[-1]))

    expected_number_of_saved_nodes_greedy = []
    expected_number_of_saved_nodes_centralities = []
    expected_number_of_saved_nodes_out_degrees = []

    for i in k_range:
        best, best_arg = -1, None

        for u in V - S:
            value = eisenberg_noe_bailout(P_bar, A, C, L, S | {u}, None)

            if value >= best:
                best = value
                best_arg = u

        S |= {best_arg}

        expected_number_of_saved_nodes_greedy.append(best)

        S_centralities = set([x[0] for x in centralities[:k]])

        expected_number_of_saved_nodes_centralities.append(eisenberg_noe_bailout(
            P_bar, A, C, L, S_centralities, None))

        S_out_degrees = set([x[0] for x in out_degrees[:k]])

        expected_number_of_saved_nodes_out_degrees.append(eisenberg_noe_bailout(
            P_bar, A, C, L, S_out_degrees, None))

    plt.figure()
    plt.plot(k_range, expected_number_of_saved_nodes_greedy, label='Greedy')
    plt.plot(k_range, expected_number_of_saved_nodes_centralities, label='Top-k Centralities')
    plt.plot(k_range, expected_number_of_saved_nodes_out_degrees, label='Top-k Out-degrees')
    plt.legend()
    plt.show()
