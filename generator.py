import networkx as nx
import numpy as np

def generate_random_data(seed=42, random_graph='ER', distribution='exponential', alpha=0.14, G=None, n=10):
    if random_graph == 'ER' and not G:
        G = generate_er(n, p=0.8, seed=seed)
    elif random_graph == 'SF' and not G:
        G = generate_scale_free(n, alpha=2, seed=seed)
    elif random_graph == 'CP' and not G:
        G = generate_core_periphery(n, p=0.7, seed=seed)

    if G:
        n = len(G)

    distributions = {
        'exponential' : lambda size: np.random.exponential(1, size=size),
        'pareto' : lambda size: np.random.pareto(1, size=size),
        'lognormal' : lambda size: np.random.lognormal(0, 1, size=size)
    }

    adj = nx.to_numpy_array(G)
    outdegree = adj.sum(0)
    indegree = adj.sum(-1)

    liabilities = adj * distributions[distribution]((n, n))

    internal_assets = liabilities.sum(-1).reshape((n, 1))
    internal_liabilities = liabilities.sum(0).reshape((n, 1))

    external_assets = n * distributions[distribution]((n, 1))
    external_liabilities = alpha * external_assets

    P_bar = internal_liabilities + external_liabilities

    A = np.copy(liabilities)
    for i in range(liabilities.shape[0]):
        A[i] /= P_bar[i]

    wealth = external_assets + internal_assets - external_liabilities - internal_liabilities

    if np.any(wealth < 0):
        return generate_random_data(seed, random_graph, distribution, alpha)
    else:
        return A, P_bar, liabilities, adj, internal_assets, internal_liabilities, outdegree, indegree, external_assets, external_liabilities, wealth, G


def generate_core_periphery(n, p, p_cc = 0.8, p_cp = 0.4, p_pp = 0.1, seed=100):
    n_c = int(n ** p)
    n_p = n - n_c

    sizes = [n_c, n_p]
    p = [[p_cc, p_cp], [p_cp, p_pp]]

    return nx.generators.community.stochastic_block_model(sizes, p, seed=seed, directed=True)

def generate_scale_free(n, alpha, seed=100):
    return nx.generators.directed.scale_free_graph(n, alpha, seed=seed, directed=True)

def generate_er(n, p=0.8, seed=100):
    return nx.generators.random_graphs.gnp_random_graph(n, p, seed=seed, directed=True)
