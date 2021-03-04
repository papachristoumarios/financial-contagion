import networkx as nx
import pystan
import numpy as np

def generate_core_periphery(n, p, p_cc = 0.8, p_cp = 0.4, p_pp = 0.1, seed=100):
    n_c = int(n ** p)
    n_p = n - n_c

    sizes = [n_c, n_p]
    p = [[p_cc, p_cp], [p_cp, p_pp]]

    return nx.generators.community.stochastic_block_model(sizes, p, seed=seed)

def generate_scale_free(n, alpha, seed=100):
    return nx.generators.directed.scale_free_graph(n, alpha, seed=seed)
