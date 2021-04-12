import pandas as pd
import networkx as nx
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pickle
import glob
from generator import *

sns.set_theme()

def load_venmo_data_and_extract_components(export_wcc=True, min_size=100):
    df = pd.read_csv('data/venmo.csv', low_memory=True)

    G = nx.from_pandas_edgelist(df, 'payment.actor.username', 'payment.target.user.username', create_using=nx.DiGraph)

    pickle.dump(G, open('data/venmo_G.pickle', 'wb'))
    wccs = []

    if export_wcc:
        wccs_gen = nx.weakly_connected_components(G)

        for i, wcc in enumerate(wccs_gen):
            if len(wcc) >= min_size:
                pickle.dump(G.subgraph(wcc).copy(), open('data/venmo_wcc_{}.pickle'.format(i), 'wb'))
                wccs.append(wcc)

    return G, wccs

def load_venmo_dataset():
    for filename in glob.glob('data/venmo_wcc_*'):
        G = nx.nx.relabel.convert_node_labels_to_integers(nx.read_gpickle(filename))
        n = len(G)

        distribution = 'pareto'
        alpha = 0.14

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

        external_assets = np.array([G.degree(u) for u in G]) * distributions[distribution]((n, 1))
        external_liabilities = alpha * external_assets

        P_bar = internal_liabilities + external_liabilities

        A = np.copy(liabilities)
        for i in range(liabilities.shape[0]):
            A[i] /= P_bar[i]

        wealth = external_assets + internal_assets - external_liabilities - internal_liabilities

        return A, P_bar, liabilities, adj, internal_assets, internal_liabilities, outdegree, indegree, external_assets, external_liabilities, wealth, G
