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
        G = pickle.load(open(filename, 'rb'))
        yield generate_random_data(random_graph='ER', distribution='pareto', alpha=0.14, G=G)
