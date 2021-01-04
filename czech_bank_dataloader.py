import pandas as pd
import networkx as nx
import swifter
import matplotlib.pyplot as plt
import numpy as np
import collections
import pickle

filename='data/czech-bank/data/trans.csv'

transactions = pd.read_csv(filename).dropna(subset=['account'])
G = nx.DiGraph()
node2idx = {}
counter = 0

def create_edge(transaction):
    global G, node2idx, counter
    if not node2idx.get(transaction['account_id'], None):
        node2idx[transaction['account_id']] = counter
        counter += 1
    
    u = node2idx[transaction['account_id']]
    if not node2idx.get(transaction['account'], None):
        node2idx[transaction['account']] = counter
        counter += 1

    v = node2idx[transaction['account']]

    amount = float(transaction['amount'])
    G.add_edge(u, v, weight=amount)

transactions.swifter.apply(create_edge, axis=1)

with open('cb.pickle', 'wb+') as f:
    pickle.dump(G, f)

def analyze_dataset(G):

    outdeg = [G.out_degree(v) for v in G]
    indeg = [G.in_degree(v) for v in G]

    outdegs, outdegs_counts = np.unique(outdeg, return_counts=True)
    indegs, indegs_counts = np.unique(indeg, return_counts=True)

    outdegs_counts = outdegs_counts / outdegs_counts.sum()
    indegs_counts = indegs_counts / indegs_counts.sum()

    plt.figure()
    plt.plot(outdegs, outdegs_counts)
    plt.title('Outdegree Distribution')
    plt.xlabel('Outdegree')
    plt.ylabel('Frequency')

    plt.figure()
    plt.plot(indegs, indegs_counts)
    plt.title('Indegree Distribution')
    plt.xlabel('Indegree')
    plt.ylabel('Frequency')

    plt.show()

