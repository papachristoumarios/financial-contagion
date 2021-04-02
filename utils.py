import pandas as pd
import networkx as nx
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pickle
import glob

sns.set_theme()

def degree_plot(G, name='indegree', degree_fcn=lambda G, u: G.in_degree(u)):

    degrees = np.array([degree_fcn(G, v) for v in G])
    degrees_values, degrees_counts = np.unique(degrees, return_counts=True)
    degrees_counts = np.log(degrees_counts / degrees_counts.sum())
    degrees_values = np.log(1 + degrees_values)

    p = np.polyfit(degrees_values, degrees_counts, deg=1)
    pearsonr = np.corrcoef(degrees_values, degrees_counts)[0, 1]

    plt.figure(figsize=(10, 10))
    plt.plot(degrees_values, degrees_counts, linewidth=0, marker='x', color='b', label='Empirical log Frequency, $y \propto x^{{{}}}, \; R^2 = {}$'.format(round(p[0], 1),  round(pearsonr, 2)))

    plt.xlabel('log {}'.format(name))
    plt.ylabel('log Frequency')
    plt.legend()

    plt.savefig(name + '.png')
