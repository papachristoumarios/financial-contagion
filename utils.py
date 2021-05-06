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

def disparity(x, p_minority=None, A=None):
    n = len(x)
    d = np.zeros(shape=(n, n))

    for i in range(n):
        for j in range(n):
            if p_minority is None and A is None:
                d[i, j] = np.abs(x[i] - x[j])
            elif not(p_minority is None):
                d[i, j] = p_minority[i, 0] * (1 - p_minority[j, 0]) * np.abs(x[i] - x[j])
            elif not(A is None):
                d[i, j] = A[i, j] *  np.abs(x[i] - x[j])
    
    return d

def gini(x, p_minority=None, A=None):
    d = disparity(x, p_minority, A)
    n = len(x)
  
    if p_minority is None and A is None:
        return d.sum() / (2 * n * np.sum(x))
    elif not(p_minority is None):
        return (d.sum()) / (np.dot(p_minority.flatten(), x.flatten()) * np.sum(1 - p_minority))
    elif not(A is None):
        return (d.sum()) / (np.sum(x.flatten() * A.sum(-1)))


def create_set_helper(arr, k, b, L):
    if isinstance(L, np.ndarray):
        total = 0
        result = []
        for v, _ in arr:
            if total + L[v, 0] > k * b:
                return set(result)
            else:
                total += L[v, 0]
                result.append(v)
        return set(result)
    else:
        return set([x[0] for x in arr[:k]])
