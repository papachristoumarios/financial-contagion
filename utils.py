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


def bar_plot(ax, data, colors=None, total_width=0.8, single_width=1, legend=True):
    """Draws a bar plot with multiple bars per data point.

    Parameters
    ----------
    ax : matplotlib.pyplot.axis
        The axis we want to draw our plot on.

    data: dictionary
        A dictionary containing the data we want to plot. Keys are the names of the
        data, the items is a list of the values.

        Example:
        data = {
            "x":[1,2,3],
            "y":[1,2,3],
            "z":[1,2,3],
        }

    colors : array-like, optional
        A list of colors which are used for the bars. If None, the colors
        will be the standard matplotlib color cyle. (default: None)

    total_width : float, optional, default: 0.8
        The width of a bar group. 0.8 means that 80% of the x-axis is covered
        by bars and 20% will be spaces between the bars.

    single_width: float, optional, default: 1
        The relative width of a single bar within a group. 1 means the bars
        will touch eachother within a group, values less than 1 will make
        these bars thinner.

    legend: bool, optional, default: True
        If this is set to true, a legend will be added to the axis.
    """

    # Check if colors where provided, otherwhise use the default color cycle
    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Number of bars per group
    n_bars = len(data)

    # The width of a single bar
    bar_width = total_width / n_bars

    # List containing handles for the drawn bars, used for the legend
    bars = []

    # Iterate over all data
    for i, (name, values) in enumerate(data.items()):
        x_axis = np.arange(1, len(values) + 1)

        # The offset in x direction of that bar
        x_offset = (i - n_bars / 2) * bar_width + bar_width / 2

        # Draw a bar for every value of that type
        for x, y in zip(x_axis, values):
            bar = ax.bar(x + x_offset, y, width=bar_width * single_width, color=colors[i % len(colors)])

        # Add a handle to the last drawn bar, which we'll need for the legend
        bars.append(bar[0])

    # Draw legend if we need
    if legend:
        ax.legend(bars, data.keys())
