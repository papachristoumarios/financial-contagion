import numpy as np
import seaborn as sns
import scipy.stats
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import warnings
from sklearn.linear_model import LinearRegression
warnings.filterwarnings("ignore")

sns.set_theme()

def load_german_banks_dataset():
    data = pd.read_csv('data/german_banks_data_aggregate.csv')
    liabilities = np.genfromtxt(
        'data/german_banks_data_liabilities_matrix.csv', delimiter=',', dtype=np.float64)
    liabilities = liabilities[1:, 1:]
    adj = (liabilities > 0).astype(np.float64)


    outdegree = adj.sum(0)
    indegree = adj.sum(-1)
    G = nx.from_numpy_matrix(adj, create_using=nx.DiGraph)

    p = adj.sum() / (len(data)**2 - len(data))

    internal_assets = liabilities.sum(-1).reshape((len(G), 1))
    internal_liabilities = liabilities.sum(0).reshape((len(G), 1))

    external_assets = data['Ext. Asset'].to_numpy().reshape((len(G), 1))
    external_liabilities = data['Ext. Liability'].to_numpy().reshape((len(G), 1))
    P_bar = internal_liabilities + external_liabilities

    assets_liabs_model = LinearRegression()
    assets_liabs_model.fit(external_assets, external_liabilities)

    A = np.copy(liabilities)
    for i in range(liabilities.shape[0]):
        A[i] /= P_bar[i]

    wealth = external_assets + internal_assets - external_liabilities - internal_liabilities


    return data, A, P_bar, liabilities, adj, internal_assets, internal_liabilities, outdegree, indegree, p, external_assets, external_liabilities, wealth, G

def pareto_fit_helper(data):
    params = scipy.stats.pareto.fit(data)
    return 'Pareto Fit: b = {}, loc = {}, scale = {}'.format(round(params[0], 1), round(params[1], 1), round(params[2], 1))

def plot_german_banks_dataset(data, A, P_bar, liabilities, adj, internal_assets, internal_liabilities, outdegree, indegree, p, external_assets, external_liabilities, wealth, G):

    print('Number of edges', outdegree.sum())
    print('ER estimate', p)
    print('Mean outdegree', outdegree.mean())

    plt.figure(figsize=(10, 10))
    positive_liabilities = liabilities[np.where(liabilities > 0)].flatten()
    sns.distplot(positive_liabilities, kde=False, fit=scipy.stats.pareto, label='Liabilities distribution {}'.format(pareto_fit_helper(positive_liabilities)))
    plt.xlim(0, positive_liabilities.max())
    plt.title('Liabilities Distribution')
    plt.legend()
    plt.savefig('german_banks_liabilities_distribution.png')


    plt.figure(figsize=(10, 10))
    jointplot = sns.jointplot(data=data, x="Ext. Liability", y="Ext. Asset", kind='reg')
    pearson_corr = scipy.stats.pearsonr(data['Ext. Asset'], data['Ext. Liability'])[0]
    coeffs = np.round(np.polyfit(external_liabilities.flatten(), external_assets.flatten(), deg=1), 2)
    jointplot.fig.suptitle('$R^2 = {}$, $a = {}$, $b = {}$'.format(round(pearson_corr, 3), round(coeffs[0], 2), round(coeffs[1], 2)), fontsize=12)
    jointplot.fig.tight_layout()
    jointplot.fig.subplots_adjust(top=0.95)
    plt.savefig('german_banks_assets_liabilities_distribution.png')

    plt.figure(figsize=(10, 10))
    sns.distplot(data['Equity'], kde=False, fit=scipy.stats.pareto, label='Wealth {}'.format(pareto_fit_helper(data['Equity'])))
    plt.xlim(0, data['Equity'].max())
    plt.title('Equity Distribution')
    plt.legend()
    plt.savefig('german_banks_equity_distribution.png')

    plt.figure(figsize=(10, 10))
    sns.distplot(external_assets, kde=False, fit=scipy.stats.pareto, label='External Assets {}'.format(pareto_fit_helper(external_assets)))
    sns.distplot(external_liabilities, kde=False,
                 fit=scipy.stats.pareto, label='External Liabilities {}'.format(pareto_fit_helper(external_liabilities)))
    plt.xlim(0, external_assets.max())
    plt.legend()
    plt.savefig('german_bank_external_assets_liabilities_distribution.png')

    plt.figure(figsize=(10, 10))
    ax = plt.gca()
    ax.set_title('Financial Network with $p = {}$'.format(round(p, 2)))
    pos = nx.shell_layout(G)
    nx.draw(G, pos, node_color=np.log(data['Equity']), node_size=800, cmap=plt.cm.Blues)
    ax.axis('off')
    plt.savefig('german_banks_financial_network.png')

    plt.show()

if __name__ == '__main__':
    data, A, P_bar, liabilities, adj, internal_assets, internal_liabilities, outdegree, indegree, p, external_assets, external_liabilities, wealth, G = load_german_banks_dataset()
    plot_german_banks_dataset(data, A, P_bar, liabilities, adj, internal_assets, internal_liabilities, outdegree, indegree, p, external_assets, external_liabilities, wealth, G)
