import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
import seaborn as sns
import numpy as np
import networkx as nx

sns.set_theme()

def preprocess_glasserman_young_data():
    data = pd.read_csv('data/glasserman_young_data.csv')

    data.sort_values('w_i', ascending=False, inplace=True)
    data['cum_w_i'] = data['w_i'].cumsum()

    # Assets from other nodes in the network
    internal_assets = data['Assets'] - data['c_i']

    internal_assets.to_csv('data/glasserman_young_data_internal_assets.csv')

    # Construct internal liabilities
    random_petrubations = np.random.normal(0, 100, size=len(internal_assets))
    peturbed_internal_assets = internal_assets + random_petrubations
    peturbed_internal_assets_sum = peturbed_internal_assets.sum()
    internal_assets_sum = internal_assets.sum()

    internal_liabilities = peturbed_internal_assets * internal_assets_sum / peturbed_internal_assets_sum
    internal_liabilities = internal_liabilities.round(1)
    internal_liabilities.iloc[-1] = internal_assets_sum - internal_liabilities[:-1].sum()

    internal_liabilities.to_csv('data/glasserman_young_data_internal_liabilities.csv')

    # Calculate external liabilities
    external_liabilities = internal_liabilities / data['beta_i'] - internal_liabilities
    external_liabilities.to_csv('data/glasserman_young_data_external_liabilities.csv')

    # External assets
    external_assets = data['c_i']
    external_assets.to_csv('data/glasserman_young_data_external_assets.csv')

    # Params are shape, loc, scale
    wealth_params = scipy.stats.pareto.fit(data['w_i'])
    print('Shape: {}, Loc: {}, Scale: {}'.format(*wealth_params))

    plt.figure()
    sns.jointplot(external_assets, external_liabilities, kind='reg')

    plt.figure()
    sns.distplot(data['w_i'], kde=False, fit=scipy.stats.pareto, label='Wealth')
    plt.xlim(0, data['w_i'].max())
    plt.legend()

    plt.figure()
    sns.distplot(external_assets, kde=False, fit=scipy.stats.pareto, label='External Assets')
    sns.distplot(external_liabilities, kde=False, fit=scipy.stats.pareto, label='External Liabilities')
    plt.xlim(0, external_assets.max())
    plt.legend()

    plt.figure()
    sns.distplot(internal_assets, kde=False, fit=scipy.stats.pareto, label='Internal Assets')
    sns.distplot(internal_liabilities, kde=False, fit=scipy.stats.pareto, label='Internal Liabilities (after pertubation)')

    plt.xlim(0, max(internal_assets.max(), internal_liabilities.max()))
    plt.legend()

    plt.figure()
    plt.plot(np.linspace(0, 1, len(data)), data['cum_w_i'].to_numpy())
    plt.ylabel('Cummulative Wealth')
    plt.xlabel('Percentile')
    plt.show()

def load_glasserman_young_dataset():
    data = pd.read_csv('data/glasserman_young_data.csv')
    external_liabilities = pd.read_csv('data/glasserman_young_data_external_liabilities.csv').to_numpy()[:, -1]
    external_assets = pd.read_csv('data/glasserman_young_data_external_assets.csv').to_numpy()[:, -1]

    for i in range(1, 51):
        liabilities = np.genfromtxt('data/glasserman_young_data_liabilities_matrix_{}.csv'.format(i), delimiter=' ', dtype=np.float64)
        adj = (liabilities > 0).astype(np.float64)

        outdegree = adj.sum(0)
        indegree = adj.sum(-1)
        G = nx.from_numpy_matrix(adj, create_using=nx.DiGraph)
        internal_assets = liabilities.sum(-1).reshape((len(G), 1))
        internal_liabilities = liabilities.sum(0).reshape((len(G), 1))
        external_assets = external_assets.reshape((len(G), 1))
        external_liabilities = external_liabilities.reshape((len(G), 1))

        wealth = external_assets + internal_assets - external_liabilities - internal_liabilities
        P_bar = internal_liabilities + external_liabilities

        p = adj.sum() / (len(data)**2 - len(data))
        A = np.copy(liabilities)
        for i in range(liabilities.shape[0]):
            A[i] /= P_bar[i]

        yield data, A, P_bar, liabilities, adj, internal_assets, internal_liabilities, outdegree, indegree, p, external_assets, external_liabilities, wealth, G