import pandas as pd
import collections
import json
import math
import random
import numpy as np
import networkx as nx
from sklearn.neighbors import NearestNeighbors
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import sys
sns.set_theme()

def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--census_data', default='/home/mp2242/safergraph-census/safegraph_open_census_data')
    parser.add_argument('--patterns_data', default='/home/mp2242/safergraph-monthly')
    parser.add_argument('--loans_data', default='/home/mp2242/safergraph-loans')
    parser.add_argument('--n_neighbors', default=5, type=int)

    return parser.parse_args()

def degree_plot(G, names=['indegree', 'outdegree'], markers=['x', 'o'], degree_fcns=[lambda G, u: G.in_degree(u), lambda G, u: G.out_degree(u)]):

    for name, marker, degree_fcn in zip(names, markers, degree_fcns):
        degrees = np.array([degree_fcn(G, v) for v in G])
        degrees_values, degrees_counts = np.unique(degrees, return_counts=True)
        degrees_counts = np.log(degrees_counts / degrees_counts.sum())
        degrees_values = np.log(degrees_values)

        p = np.polyfit(degrees_values, degrees_counts, deg=1)
        pearsonr = np.corrcoef(degrees_values, degrees_counts)[0, 1]

        plt.figure(figsize=(10, 10))
        plt.plot(degrees_values, degrees_counts, linewidth=0, marker=marker, color='b', label='Empirical log Frequency, $y \propto x^{{{}}}, \; R^2 = {}$'.format(round(p[0], 1),  round(pearsonr, 2)))

    plt.xlabel('log Degree')
    plt.ylabel('log Frequency')
    plt.legend()

    plt.savefig('degree.png')

def dwell_workers(u):
    result = collections.defaultdict(float)

    u_p = json.loads(u)
    for key, val in u_p.items():
        if key == '>240':
            result['workers'] += float(val)
        else:
            result['non-workers'] += float(val)

    result['workers'] /= (result['workers'] + result['non-workers'])
    result['non-workers'] /= (result['workers'] + result['non-workers'])

    return json.dumps(result)

def create_multi_graph(patterns, households_dependents, households_race):

    G = nx.MultiDiGraph()
    multi_edges = collections.defaultdict(int)
    dist_dependents = collections.defaultdict(int)
    dist_minority = collections.defaultdict(float)
    households_race['p_minority'] = households_race['minority'] / (households_race['minority'] + households_race['white'])

    for _, x in patterns.iterrows():
        workers = json.loads(x['workers'])
        home_cbg = json.loads(x['visitor_home_cbgs'])

        for cbg, val in home_cbg.items():
            cbg_workers = int(round(workers['workers'] * val))
            cbg_non_workers = int(val - cbg_workers)
            if G.has_node(cbg):
                G.add_node(cbg, dist_minority = households_race[housholds_race['census_block_group'] == cbg]['p_minority'].iloc[0], dist_dependents = list(households_dependents[households_dependents['census_block_group'] == cbg][['B11016e2', 'B11016e3', 'B11016e4', 'B11016e5', 'B11016e6', 'B11016e7', 'B11016e8']].iloc[0]))

            [G.add_edge(int(cbg), x['placekey']) for _ in range(cbg_non_workers)]
            [G.add_edge(x['placekey'], int(cbg)) for _ in range(cbg_workers)]

    return G

def create_graph(patterns, households_dependents, households_race):

    multi_edges = collections.defaultdict(int)
    num_dependents = collections.defaultdict(int)
    is_minority = collections.defaultdict(int)

    for _, x in patterns.iterrows():
        workers = json.loads(x['workers'])
        home_cbg = json.loads(x['visitor_home_cbgs'])

        for cbg, val in home_cbg.items():
            cbg_workers = int(round(workers['workers'] * val))
            cbg_non_workers = int(val - cbg_workers)

            if cbg_non_workers > 0:
                multi_edges[int(cbg), x['placekey']] += cbg_non_workers
            if cbg_workers > 0:
                multi_edges[x['placekey'], int(cbg)] += cbg_workers


    G = nx.DiGraph()
    idx_2_node_from = {}
    idx_2_node_to = {}
    counter_from = 0
    counter_to = 0
    households_race['p_minority'] = households_race['minority'] / (households_race['minority'] + households_race['white'])
    multi_edges_keys = list(multi_edges.keys())


    for (u, v) in multi_edges_keys:

        if isinstance(u, int) and isinstance(v, str):
            idx = 0
            weights = list(households_dependents[households_dependents['census_block_group'] == u][['B11016e2', 'B11016e3', 'B11016e4', 'B11016e5', 'B11016e6', 'B11016e7', 'B11016e8']].iloc[0])

            p_minority = households_race[households_race['census_block_group'] == u]['p_minority'].iloc[0]
            for j in range(multi_edges[u, v]):
                G.add_edge('{}_{}'.format(u, idx), v)
                idx_2_node_from[counter_from] = '{}_{}'.format(u, idx)
                counter_from += 1
                num_dependents['{}_{}'.format(u, idx)] = random.choices([1, 2, 3, 4, 5, 6, 7], weights=weights, k=1)[0]
                is_minority['{}_{}'.format(u, idx)] = np.random.uniform() <= p_minority
                idx += 1
            # Dependent or independent?
            for j in range(multi_edges[v, u]):
                G.add_edge(v, '{}_{}'.format(u, idx))
                idx_2_node_to[counter_to] = '{}_{}'.format(u, idx)
                counter_to += 1
                num_dependents['{}_{}'.format(u, idx)] = random.choices([1, 2, 3, 4, 5, 6, 7], weights=weights, k=1)[0]
                is_minority['{}_{}'.format(u, idx)] = np.random.uniform() <= p_minority
                idx += 1


    H = nx.DiGraph(nx.scale_free_graph(counter_from + counter_to))

    for (u, v) in H.edges():
        if u < counter_from:
            uu = idx_2_node_from[u]
        else:
            uu = idx_2_node_to[u - counter_from]

        if v < counter_from:
            vv = idx_2_node_from[v]
        else:
            vv = idx_2_node_to[v - counter_from]

        G.add_edge(uu, vv)

    return G, num_dependents

args = get_argparser()

# Read geographic data relating (lat, lon) to cbg
geographic_data = pd.read_csv(os.path.join(args.census_data, 'metadata/cbg_geographic_data.csv'))
X = np.vstack([geographic_data['latitude'].to_numpy(), geographic_data['longitude'].to_numpy()]).T
knn = NearestNeighbors(n_neighbors=args.n_neighbors, metric='haversine')
knn.fit(X)

#_, indices = knn.kneighbors([[42.439499559524485, -76.49499020058316]])

_, indices = knn.kneighbors([[40.834998005599736, -73.9917002004307]])

# Pick k-nearest cbgs according to haversine distance from a user-given (lat, lon) pair
neighbor_cbgs = geographic_data.iloc[indices[0].tolist()]
neighbor_cbgs = list(neighbor_cbgs['census_block_group'])

# Load all brands (i.e. shops etc.) and their corresponding safegraph ids (a brand may contain multiple POIs)
brands = pd.read_csv(os.path.join(args.patterns_data, 'brand_info_backfill/2020/12/13/04/2018/03/brand_info.csv'))
brands.rename(columns={'safegraph_brand_id' : 'safegraph_brand_ids'}, inplace=True)

brands = brands[['safegraph_brand_ids', 'top_category']]
brands.set_index('safegraph_brand_ids', inplace=True)

# Monthly patterns
patterns_helper = []
for i in range(1, 5):
    patterns = pd.read_csv(os.path.join(args.patterns_data, 'patterns/2020/12/04/04/patterns-part{}.csv'.format(i)))
    patterns.dropna(subset=['safegraph_brand_ids'], inplace=True)
    patterns.dropna(subset=['poi_cbg'], inplace=True)
    # Get all stores located within the requested cbgs
    # This query should yield the businesses of the nearby areas
    patterns = patterns[patterns['poi_cbg'].isin(neighbor_cbgs)]
    patterns_helper.append(patterns)

patterns = pd.concat(patterns_helper, axis=0, ignore_index=True)


# Get the categories of POIs through the brands dataframe
patterns = patterns.join(brands, on='safegraph_brand_ids')

# Drop unneeded columns
patterns = patterns[['placekey', 'top_category', 'poi_cbg', 'visitor_home_cbgs', 'bucketed_dwell_times']]

# Estimate the probability of a device belonging to a worker (resp. a non-worker)
patterns['workers'] = patterns['bucketed_dwell_times'].apply(dwell_workers)


# Size of households
households = pd.read_csv(os.path.join(args.census_data, 'data/cbg_b11.csv'))
households_race = households[['census_block_group', 'B11001Ae1', 'B11001Ie1', 'B11001Be1', 'B11001Ce1', 'B11001De1', 'B11001Ee1', 'B11001Fe1']]
households_race.rename(inplace=True, columns={'B11001Ae1': 'white', 'B11001Ie1': 'hispanic/latino', 'B11001Be1' : 'african_american', 'B11001Ce1' : 'native/indian', 'B11001De1' : 'asian', 'B11001Ee1' : 'hawaian', 'B11001Fe1': 'other'})
households_race['minority'] = households_race['hispanic/latino'] + households_race['african_american'] + households_race['native/indian'] + households_race['asian'] + households_race['hawaian'] + households_race['other']
households_race.set_index('census_block_group')
households_dependents = households[['census_block_group', 'B11016e2', 'B11016e3', 'B11016e4', 'B11016e5', 'B11016e6', 'B11016e7', 'B11016e8']]
households.set_index('census_block_group')

G = create_multi_graph(patterns, households_dependents, households_race)
