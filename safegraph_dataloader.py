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
from stimuli import *
sns.set_theme()

def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--census_data', default='/home/mp2242/safergraph-census/safegraph_open_census_data')
    parser.add_argument('--patterns_data', default='/home/mp2242/safergraph-monthly')
    parser.add_argument('--loans_data', default='/home/mp2242/safergraph-loans')
    parser.add_argument('--n_neighbors', default=5, type=int)
    parser.add_argument('--payroll_data', default='/home/mp2242/payroll')
    parser.add_argument('--expenditures_data', default='/home/mp2242/consumer-expenditures')
    parser.add_argument('--lat', default=42.439499559524485, type=float)
    parser.add_argument('--lon', default=-76.49499020058316, type=float)
    parser.add_argument('--business_assets', default='/home/mp2242/business-assets')
    parser.add_argument('--business_expenses', default='/home/mp2242/business-expenses')

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

def create_multi_graph(patterns, expenditures):

    G = nx.MultiDiGraph()
    expenditures_freq = collections.defaultdict(lambda: collections.defaultdict(float))

    for _, x in patterns.iterrows():
        naics_code_prefix = int(str(x['naics_code'])[:2])
        workers = json.loads(x['workers'])
        home_cbg = json.loads(x['visitor_home_cbgs'])
        for cbg, val in home_cbg.items():
            cbg_workers = int(round(workers['workers'] * val))
            cbg_non_workers = int(val - cbg_workers)
            
            if not expenditures[expenditures['naics_code_prefix'] == naics_code_prefix].empty:
                expenditures_freq[cbg][naics_code_prefix] += 1
            else:
                expenditures_freq[cbg][0] += 1

            [G.add_edge(x['placekey'], int(cbg), weight=x['payroll']) for _ in range(cbg_workers)]


    for _, x in patterns.iterrows():
        naics_code_prefix = int(str(x['naics_code'])[:2])
        workers = json.loads(x['workers'])
        home_cbg = json.loads(x['visitor_home_cbgs'])
        try:
            expenditure_naics = expenditures[expenditures['naics_code_prefix'] == naics_code_prefix]['monthly'].iloc[0]
        except:
            expenditure_naics = expenditures[expenditures['naics_code_prefix'] == 0]['monthly'].iloc[0]
            naics_code_prefix = 0

        for cbg, val in home_cbg.items():
            cbg_workers = int(round(workers['workers'] * val))
            cbg_non_workers = int(val - cbg_workers)

            weight = cbg_non_workers * expenditure_naics / expenditures_freq[cbg][naics_code_prefix] 
            [G.add_edge(int(cbg), x['placekey'], weight=weight) for _ in range(cbg_non_workers)]
    

    nx.set_node_attributes(G, patterns.set_index('placekey')['naics_code'].to_dict(), 'naics_code')

    return G


def create_eisenberg_noe_data(G):
    
    n = len(G)
    P = np.zeros(shape=(n, n))
    L = np.zeros(shape=(n, 1))
    C = np.zeros(shape=(n, 1))
    B = np.zeros(shape=(n, 1))
    A = np.zeros(shape=(n, n))

    G = nx.relabel.convert_node_labels_to_integers(G)

    for u, v, data in G.edges(data=True):
        P[u, v] += data.get('weight', 0)

    for u, data in G.nodes(data=True):
        L[u] += data.get('L', 0)
        C[u] += data.get('assets', 0)
        B[u] += data.get('liabilities', 0)

    P_bar = B + P.sum(-1).reshape(n, 1)

    A = np.copy(P)
    for i in range(n):
        A[i] /= P_bar[i]

    np.savetxt('data/safegraph/safegraph_liability_matrix.csv', P, delimiter=',')
    np.savetxt('data/safegraph/safegraph_external_liabilities.csv', B, delimiter=',')
    np.savetxt('data/safegraph/safegraph_external_assets.csv', C, delimiter=',')
    np.savetxt('data/safegraph/safegraph_proportional_liability_matrix.csv', A, delimiter=',')
    np.savetxt('data/safegraph/safegraph_bailouts.csv', L, delimiter=',')


args = get_argparser()

# Read geographic data relating (lat, lon) to cbg
geographic_data = pd.read_csv(os.path.join(args.census_data, 'metadata/cbg_geographic_data.csv'))
X = np.vstack([geographic_data['latitude'].to_numpy(), geographic_data['longitude'].to_numpy()]).T
knn = NearestNeighbors(n_neighbors=args.n_neighbors, metric='haversine')
knn.fit(X)

_, indices = knn.kneighbors([[args.lat, args.lon]])

# Pick k-nearest cbgs according to haversine distance from a user-given (lat, lon) pair
neighbor_cbgs = geographic_data.iloc[indices[0].tolist()]
neighbor_cbgs = list(neighbor_cbgs['census_block_group'])

# Load all brands (i.e. shops etc.) and their corresponding safegraph ids (a brand may contain multiple POIs)
brands = pd.read_csv(os.path.join(args.patterns_data, 'brand_info_backfill/2020/12/13/04/2018/03/brand_info.csv'))
brands.rename(columns={'safegraph_brand_id' : 'safegraph_brand_ids'}, inplace=True)

brands = brands[['safegraph_brand_ids', 'naics_code']]
brands.set_index('safegraph_brand_ids', inplace=True)

# Load payroll data
payroll = pd.read_csv(os.path.join(args.payroll_data, 'employment_clean.csv'))
payroll.rename(columns={'NAICS' : 'naics_code', 'PAYROLL' : 'payroll'}, inplace=True)

# Join payroll data
brands = brands.join(payroll.set_index('naics_code'), on='naics_code', how='left')

brands.fillna(value={'payroll' : brands['payroll'].mean()}, inplace=True)

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
patterns = patterns[['placekey', 'naics_code', 'payroll', 'poi_cbg', 'visitor_home_cbgs', 'bucketed_dwell_times']]

# Estimate the probability of a device belonging to a worker (resp. a non-worker)
patterns['workers'] = patterns['bucketed_dwell_times'].apply(dwell_workers)


# Size of households
households = pd.read_csv(os.path.join(args.census_data, 'data/cbg_b11.csv'))
households_race = households[['census_block_group', 'B11001Ae1', 'B11001Ie1', 'B11001Be1', 'B11001Ce1', 'B11001De1', 'B11001Ee1', 'B11001Fe1']]
households_race.rename(inplace=True, columns={'B11001Ae1': 'white', 'B11001Ie1': 'hispanic/latino', 'B11001Be1' : 'african_american', 'B11001Ce1' : 'native/indian', 'B11001De1' : 'asian', 'B11001Ee1' : 'hawaian', 'B11001Fe1': 'other'})
households_race['minority'] = households_race['hispanic/latino'] + households_race['african_american'] + households_race['native/indian'] + households_race['asian'] + households_race['hawaian'] + households_race['other']
households_race['p_minority'] = households_race['minority'] / (households_race['minority'] + households_race['white'])
households_race.set_index('census_block_group', inplace=True)
households_dependents = households[['census_block_group', 'B11016e2', 'B11016e3', 'B11016e4', 'B11016e5', 'B11016e6', 'B11016e7', 'B11016e8']]
households_dependents.set_index('census_block_group', inplace=True)


# Household income
households_income = pd.read_csv(os.path.join(args.census_data, 'data/cbg_b19.csv'))
households_income = households_income.loc[:,~households_income.columns.str.contains('m', case=False)]
households_income['annual_income'] = households_income.loc[:, 'B19052e2':'B19060e3'].sum(axis=1) 

# Employment status
households_employment = pd.read_csv(os.path.join(args.census_data, 'data/cbg_b23.csv'))
households_employment['p_employed'] = households_employment['B23025e2'] / households_employment['B23025e2']
households_employment = households_employment[['census_block_group', 'p_employed']]
households_employment.set_index('census_block_group', inplace=True)

# Business Bailouts
loans = pd.read_csv(os.path.join(args.loans_data, 'loans.csv'))
loans.rename(inplace=True, columns={'placekey_poi_match' : 'placekey'})
loans.set_index('placekey', inplace=True)
loans = loans[loans.index.isin(patterns['placekey'])]

# Consumer Expenditures
expenditures = pd.read_csv(os.path.join(args.expenditures_data, 'expenditures.csv'))

# Business Assets
business_assets = pd.read_csv(os.path.join(args.business_assets, 'assets.csv'))

# Business Expenses
business_expenses = pd.read_csv(os.path.join(args.business_expenses, 'expenses_processed.csv'))

# Create multi-graph topology
G = create_multi_graph(patterns, expenditures)

# Create node attributes
loans_amount = loans['amount'] / 12
loans_amount = loans['amount'].to_dict()

for x in G.nodes():
    if isinstance(x, str):
        if x in loans_amount:
            loans_amount[x] = G.out_degree(x) * loans_amount[x]

nx.set_node_attributes(G, loans_amount, 'L')

loans_race = loans['race'].to_dict()
nx.set_node_attributes(G, loans_race, 'race')

households_dependents = households_dependents[households_dependents.index.isin(list(G.nodes()))]
households_dependents_avg = collections.defaultdict(float)
num_households = collections.defaultdict(int)
households_bailouts = collections.defaultdict(float)
households_external_assets = collections.defaultdict(float)
business_external_assets = collections.defaultdict(float)
business_external_liabilities = collections.defaultdict(float)

annual_expenditures = 63000
monthly_expenditures = annual_expenditures / 12
households_external_liabilities = collections.defaultdict(float)

for idx, dist in households_dependents.iterrows():
    if np.sum(list(dist)) > 0:
        try:
            p_employed = households_employment[households_employment['census_block_group'] == idx]['p_employed'].iloc[0]
        except:
            p_employed = 1
        households_dependents_avg[idx] = np.average(np.arange(1, 1 + len(dist)), weights=list(dist))
        num_households[idx] = int(np.ceil((G.in_degree(idx) / p_employed + G.out_degree(idx)) / households_dependents_avg[idx]))
    else:
        households_dependents_avg[idx] = 1
        num_households[idx] = int(np.ceil(G.in_degree(idx) + G.out_degree(idx)))

    annual_income = households_income[households_income['census_block_group'] == idx]['annual_income'].iloc[0]
    monthly_income = annual_income / 12
    households_bailouts[idx] = num_households[idx] * stimulus_check(annual_income, households_dependents_avg[idx])

    total_weight = 0
    for v in G.predecessors(idx):
        for key, val in G.get_edge_data(v, idx).items():
            total_weight += G[v][idx][key].get('weight', 0)

    households_external_assets[idx] = max(0, num_households[idx] * monthly_income - total_weight)


nx.set_node_attributes(G, households_dependents_avg, 'avg_dependents')
nx.set_node_attributes(G, num_households, 'num_people')
nx.set_node_attributes(G, households_race['p_minority'].to_dict(), 'p_minority')
nx.set_node_attributes(G, households_bailouts, 'L')
nx.set_node_attributes(G, households_external_assets, 'assets')

for x in G.nodes():
    if isinstance(x, str):
        for cbg in G.predecessors(x):
            for key, val in G.get_edge_data(cbg, x).items():
                G[cbg][x][key]['weight'] = G[cbg][x][key]['weight'] / households_dependents_avg[cbg]

for x, data in G.nodes(data=True):

    if isinstance(x, int):
        total_weight = 0
        for y in G.successors(x):
            for key, val in G.get_edge_data(x, y).items():
                total_weight += G[x][y][key]['weight']
        households_external_liabilities[x] = max(100, num_households[x] * monthly_expenditures - total_weight)
    
    elif isinstance(x, str):
        total_weight = 0
        for y in G.predecessors(x):
            for key, val in G.get_edge_data(y, x).items():
                total_weight += G[y][x][key]['weight']
        naics_code_prefix = int(str(data['naics_code'])[:2])
        assets = business_assets[business_assets['naics_code_prefix'] == naics_code_prefix]
        if assets.empty:
            assets = business_assets['monthly_revenue'].mean()
        else:
            assets = assets['monthly_revenue'].iloc[0]

        business_external_assets[x] = max(0, assets - total_weight)

        total_weight = 0
        for y in G.successors(x):
            for key, val in G.get_edge_data(x, y).items():
                total_weight += G[x][y][key]['weight']

        expenses = business_expenses[business_expenses['naics_code'] == data['naics_code']]
        if expenses.empty:
            expenses = G.out_degree(x) * expenses['monthly_expenses_per_employee'].mean() 
        else:
            expenses = G.out_degree(x) * expenses['monthly_expenses_per_employee'].iloc[0]

        business_external_liabilities[x] = max(100, expenses - total_weight)

nx.set_node_attributes(G, households_external_liabilities, 'liabilities')
nx.set_node_attributes(G, business_external_assets, 'assets')
nx.set_node_attributes(G, business_external_liabilities, 'liabilities')

create_eisenberg_noe_data(G)
nx.write_gpickle(G, "safegraph.gpickle")

