from eisenberg_noe import *
from german_banks_dataloader import *
from eba_dataloader import *
from venmo_dataloader import *
from safegraph_dataloader import *
from generator import *
from metrics import *
from utils import *
import seaborn as sns
import argparse
import matplotlib.cm as cm
import random
import copy
from networkx.drawing.nx_agraph import to_agraph

def create_set_helper(arr, k, b, L):
    if isinstance(L, np.ndarray):
        total = 0
        result = []
        for i in range(len(arr)):
            if total + L[i, 0] > k * b:
                break
            else:
                total += L[i, 0]
                result.append(arr[i][0])
        return set(result)
    else:
        return set([x[0] for x in arr[:k]])


def get_argparser():
    parser = argparse.ArgumentParser(
        description='Discrete stimulus allocation algorithm to maximize SoP, SoT, or SoIT objectives.')
    parser.add_argument('--obj', type=str, default='SoP', help='Type of objective (SoP, SoT, SoIT)',
                        choices=['SoP', 'SoT', 'SoIP', 'FS', 'AS', 'MD'])
    parser.add_argument('--num_iters', type=int, default=-1,
                        help='Number of iterations for Monte Carlo approximation')
    parser.add_argument('-L', type=str, default='1000000', help='Stimulus value (enter integer for same-everywhere stimulus or enter filename location for different stimuli)')
    parser.add_argument('--dataset', type=str, default='german_banks',
                        help='Dataset to run simulation on', choices=['german_banks', 'eba', 'venmo', 'safegraph', 'random'])
    parser.add_argument('--random_graph', type=str, default='ER', choices=['ER', 'CP', 'SF'],
                        help='Random graph model for artificial data')
    parser.add_argument('-k', type=int, default=-1,
                        help='Number of people to bailout through simulation')
    parser.add_argument('--resource_augmentation', action='store_true',
                        help='Apply resource augmentation to the randomized rounding LP algorithm')
    parser.add_argument('--seed', type=int, default=42, help='Random seed to be used')
    parser.add_argument('--shocks_distribution', type=str, default='beta')
    parser.add_argument('--assets_distribution', type=str, default='exponential')
    parser.add_argument('--workers', type=int, default=1, help='Number of workers')
    parser.add_argument('--num_std', type=float, default=0.5,
                        help='Number of stds to plot in the uncertainty plot')
    parser.add_argument('--untruncated_violin', action='store_true',
                        help='Untruncated violin plots')
    parser.add_argument('--eps', type=float, default=1e-4,
                        help='Parameter in the transformation of the increasing objective to a strictly increasing objective')
    parser.add_argument('-b', type=int, default=100000, help='Rate of increase of availbale budget (if different bailouts are selected)')

    parser.add_argument('--layout', default='spectral')

    return parser


def graph_plot(k, results, G, P, outfile, obj, L, layout, show=False):
    colors = iter(cm.rainbow(np.linspace(0, 1, 1 + len(results))))

    edges = list(G.edges())
    weights = [int(np.log10(P[u, v])) for u, v in edges]

    for data in results:
        result = data[0]
        title = data[-1]

        if len(data) == 3:
            S = data[1]
            if layout == 'spectral':
                pos = nx.layout.spectral_layout(G)
            elif layout == 'spring':
                pos = nx.spring_layout(G, iterations=50)
            elif layout == 'bipartite':
                top = []
                for H in nx.weakly_connected_components(G):
                    top.extend(nx.bipartite.sets(G.subgraph(H))[0])
                pos = nx.bipartite_layout(G, top)
            elif layout == 'circular':
                pos = nx.circular_layout(G)
            elif layout == 'kamada_kawai':
                pos = nx.kamada_kawai_layout(G)
            elif layout == 'shell':
                shells = [list(S), list(V - S)]
                pos = nx.shell_layout(G, shells)
            elif layout == 'spiral':
                pos = nx.spiral_layout(G)
            elif layout == 'random':
                pos = nx.random_layout(G)

            fig = plt.figure(figsize=(10, 10), frameon=False)
            ax = fig.add_axes([0, 0, 1, 1])
            ax.axis('off')

            nx.draw_networkx_nodes(G, pos, node_size=20, nodelist=list(S), node_color='k', alpha=0.5)
            nx.draw_networkx_nodes(G, pos, node_size=20, nodelist=list(V - S), node_color="#3182bd", linewidths=1)
            nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=weights, width=1, edge_cmap=plt.cm.viridis)

            plt.savefig('graph_{}_{}'.format(title.replace(' ', '_'),  outfile))
        else:
            stimuli_mean = result[-2]
            if layout == 'spectral':
                pos = nx.layout.spectral_layout(G)
            elif layout == 'spring':
                pos = nx.spring_layout(G, iterations=50)
            elif layout == 'bipartite':
                top = []
                for H in nx.weakly_connected_components(G):
                    top.extend(nx.bipartite.sets(G.subgraph(H))[0])
                pos = nx.bipartite_layout(G, top)
            elif layout == 'circular':
                pos = nx.circular_layout(G)
            elif layout == 'kamada_kawai':
                pos = nx.kamada_kawai_layout(G)
            elif layout == 'shell':
                pos = nx.shell_layout(G)
            elif layout == 'spiral':
                pos = nx.spiral_layout(G)
            elif layout == 'random':
                pos = nx.random_layout(G)

            fig = plt.figure(figsize=(10, 10), frameon=False)
            ax = fig.add_axes([0, 0, 1, 1])
            ax.axis('off')

            nx.draw_networkx_nodes(G, pos, node_size=20, nodelist=list(G.nodes()), node_color=stimuli_mean.tolist(), cmap=plt.cm.viridis)
            nx.draw_networkx_edges(G, pos, node_size=20, edgelist=edges, edge_color=weights, width=1, edge_cmap=plt.cm.viridis)

            plt.savefig('graph_{}_{}'.format(title.replace(' ', '_'),  outfile))


    if show:
        plt.show()

if __name__ == '__main__':
    args = get_argparser().parse_args()
    seed = args.seed
    workers = args.workers
    sns.set_theme()
    LARGE_SIZE = 16
    plt.rc('axes', labelsize=LARGE_SIZE)
    plt.rc('axes', titlesize=LARGE_SIZE)

    np.random.seed(seed)
    random.seed(seed)

    if args.dataset == 'german_banks':
        data, A, P_bar, P, adj, _, _, _, _, _, C, B, w, G = load_german_banks_dataset()
    elif args.dataset == 'eba':
        data, A, P_bar, P, adj, _, _, _, _, _, C, B, w, G = next(load_eba_dataset())
    elif args.dataset == 'venmo':
        A, P_bar, P, adj, _, _, _, _, C, B, w, G = load_venmo_dataset()
    elif args.dataset == 'safegraph':
        A, P_bar, P, C, B, L, w, G = load_safegraph_dataset()
    elif args.dataset == 'random':
        A, P_bar, P, adj, _, _, _, _, C, B, w, G = generate_random_data(
            args.seed, args.random_graph, args.assets_distribution)

    beta = B / P_bar

    if args.obj == 'SoP':
        v = np.ones(shape=(len(G), 1))
    elif args.obj == 'SoT':
        v = 1 - beta
    elif args.obj == 'SoIP':
        v = beta
    elif args.obj == 'FS':
        v = 1 / P_bar

    n = len(G)

    if args.num_iters <= 0:
        eps = 1
        num_iters = int(n**2 / (eps**2) * np.log(n))
    else:
        num_iters = args.num_iters

    if args.dataset != 'safegraph':
        try:
            L = float(args.L)
            if L <= 0:
                raise Exception('Please use a positive amount for L')
            else:
                L = min(args.L, C.max())
        except ValueError:
            L = np.genfromtxt(args.L, delimiter=',', dtype=np.float64)

    b = args.b
    k = args.k

    V = set(list(G.nodes()))
    S_greedy = set()
    eps = args.eps

    pageranks = nx.algorithms.pagerank(G)
    pageranks = list(pageranks.items())
    pageranks = list(sorted(pageranks, key=lambda x: -x[-1]))

    centralities = nx.algorithms.centrality.betweenness_centrality(G)
    centralities = list(centralities.items())
    centralities = list(sorted(centralities, key=lambda x: -x[-1]))

    out_degrees = list(sorted([(v, G.out_degree(v)) for v in G], key=lambda x: -x[-1]))

    wealths = list(sorted([(v, w[v, 0]) for v in G], key=lambda x: x[-1]))

    random_order = [(v, 0) for v in G]
    random.shuffle(random_order)

    if args.resource_augmentation:
        if isinstance(L, int):
            tol = k / 10
        elif isinstance(L, np.ndarray):
            tol = k * b / 10
    else:
        tol = 1e-9

    if args.obj in ['SoP', 'SoT', 'FS', 'SoIP']:

        for i in range(k):
            S_greedy, best = eisenberg_noe_bailout_greedy(
                P_bar, A, C, L, b, k, V, S_greedy, v, num_iters=num_iters, workers=workers)

        expected_objective_value_greedy = best

        S_centralities = create_set_helper(centralities, k, b, L)

        expected_objective_value_centralities = eisenberg_noe_bailout(
            P_bar, A, C, L, S_centralities, None, v, num_iters=num_iters, workers=workers)

        S_out_degrees = create_set_helper(out_degrees, k, b, L)

        expected_objective_value_out_degrees = eisenberg_noe_bailout(
            P_bar, A, C, L, S_out_degrees, None, v, num_iters=num_iters, workers=workers)

        S_pageranks = create_set_helper(pageranks, k, b, L)

        expected_objective_value_pageranks = eisenberg_noe_bailout(
            P_bar, A, C, L, S_pageranks, None, v, num_iters=num_iters, workers=workers)

        S_wealths = create_set_helper(wealths, k, b, L)

        expected_objective_value_wealths = eisenberg_noe_bailout(
            P_bar, A, C, L, S_wealths, None, v, num_iters=num_iters, workers=workers)

        S_random = create_set_helper(random_order, k, b, L)

        expected_objective_value_random = eisenberg_noe_bailout(
            P_bar, A, C, L, S_random, None, v, num_iters=num_iters, workers=workers)

        expected_objective_value_randomized_rounding = eisenberg_noe_bailout_randomized_rounding(
            P_bar, A, C, L, b, k, None, v, tol=tol, num_iters=num_iters, workers=workers)

    elif args.obj == 'MD':

        for i in range(k):
            S_greedy, best = eisenberg_noe_bailout_greedy_min_default(
                P_bar, A, C, L, V, S_greedy, eps, num_iters=num_iters, workers=workers)

        expected_objective_value_greedy = best

        S_centralities = create_set_helper(centralities, k, b, L)


        expected_objective_value_centralities = eisenberg_noe_bailout_min_default(
            P_bar, A, C, L, S_centralities, None, eps, num_iters=num_iters, workers=workers)

        S_out_degrees = create_set_helper(out_degrees, k, b, L)

        expected_objective_value_out_degrees = eisenberg_noe_bailout_min_default(
            P_bar, A, C, L, S_out_degrees, None, eps, num_iters=num_iters, workers=workers)

        S_pageranks = create_set_helper(pageranks, k, b, L)

        expected_objective_value_pageranks = eisenberg_noe_bailout_min_default(
            P_bar, A, C, L, S_pageranks, None, eps, num_iters=num_iters, workers=workers)

        S_wealths = create_set_helper(wealths, k, b, L)

        expected_objective_value_wealths = eisenberg_noe_bailout_min_default(
            P_bar, A, C, L, S_wealths, None, eps, num_iters=num_iters, workers=workers)

        S_random = create_set_helper(random_order, k, b, L)

        expected_objective_value_random = eisenberg_noe_bailout(
            P_bar, A, C, L, S_random, None, eps, num_iters=num_iters, workers=workers)

        expected_objective_value_randomized_rounding = eisenberg_noe_bailout_randomized_rounding_min_default(
            P_bar, A, C, L, b, k, None, eps, tol=tol, num_iters=num_iters, workers=workers)


    outfile_suffix = '{}_{}_{}.png'.format(args.obj, args.dataset, L if isinstance(L, int) else 'custom')



    graph_plot(k, [(expected_objective_value_greedy, S_greedy, 'Greedy'),
                               (expected_objective_value_centralities, S_centralities, 'Top-k Centralities'),
                               (expected_objective_value_out_degrees, S_out_degrees, 'Top-k Outdegrees'),
                               (expected_objective_value_pageranks, S_pageranks, 'Top-k Pagerank'),
                               (expected_objective_value_wealths, S_wealths, 'Top-k Wealths (poorest)'),
                               (expected_objective_value_randomized_rounding, 'Randomized Rounding'),
                               (expected_objective_value_random, S_random, 'Random Permutation')],
                     G, P, outfile_suffix, args.obj, L, args.layout)
