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


def get_argparser():
    parser = argparse.ArgumentParser(
        description='Discrete stimulus allocation algorithm to maximize SoP, SoT, or SoIT objectives.')
    parser.add_argument('--obj', type=str, default='SoP', help='Type of objective (SoP, SoT, SoIT)',
                        choices=['SoP', 'SoT', 'SoIP', 'FS', 'AS', 'MD'])
    parser.add_argument('--num_iters', type=int, default=-1,
                        help='Number of iterations for Monte Carlo approximation')
    parser.add_argument('-L', type=int, default=1000000, help='Stimulus value')
    parser.add_argument('--dataset', type=str, default='german_banks',
                        help='Dataset to run simulation on', choices=['german_banks', 'eba', 'venmo', 'safegraph', 'random'])
    parser.add_argument('--random_graph', type=str, default='ER', choices=['ER', 'CP', 'SF'],
                        help='Random graph model for artificial data')
    parser.add_argument('--max_k', type=int, default=-1,
                        help='Maximum number of people to bailout through simulation')
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
    parser.add_argument('-b', type=int, default=10000, help='Rate of increase of availbale budget (if different bailouts are selected)')
    parser.add_argument('--randperm_only', action='store_true', help='Plot random permutation only for comparison heuristics')

    return parser


def uncertainty_plot(k_range, results, outfile, obj, L, b, num_std=0.5, show=False):
    plt.figure(figsize=(10, 10))
    colors = iter(cm.rainbow(np.linspace(0, 1, 1 + len(results))))
    if isinstance(L, int):
        plt.title('{} objective for $L = {}$'.format(obj, L))
        plt.xlabel('Number of bailed-out nodes $k$')
    elif isinstance(L, np.ndarray):
        plt.title('{} objective for custom bailouts with budget increase rate {}'.format(obj, b))
        plt.xlabel('Multiples of budget increase $k$')
    plt.ylabel(obj)

    for result, label in results:
        result_means = np.array([x[0] for x in result])
        result_std = np.array([x[1] for x in result])
        if len(result[0]) > 2:
            opt_lp_means = np.array([x[2] for x in result])
            opt_lp_std = np.array([x[3] for x in result])
            c = next(colors)

            plt.plot(k_range, opt_lp_means, c=c, label='Relaxation Optimum')
            plt.fill_between(k_range, result_means - num_std * result_std,
                             result_means + num_std * result_std, color=c, alpha=0.3)

        c = next(colors)
        plt.plot(k_range, result_means, c=c, label=label)
        plt.fill_between(k_range, result_means - num_std * result_std,
                         result_means + num_std * result_std, color=c, alpha=0.3)

    plt.legend()
    plt.xlim(k_range[0], k_range[-1])
    plt.savefig('bailouts_' + outfile)

    if show:
        plt.show()


def truncated_violinplot(data):
    fit_kde_func = sns.categorical._ViolinPlotter.fit_kde

    def reflected_once_kde(self, x, bw):
        lb = 0
        ub = 1

        kde, bw_used = fit_kde_func(self, x, bw)

        kde_evaluate = kde.evaluate

        def truncated_kde_evaluate(x):
            val = np.where((x >= lb) & (x <= ub), kde_evaluate(x), 0)
            val += np.where((x >= lb) & (x <= ub), kde_evaluate(lb-x), 0)
            val += np.where((x > lb) & (x <= ub), kde_evaluate(ub-(x-ub)), 0)
            return val

        kde.evaluate = truncated_kde_evaluate
        return kde, bw_used

    sns.categorical._ViolinPlotter.fit_kde = reflected_once_kde
    sns.violinplot(data=data, cut=0, inner=None, palette='husl')
    sns.categorical._ViolinPlotter.fit_kde = fit_kde_func


def stimuli_plot(k_range, expected_objective_value_randomized_rounding, obj, untruncated_violin, L, b, outfile):
    plt.figure(figsize=(10, 10))
    mean_supports = []

    for k, result in zip(k_range, expected_objective_value_randomized_rounding):
        stimuli_mean = result[-2]
        stimuli_std = result[-1]
        stimuli_mean_support = stimuli_mean[np.where(stimuli_mean > 0)]
        mean_supports.append(stimuli_mean_support)

    if untruncated_violin:
        sns.violinplot(data=mean_supports, palette='husl')
    else:
        truncated_violinplot(mean_supports)
    if isinstance(L, int):
        plt.xlabel('Number of bailed-out nodes $k$')
    elif isinstance(L, int):
        plt.xlabel('Multiples of budget increase $k$')

    plt.ylabel('Significance distributions (support of LP relaxation variables)')
    plt.xticks(k_range - 1, k_range)
    plt.savefig('stimuli.png')

    zs = np.vstack([result[-2] for result in expected_objective_value_randomized_rounding])

    ginis = np.zeros_like(k_range).astype(np.float64)

    for i in range(len(ginis)):
        ginis[i] = gini(zs[i, :])

    plt.figure(figsize=(10, 10))
    plt.plot(k_range, ginis)
    plt.legend()
    if isinstance(L, int):
        plt.title('Gini Coefficients for $L = {}$'.format(L))
        plt.xlabel('Number of bailed-out nodes $k$')
    elif isinstance(L, np.ndarray):
        plt.title('Gini Coefficients for custom bailouts with budget increase rate {}'.format(b))
    plt.ylabel('Gini Coefficient')
    # plt.ylim(0, 1)
    plt.savefig('gini_' + outfile)

    plt.figure(figsize=(15, 10))
    for i in range(zs.shape[-1]):
        plt.plot(np.gradient(zs[:, i]), label='Node {}'.format(i))
    plt.legend()
    if isinstance(L, int):
        plt.xlabel('Number of bailed-out nodes $k$')
    elif isinstance(L, np.ndarray):
        plt.xlabel('Multiples of budget increase $k$')

    plt.ylabel('$\Delta z_i^*$')
    plt.savefig('fractional_stimuli.png')

    zs_mean = np.mean(zs, axis=0)
    most_significant_ranks = np.argsort(zs_mean)[::-1]
    most_significant = zs_mean[most_significant_ranks]

    plt.figure(figsize=(15, 10))
    plt.bar(np.arange(1, 1 + len(most_significant)), most_significant)
    plt.xlabel('Rank of node $r$')
    plt.ylabel('Mean probability of node')
    plt.savefig('barplot_' + outfile)


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
        A, P_bar, P, C, B, L, _, w, G = load_safegraph_dataset()
    elif args.dataset == 'random':
        A, P_bar, P, adj, _, _, _, _, C, B, w, G = generate_random_data(
            args.seed, args.random_graph, args.assets_distribution)

    p_minority = None
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

    if args.max_k <= 0:
        k_range = np.arange(1, 1 + len(G))
    else:
        k_range = np.arange(1, 1 + args.max_k)

    V = set(list(G.nodes()))
    S_greedy = set()
    eps = args.eps

    pageranks = nx.algorithms.pagerank(G)
    pageranks = list(pageranks.items())
    pageranks = list(sorted(pageranks, key=lambda x: (-x[-1], -x[0])))

    centralities = nx.algorithms.centrality.betweenness_centrality(G)
    centralities = list(centralities.items())
    centralities = list(sorted(centralities, key=lambda x: (-x[-1], -x[0])))

    out_degrees = list(sorted([(v, G.out_degree(v)) for v in G], key=lambda x: -x[-1]))

    wealths = list(sorted([(v, w[v, 0]) for v in G], key=lambda x: x[-1]))

    random_order = [(v, 0) for v in G]
    random.shuffle(random_order)

    expected_objective_value_greedy = []
    expected_objective_value_centralities = []
    expected_objective_value_out_degrees = []
    expected_objective_value_pageranks = []
    expected_objective_value_wealths = []
    expected_objective_value_randomized_rounding = []
    expected_objective_value_random = []

    pbar = tqdm.tqdm(k_range)

    for k in k_range:

        if args.resource_augmentation:
            if isinstance(L, int):
                tol = k / 10
            elif isinstance(L, np.ndarray):
                tol = k * b / 10
        else:
            tol = 1e-9

        if args.obj in ['SoP', 'SoT', 'FS', 'SoIP']:

            S_greedy, best = eisenberg_noe_bailout_greedy(
                P_bar, A, C, L, b, k, V, S_greedy, v, num_iters=num_iters, workers=workers)

            expected_objective_value_greedy.append(best)

            S_centralities = create_set_helper(centralities, k, b, L)

            expected_objective_value_centralities.append(eisenberg_noe_bailout(
                P_bar, A, C, L, S_centralities, None, v, num_iters=num_iters, workers=workers))

            S_out_degrees = create_set_helper(out_degrees, k, b, L)

            expected_objective_value_out_degrees.append(eisenberg_noe_bailout(
                P_bar, A, C, L, S_out_degrees, None, v, num_iters=num_iters, workers=workers))

            S_pageranks = create_set_helper(pageranks, k, b, L)

            expected_objective_value_pageranks.append(eisenberg_noe_bailout(
                P_bar, A, C, L, S_pageranks, None, v, num_iters=num_iters, workers=workers))

            S_wealths = create_set_helper(wealths, k, b, L)

            expected_objective_value_wealths.append(eisenberg_noe_bailout(
                P_bar, A, C, L, S_wealths, None, v, num_iters=num_iters, workers=workers))

            S_random = create_set_helper(random_order, k, b, L)

            expected_objective_value_random.append(eisenberg_noe_bailout(
                P_bar, A, C, L, S_random, None, v, num_iters=num_iters, workers=workers))


            expected_objective_value_randomized_rounding.append(eisenberg_noe_bailout_randomized_rounding(
                P_bar, A, C, L, b, k, None, None, v, network_based=False, tol=tol, num_iters=num_iters, workers=workers))

        elif args.obj == 'MD':

            S_greedy, best = eisenberg_noe_bailout_greedy_min_default(
                P_bar, A, C, L, b, k, V, S_greedy, eps, num_iters=num_iters, workers=workers)

            expected_objective_value_greedy.append(best)

            S_centralities = create_set_helper(centralities, k, b, L)

            expected_objective_value_centralities.append(eisenberg_noe_bailout_min_default(
                P_bar, A, C, L, S_centralities, None, eps, num_iters=num_iters, workers=workers))

            S_out_degrees = create_set_helper(out_degrees, k, b, L)

            expected_objective_value_out_degrees.append(eisenberg_noe_bailout_min_default(
                P_bar, A, C, L, S_out_degrees, None, eps, num_iters=num_iters, workers=workers))

            S_pageranks = create_set_helper(pageranks, k, b, L)

            expected_objective_value_pageranks.append(eisenberg_noe_bailout_min_default(
                P_bar, A, C, L, S_pageranks, None, eps, num_iters=num_iters, workers=workers))

            S_wealths = create_set_helper(wealths, k, b, L)

            expected_objective_value_wealths.append(eisenberg_noe_bailout_min_default(
                P_bar, A, C, L, S_wealths, None, eps, num_iters=num_iters, workers=workers))

            S_random = create_set_helper(random_order, k, b, L)

            expected_objective_value_random.append(eisenberg_noe_bailout(
                P_bar, A, C, L, S_random, None, eps, num_iters=num_iters, workers=workers))

            expected_objective_value_randomized_rounding.append(eisenberg_noe_bailout_randomized_rounding_min_default(
                P_bar, A, C, L, b, k, None, None, eps, network_based=False, tol=tol, num_iters=num_iters, workers=workers))

        pbar.update()

    pbar.close()

    outfile_suffix = '{}_{}_{}.png'.format(args.obj, args.dataset, L if isinstance(L, int) else 'custom')

    if args.randperm_only:
        uncertainty_plot(k_range, [(expected_objective_value_greedy, 'Greedy'),
                                    (expected_objective_value_wealths, 'Top-k Wealths (poorest)'),
                                    (expected_objective_value_randomized_rounding, 'Randomized Rounding'),
                                    (expected_objective_value_random, 'Random Permutation')],
                                    outfile_suffix, args.obj, L, b,
                                    num_std=args.num_std)
    else:
        uncertainty_plot(k_range, [(expected_objective_value_greedy, 'Greedy'),
                                   (expected_objective_value_centralities, 'Top-k Centralities'),
                                   (expected_objective_value_out_degrees, 'Top-k Outdegrees'),
                                   (expected_objective_value_pageranks, 'Top-k Pagerank'),
                                   (expected_objective_value_wealths, 'Top-k Wealths (poorest)'),
                                   (expected_objective_value_randomized_rounding, 'Randomized Rounding'),
                                   (expected_objective_value_random, 'Random Permutation')],
                                   outfile_suffix, args.obj, L, b,
                                   num_std=args.num_std)

    stimuli_plot(k_range, expected_objective_value_randomized_rounding,
                 args.obj, args.untruncated_violin, L, b, outfile_suffix)
