from eisenberg_noe import *
from german_banks_dataloader import *
from glasserman_young_dataloader import *
from generator import *
import seaborn as sns
import argparse
import matplotlib.cm as cm


def get_argparser():
    parser = argparse.ArgumentParser(
        description='Discrete stimulus allocation algorithm to maximize SoP, SoT, or SoIT objectives.')
    parser.add_argument('--obj', type=str, default='SoP', help='Type of objective (SoP, SoT, SoIT)')
    parser.add_argument('--num_iters', type=int, default=-1,
                        help='Number of iterations for Monte Carlo approximation')
    parser.add_argument('-L', type=int, default=1000000, help='Stimulus value')
    parser.add_argument('--dataset', type=str, default='german_banks',
                        help='Dataset to run simulation on (german_banks, glasserman_young, random)')
    parser.add_argument('--random_graph', type=str, default='ER',
                        help='Random graph model for artificial data')
    parser.add_argument('--max_k', type=int, default=-1,
                        help='Maximum number of people to bailout through simulation')
    parser.add_argument('--resource_augmentation', action='store_true',
                        help='Apply resource augmentation to the randomized rounding LP algorithm')
    parser.add_argument('--seed', type=int, default=42, help='Random seed to be used')
    parser.add_argument('--shocks_distribution', type=str, default='beta')
    parser.add_argument('--assets_distribution', type=str, default='exponential')
    parser.add_argument('--workers', type=int, default=1, help='Number of workers')
    parser.add_argument('--num_std', type=float, default=0.5, help='Number of stds to plot in the uncertainty plot')
    parser.add_argument('--untruncated_violin', action='store_true', help='Untruncated violin plots')

    return parser


def uncertainty_plot(k_range, results, outfile, obj, L, apx_factor=1, num_std=0.5, show=False):
    plt.figure(figsize=(10, 10))
    colors = iter(cm.rainbow(np.linspace(0, 1, 1 + len(results))))
    plt.title('{} objective for $L = {}$'.format(obj, L))
    plt.xlabel('Number of bailed-out nodes $k$')
    plt.ylabel(obj)

    for result, label in results:
        result_means = np.array([x[0] for x in result])
        result_std = np.array([x[1] for x in result])
        # if len(result[0]) > 2 and apx_factor < 1:
            # opt_lp_means = np.array([x[2] for x in result])
            # opt_lp_std = np.array([x[3] for x in result])
            # c = next(colors)

            # plt.plot(k_range, apx_factor * opt_lp_means, c=c, label='LP Lower Bound')
            # plt.fill_between(k_range, apx_factor * result_means - num_std * result_std,
                             # apx_factor * result_means + num_std * result_std, color=c, alpha=0.3)


        c = next(colors)
        plt.plot(k_range, result_means, c=c, label=label)
        plt.fill_between(k_range, result_means - num_std * result_std,
                         result_means + num_std * result_std, color=c, alpha=0.3)

    plt.legend()
    plt.xlim(k_range[0], k_range[-1])
    plt.savefig(outfile)

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
            val = np.where((x>=lb)&(x<=ub), kde_evaluate(x), 0)
            val += np.where((x>=lb)&(x<=ub), kde_evaluate(lb-x), 0)
            val += np.where((x>lb)&(x<=ub), kde_evaluate(ub-(x-ub)), 0)
            return val

        kde.evaluate = truncated_kde_evaluate
        return kde, bw_used

    sns.categorical._ViolinPlotter.fit_kde = reflected_once_kde
    sns.violinplot(data=data, cut=0, inner=None, palette='husl')
    sns.categorical._ViolinPlotter.fit_kde = fit_kde_func

def stimuli_plot(k_range, expected_objective_value_randomized_rounding, untruncated_violin):
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
    plt.xlabel('Number of bailed-out nodes $k$')
    plt.ylabel('Significance distributions (support of LP relaxation variables)')
    plt.xticks(k_range - 1, k_range)
    plt.savefig('stimuli.png')

    fairness = np.zeros_like(k_range, dtype=np.float64)
    standard_deviation = np.zeros_like(k_range, dtype=np.float64)
    quantile = np.zeros_like(k_range, dtype=np.float64)

    for k, mean_support in zip(k_range, expected_objective_value_randomized_rounding):
        stimuli_mean = np.array(result[-2])
        fairness[k-1] = np.mean(np.abs(stimuli_mean - k / len(stimuli_mean)))
        standard_deviation[k-1] = np.std(mean_supports[k-1], ddof=1)
        stimuli_mean = np.sort(stimuli_mean)
        stimuli_mean_cumsum = np.cumsum(stimuli_mean)
        # import pdb; pdb.set_trace()
        # quantile[k - 1] = (len(stimuli_mean) - np.where(stimuli_mean_cumsum >= k)[0]) / len(stimuli_mean)


    plt.figure(figsize=(15, 10))
    # plt.plot(k_range, np.log(fairness + 1e-6), label='Fairness Score (log)')
    plt.plot(k_range, np.log(standard_deviation + 1e-6), label='Standard Deviation of $\\bar z_i^*$ (log)')
    # plt.plot(k_range, np.log(quantile + 1e-6), label='Quantile (log)')
    plt.xlabel('Number of bailed-out nodes $k$')
    plt.ylabel('Fairness Metrics')
    plt.legend()

    plt.savefig('fairness.png')

if __name__ == '__main__':
    args = get_argparser().parse_args()
    seed = args.seed
    workers = args.workers
    sns.set_theme()

    if args.dataset == 'german_banks':
        data, A, P_bar, P, adj, _, _, _, _, _, C, B, w, G = load_german_banks_dataset()
    elif args.dataset == 'glasserman_young':
        data, A, P_bar, P, adj, _, _, _, _, _, C, B, w, G = next(load_glasserman_young_dataset())
    elif args.dataset == 'random':
        A, P_bar, P, adj, _, _, _, _, C, B, w, G = generate_random_data(
            args.seed, args.random_graph, args.assets_distribution)

    else:
        raise Exception(
            'Incorrect dataset specified. Please try one of the following: (german_banks, glasserman_young).')

    beta = B / P_bar

    if args.obj == 'SoP':
        v = np.ones(shape=(len(G), 1))
        apx_factor = 1 - beta.max()
    elif args.obj == 'SoT':
        v = 1 - beta
        apx_factor = 1 - beta.min()
    elif args.obj == 'SoIP':
        v = beta
        apx_factor = -1
        raise Exception('TODO find apx factor')
    else:
        raise Exception('Unknown objective.')

    n = len(G)

    if args.num_iters <= 0:
        eps = 1
        num_iters = int(n**2 / (eps**2) * np.log(n))
    else:
        num_iters = args.num_iters

    if args.L <= 0:
        raise Exception('Please use a positive amount for L')
    else:
        L = min(args.L, C.max())

    if args.max_k <= 0:
        k_range = np.arange(1, 1 + len(G))
    else:
        k_range = np.arange(1, 1 + args.max_k)

    V = set(list(G.nodes()))
    S = set()

    pageranks = nx.algorithms.pagerank(G)
    pageranks = list(pageranks.items())
    pageranks = list(sorted(pageranks, key=lambda x: -x[-1]))

    centralities = nx.algorithms.centrality.betweenness_centrality(G)
    centralities = list(centralities.items())
    centralities = list(sorted(centralities, key=lambda x: -x[-1]))

    out_degrees = list(sorted([(v, G.out_degree(v)) for v in G], key=lambda x: -x[-1]))

    wealths = list(sorted([(v, w[v, 0]) for v in G], key=lambda x: x[-1]))

    expected_objective_value_greedy = []
    expected_objective_value_centralities = []
    expected_objective_value_out_degrees = []
    expected_objective_value_pageranks = []
    expected_objective_value_wealths = []
    expected_objective_value_randomized_rounding = []

    pbar = tqdm.tqdm(k_range)

    for k in k_range:
        best, best_arg = (-1, -1), None
        for u in V - S:
            value = eisenberg_noe_bailout(
                P_bar, A, C, L, S | {u}, None, v, num_iters=num_iters, workers=workers)
            if value[0] >= best[0]:
                best = value
                best_arg = u

        S |= {best_arg}

        expected_objective_value_greedy.append(best)

        S_centralities = set([x[0] for x in centralities[:k]])

        expected_objective_value_centralities.append(eisenberg_noe_bailout(
            P_bar, A, C, L, S_centralities, None, v, num_iters=num_iters, workers=workers))

        S_out_degrees = set([x[0] for x in out_degrees[:k]])

        expected_objective_value_out_degrees.append(eisenberg_noe_bailout(
            P_bar, A, C, L, S_out_degrees, None, v, num_iters=num_iters, workers=workers))

        S_pageranks = set([x[0] for x in pageranks[:k]])

        expected_objective_value_pageranks.append(eisenberg_noe_bailout(
            P_bar, A, C, L, S_pageranks, None, v, num_iters=num_iters, workers=workers))

        S_wealths = set([x[0] for x in wealths[:k]])

        expected_objective_value_wealths.append(eisenberg_noe_bailout(
            P_bar, A, C, L, S_wealths, None, v, num_iters=num_iters, workers=workers))

        if args.resource_augmentation:
            tol = np.sqrt(np.log(k))
        else:
            tol = 1e-9

        expected_objective_value_randomized_rounding.append(eisenberg_noe_bailout_randomized_rounding(
            P_bar, A, C, L, k, v, tol=tol, num_iters=num_iters, workers=workers))

        pbar.update()

    pbar.close()

    uncertainty_plot(k_range, [(expected_objective_value_greedy, 'Greedy'),
                                (expected_objective_value_centralities, 'Top-k Centralities'),
                                (expected_objective_value_out_degrees, 'Top-k Outdegrees'),
                                (expected_objective_value_pageranks, 'Top-k Pagerank'),
                                (expected_objective_value_wealths, 'Top-k Wealths (poorest)'),
                                (expected_objective_value_randomized_rounding, 'Randomized Rounding (ours)')],
                                'bailouts_{}_{}_{}.png'.format(args.obj, args.dataset, L), args.obj, L,
                                num_std=args.num_std, apx_factor=apx_factor)

    stimuli_plot(k_range, expected_objective_value_randomized_rounding, args.untruncated_violin)
