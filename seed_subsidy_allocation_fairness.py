from eisenberg_noe import *
from german_banks_dataloader import *
from eba_dataloader import *
from venmo_dataloader import *
from safegraph_dataloader import *
from generator import *
from metrics import *
import utils
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
    parser.add_argument('--ginis', type=str, default='', help='Delimited list of gini indices')
    parser.add_argument('--enable_minorities', action='store_true', help='Optimize gini subject to minority properties')
    args = parser.parse_args()
    args.ginis = [float(x) for x in args.ginis.split(',')]
    return args


def uncertainty_plot(k_range, results, outfile, obj, L, b, num_std=0.5, show=False):
    plt.figure(figsize=(10, 10))
    colors = iter(cm.rainbow(np.linspace(0, 1, 2 * len(results))))
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
            plt.plot(k_range, opt_lp_means, c=c, label='Relaxation Optimum ({})'.format(label))
            plt.fill_between(k_range, result_means - num_std * result_std,
                             result_means + num_std * result_std, color=c, alpha=0.3)

        c = next(colors)
        plt.plot(k_range, result_means, c=c, label='Rounded ({})'.format(label))
        plt.fill_between(k_range, result_means - num_std * result_std,
                         result_means + num_std * result_std, color=c, alpha=0.3)

    plt.legend()
    plt.xlim(k_range[0], k_range[-1])
    plt.savefig('bailouts_gini_target' + outfile)

    if show:
        plt.show()

def ginis_plot(k_range, expected_objective_value_ginis, obj, L, b, outfile):
    plt.figure(figsize=(10, 10))

    for gini, expected_objective_value_randomized_rounding in expected_objective_value_ginis.items():

        zs = np.vstack([result[-2] for result in expected_objective_value_randomized_rounding])

        ginis = np.zeros_like(k_range).astype(np.float64)

        for i in range(len(ginis)):
            if isinstance(L, int):
                ginis[i] = utils.gini(zs[i, :])
            elif isinstance(L, np.ndarray):
                ginis[i] = utils.gini(zs[i, :].flatten() * L.flatten())

        plt.plot(k_range, ginis, label='Target Gini = {}'.format(gini))

    plt.legend()
    if isinstance(L, int):
        plt.title('Gini Coefficients for $L = {}$'.format(L))
        plt.xlabel('Number of bailed-out nodes $k$')
    elif isinstance(L, np.ndarray):
        plt.title('Gini Coefficients for custom bailouts with budget increase rate {}'.format(b))
    plt.ylabel('Gini Coefficient')

    plt.savefig('gini_target_gini' + outfile)


if __name__ == '__main__':
    args = get_argparser()
    seed = args.seed
    workers = args.workers
    sns.set_theme()
    LARGE_SIZE = 16
    plt.rc('axes', labelsize=LARGE_SIZE)
    plt.rc('axes', titlesize=LARGE_SIZE)

    np.random.seed(seed)
    random.seed(seed)

    p_minority = None

    if args.dataset == 'german_banks':
        data, A, P_bar, P, adj, _, _, _, _, _, C, B, w, G = load_german_banks_dataset()
    elif args.dataset == 'eba':
        data, A, P_bar, P, adj, _, _, _, _, _, C, B, w, G = next(load_eba_dataset())
    elif args.dataset == 'venmo':
        A, P_bar, P, adj, _, _, _, _, C, B, w, G = load_venmo_dataset()
    elif args.dataset == 'safegraph':
        A, P_bar, P, C, B, L, p_minority, w, G = load_safegraph_dataset()
        if not args.enable_minorities:
            p_minority = None
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

    if args.max_k <= 0:
        k_range = np.arange(1, 1 + len(G))
    else:
        k_range = np.arange(1, 1 + args.max_k)

    V = set(list(G.nodes()))
    eps = args.eps
    ginis = args.ginis

    expected_objective_value_randomized_ginis = collections.defaultdict(list)

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

            for gini in ginis:
                expected_objective_value_randomized_ginis[gini].append(eisenberg_noe_bailout_randomized_rounding(
                    P_bar, A, C, L, b, k, gini, p_minority, v, tol=tol, num_iters=num_iters, workers=workers))

        elif args.obj == 'MD':

            for gini in ginis:

                expected_objective_value_randomized_ginis[gini].append(eisenberg_noe_bailout_randomized_rounding_min_default(
                    P_bar, A, C, L, b, k, gini, p_minority, eps, tol=tol, num_iters=num_iters, workers=workers))

        pbar.update()

    pbar.close()

    outfile_suffix = '{}_{}_{}.png'.format(args.obj, args.dataset, L if isinstance(L, int) else 'custom')

    uncertainty_plot(k_range, [(val, 'Target Gini = {}'.format(key)) for key, val in expected_objective_value_randomized_ginis.items()],
                     outfile_suffix, args.obj, L, b,
                     num_std=args.num_std)

    ginis_plot(k_range, expected_objective_value_randomized_ginis,
                 args.obj, L, b, outfile_suffix)
