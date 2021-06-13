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
    parser.add_argument('--enable_minorities', action='store_true', help='Optimize gini subject to minority properties')
    parser.add_argument('--enable_network', action='store_true', help='Optimize gini subject to network constraints')
    args = parser.parse_args()
    return args


def gini_pof_plot(k_range, ginis, results, outfile, obj, L, b, num_std=0.5, show=False):
    plt.figure(figsize=(10, 10))
    colors = iter(cm.rainbow(np.linspace(0, 1, 2 * len(k_range))))
    if isinstance(L, int):
        plt.title('{} objective for $L = {}$'.format(obj, L))
        plt.xlabel('Target Gini Coefficient Bound')
    elif isinstance(L, np.ndarray):
        plt.title('{} objective for custom bailouts with budget increase rate {}'.format(obj, b))
        plt.xlabel('Target Gini Coefficient')
    plt.ylabel('PoF')

    pofs = []


    unconstrained_lp_means = np.array([x[2] for x in results[-1][0]])

    for i, (result, label) in enumerate(results):
        constrained_lp_means = np.array([x[2] for x in result])
        pofs.append(unconstrained_lp_means / constrained_lp_means)

    pofs = np.vstack(pofs)


    for i, k in enumerate(k_range):
        c = next(colors)
        plt.plot(ginis, pofs[:, i], color=c, marker='x', label='k = {}'.format(k))

    plt.legend()
    plt.xlim(ginis[0], ginis[-1])
    plt.savefig('pof_target_gini' + outfile)

    if show:
        plt.show()

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

    if (args.enable_minorities or args.enable_network) and not (args.enable_minorities ^ args.enable_network):
        raise Exception('Only one of the arguments can be used')

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

    beta = 1 - B / P_bar

    if args.obj == 'SoP':
        v = np.ones(shape=(len(G), 1))
    elif args.obj == 'SoT':
        v = 1 - beta
    elif args.obj == 'SoIP':
        v = beta
    elif args.obj == 'FS':
        v = 1 / P_bar


    n = len(G)

    if args.enable_minorities and args.dataset != 'safegraph':
        p_minority = np.random.beta(a=2, b=5, size=(n, 1))

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
        k_range = np.arange(1, 1 + len(G), len(G) // 6)
    else:
        k_range = np.arange(1, 1 + args.max_k, args.max_k // 6)

    V = set(list(G.nodes()))
    eps = args.eps
    ginis = np.linspace(0, 1, 6)

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
                    P_bar, A, C, L, b, k, gini, p_minority, v, network_based=args.enable_network, tol=tol, num_iters=num_iters, workers=workers))

        elif args.obj == 'MD':

            for gini in ginis:

                expected_objective_value_randomized_ginis[gini].append(eisenberg_noe_bailout_randomized_rounding_min_default(
                    P_bar, A, C, L, b, k, gini, p_minority, eps, network_based=args.enable_network, tol=tol, num_iters=num_iters, workers=workers))

        pbar.update()

    pbar.close()

    outfile_suffix = '{}_{}_{}.png'.format(args.obj, args.dataset, L if isinstance(L, int) else 'custom')

    gini_pof_plot(k_range, ginis, [(val, 'Target Gini = {}'.format(key)) for key, val in expected_objective_value_randomized_ginis.items()],
                     outfile_suffix, args.obj, L, b,
                     num_std=args.num_std)
