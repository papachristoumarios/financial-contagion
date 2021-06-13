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
    parser.add_argument('--eps', type=float, default=1e-4,
                        help='Parameter in the transformation of the increasing objective to a strictly increasing objective')
    parser.add_argument('-b', type=int, default=100000, help='Rate of increase of availbale budget (if different bailouts are selected)')
    parser.add_argument('-n', type=int, default=20, help='Number of nodes')
    parser.add_argument('--stochastic', action='store_true', help='SBM')

    return parser


def sbm_plot(results, results_unconstrained, D_range, gini_range, outfile, obj, L, n, k, stochastic, show=False):
    colors = iter(cm.rainbow(np.linspace(0, 1, 1 + len(results))))

    plt.figure(figsize=(10, 10))
    plt.title('PoF for $SBM(n={})$ for {} Objective, $L = {}$, $k = {}$'.format(n, obj, L, k))
    if stochastic:
        plt.xlabel('$\log r$')
    else:
        plt.xlabel('$\log D$')
    plt.ylabel('$\log$ PoF')

    colors = iter(cm.rainbow(np.linspace(0, 1, 1 + len(results))))


    bound = 1 / D_range
    y_lim_max = -1

    D_range = np.log(D_range)

    for gini in gini_range:

        color = next(colors)

        unconstrained_lp_mean = np.array([x[0] for x in results_unconstrained[gini]])
        unconstrained_lp_std = np.array([x[1] for x in results_unconstrained[gini]])


        constrained_lp_mean = np.array([x[0] for x in results[gini]])
        constrained_lp_std = np.array([x[1] for x in results[gini]])


        pof_lp_mean = np.log(unconstrained_lp_mean / constrained_lp_mean)

        p = np.polyfit(D_range, pof_lp_mean, deg=1)

        plt.plot(D_range, pof_lp_mean, color=color, label='Target Gini = {}, '.format(gini) + r"$y \propto x^{" + str(round(p[0], 5)) + r"}$")


        # y_lim_max = max(y_lim_max, pof_lp_mean.max())


    plt.legend()

    plt.savefig('pof_sbm_{}'.format(outfile))


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

    p_minority = None

    if args.num_iters <= 0:
        eps = 1
        num_iters = int(n**2 / (eps**2) * np.log(args.n))
    else:
        num_iters = args.num_iters

    L = int(args.L)

    b = args.b
    k = args.k

    if args.resource_augmentation:
        if isinstance(L, int):
            tol = k / 10
        elif isinstance(L, np.ndarray):
            tol = k * b / 10
    else:
        tol = 1e-9

    if args.stochastic:
        D_range = np.linspace(0.1, 1, 10)
    else:
        D_range = []
        for i in range(args.n // 2 + 1):
            if i % args.n // 2:
                D_range.append(i)

        D_range = np.array(D_range)

    gini_range = np.array([0, 0.01, 0.05, 0.1])

    expected_objective_value_randomized_rounding = collections.defaultdict(list)
    expected_objective_value_randomized_rounding_unconstrained = collections.defaultdict(list)

    for gini in gini_range:
        pbar = tqdm.tqdm(D_range)

        for D in D_range:
            A, P_bar, P, _, _, _, _, C, B, w, G = generate_sbm_pair(args.n, D, seed=args.seed, stochastic=args.stochastic)
            beta = 1 - B / P_bar

            if args.obj == 'SoP':
                v = np.ones(shape=(len(G), 1))
            elif args.obj == 'SoT':
                v = 1 - beta
            elif args.obj == 'SoIP':
                v = beta
            elif args.obj == 'FS':
                v = 1 / P_bar


            if args.obj in ['SoP', 'SoT', 'FS', 'SoIP']:

                expected_objective_value_randomized_rounding[gini].append(eisenberg_noe_bailout_randomized_rounding(
                    P_bar, A, C, L, b, k, gini, None, v, network_based=True, rounding=False, tol=tol, num_iters=num_iters, workers=workers))

            elif args.obj == 'MD':

                expected_objective_value_randomized_rounding[gini].append(eisenberg_noe_bailout_randomized_rounding_min_default(
                    P_bar, A, C, L, b, k, gini, None, eps, True, network_based=True, rounding=False, tol=tol, num_iters=num_iters, workers=workers))


            if args.obj in ['SoP', 'SoT', 'FS', 'SoIP']:
                expected_objective_value_randomized_rounding_unconstrained[gini].append(eisenberg_noe_bailout_randomized_rounding(
                    P_bar, A, C, L, b, k, None, None, v, network_based=False, rounding=False, tol=tol, num_iters=num_iters, workers=workers))
            elif args.obj == 'MD':
                expected_objective_value_randomized_rounding_unconstrained[gini].append(eisenberg_noe_bailout_randomized_rounding_min_default(
                    P_bar, A, C, L, b, k, None, None, eps, network_based=False, rounding=False, tol=tol, num_iters=num_iters, workers=workers))

            pbar.update()


        outfile_suffix = '{}_sbm_{}_{}.png'.format(args.obj, L if isinstance(L, float) else 'custom', 'stochastic' if args.stochastic else '')

        pbar.close()



    sbm_plot(expected_objective_value_randomized_rounding, expected_objective_value_randomized_rounding_unconstrained, D_range, gini_range, outfile_suffix, args.obj, L, args.n, args.k, args.stochastic)
