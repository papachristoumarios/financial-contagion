from eisenberg_noe import *
from german_banks_dataloader import *
import seaborn as sns


if __name__ == '__main__':
    sns.set_theme()

    data, A, P_bar, P, adj, _, _, _, _, _, C, B, w, G = load_german_banks_dataset()
    n = len(G)
    eps = 1
    num_iters = int(n**2 / (eps**2) * np.log(n))

    expected_numer_of_saved_nodes_no_intervention = eisenberg_noe_bailout(P_bar, A, C, 0, set(), None, num_iters=num_iters)

    L = 1000000
    k_range = np.arange(1, 1 + len(G))
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

    expected_number_of_saved_nodes_greedy = []
    expected_number_of_saved_nodes_centralities = []
    expected_number_of_saved_nodes_out_degrees = []
    expected_number_of_saved_nodes_pageranks = []
    expected_number_of_saved_nodes_wealths = []

    for k in k_range:
        print('k = {}'.format(k))
        best, best_arg = -1, None
        pbar = tqdm.tqdm(range(len(G) - k))
        for u in V - S:
            value = eisenberg_noe_bailout(P_bar, A, C, L, S | {u}, None, num_iters=num_iters)
            pbar.update()
            if value >= best:
                best = value
                best_arg = u

        pbar.close()
        S |= {best_arg}

        expected_number_of_saved_nodes_greedy.append(best)

        S_centralities = set([x[0] for x in centralities[:k]])

        expected_number_of_saved_nodes_centralities.append(eisenberg_noe_bailout(
            P_bar, A, C, L, S_centralities, None, num_iters=num_iters))

        S_out_degrees = set([x[0] for x in out_degrees[:k]])

        expected_number_of_saved_nodes_out_degrees.append(eisenberg_noe_bailout(
            P_bar, A, C, L, S_out_degrees, None, num_iters=num_iters))

        S_pageranks = set([x[0] for x in pageranks[:k]])

        expected_number_of_saved_nodes_pageranks.append(eisenberg_noe_bailout(
            P_bar, A, C, L, S_pageranks, None, num_iters=num_iters))

        S_wealths = set([x[0] for x in wealths[:k]])

        expected_number_of_saved_nodes_wealths.append(eisenberg_noe_bailout(
            P_bar, A, C, L, S_wealths, None, num_iters=num_iters))

    plt.figure()
    plt.plot(k_range, expected_number_of_saved_nodes_greedy, label='Greedy')
    plt.plot(k_range, expected_number_of_saved_nodes_centralities, label='Top-k Centralities')
    plt.plot(k_range, expected_number_of_saved_nodes_out_degrees, label='Top-k Out-degrees')
    plt.plot(k_range, expected_number_of_saved_nodes_pageranks, label='Top-k Pagerank')
    plt.plot(k_range, expected_number_of_saved_nodes_wealths, label='Top-k Wealths')


    plt.legend()
    plt.savefig('bailouts.png')
    #plt.show()
