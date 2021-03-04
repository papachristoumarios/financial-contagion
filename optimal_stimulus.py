from eisenberg_noe import *

if __name__ == '__main__':

    G = nx.gnp_random_graph(40, 0.7).to_directed()

    n = len(G)

    G, adj, B, P, P_bar, C, A = process_financial_network(G)

    print(eisenberg_noe_bailout_total_budget(P_bar, A, C, 100, num_iters=10))
