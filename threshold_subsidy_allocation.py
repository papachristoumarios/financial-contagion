from eisenberg_noe import *

if __name__ == '__main__':
    G = nx.gnp_random_graph(40, 0.7).to_directed()

    n = len(G)

    G, adj, B, P, P_bar, C, A, w = process_financial_network(G)

    eisenberg_noe_bailout_threshold_model(P_bar, A, C, B, w)
