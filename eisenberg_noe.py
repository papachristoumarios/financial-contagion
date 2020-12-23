import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

G = nx.complete_graph(10)

adj = nx.to_numpy_array(G).astype(np.float64)

P = adj * np.random.randint(100, size=adj.shape).astype(np.float64)

C = 100 + np.random.randint(100, size=(adj.shape[0], 1))

B = np.zeros_like(C)

X = np.random.randint(20, size=(adj.shape[0], 1))

P_bar = P.sum(0)
P_bar = P_bar.reshape(len(P_bar), 1) + B

A = np.apply_along_axis(lambda x: x / x.sum() if x.sum() > 0 else 0, 1, P)

P_eq = P_bar

for i in range(20):
    P_eq = np.minimum(P_bar, np.maximum(0, A @ P_eq + C - X)) 



plt.figure()
plt.plot(P_bar, label='Initial liabilities')
plt.plot(P_eq, label='Equilibrium solution')
plt.title('Eisenberg-Noe Model Simulation')
plt.legend()
plt.show()


