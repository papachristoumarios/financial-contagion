import numpy as np

def fair_partition(z, eps=0.01):
	z_new = ptas_convert(z, eps)
	result = min_sum_partition(z_new)
	return result

def ptas_convert(x, eps=0.01):
	X = x.max()
	k = int(np.ceil(eps * X / len(x)))
	x_new = k * np.ceil(x / k)
	return x_new.astype(np.int64)

def min_sum_partition(x):
	S = np.sum(x)
	n = len(x)

	# DP[i, s] = True iff we can construct a sum of value s including elements
	# from the prefix sequence from 1 to i and is False otherwise
	# The DP runs in pseudopolynomial time O(S * n)
	DP = {}

	# Parent array tracks the value we came from such that we can reconstruct
	# the answer regarding the indices of the elements that belong to the 1st set
	parent = {}

	# Will hold our answer
	indices = []

	# We can make a sum of 0 for every i since taking no elements will work
	for i in range(n + 1):
		DP[i, 0] = True
		parent[i, 0] = (i, -1)

	# We cannot make any possible nonzero sum using 0 elements
	for s in range(1, S + 1):
		DP[0, s] = False

	for i in range(1, n + 1):
		for s in range(1, S + 1):

			DP[i, s] = DP[i - 1, s]
			if x[i - 1] <= s and DP[i, s - x[i - 1]]:
				DP[i, s] = DP[i - 1, s - x[i - 1]]
				# Add traces to the parent array
				parent[i, s] = (i - 1, s - x[i - 1])
			elif DP[i, s]:
				parent[i, s] = (i - 1, s)


	# The optimal solution is the value that is closer to S // 2
	for s in range(S // 2, -1, -1):
		if DP[n, s]:
			opt = S - (2 * s)
			current = (n, s)
			break

	# import pdb; pdb.set_trace()

	#Trace back the solution from the parent array
	while True:
		i, s = current
		_, s_parent = parent[current]

		if s_parent < s and s_parent != -1:
			indices.append(i - 1)

		if s_parent == -1:
			break
		else:
			current = parent[current]

	# Get the rest of indices
	rest = np.array(list(set(range(n)) - set(indices)), dtype=np.int64)

	# Convert to np.array
	indices = np.array(indices, dtype=np.int64)

	return opt, indices, rest
