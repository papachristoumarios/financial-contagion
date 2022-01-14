# Supplementary Code for "Allocating Stimulus Checks in Times of Crisis"

## Setup

Install the required packages with

```bash
pip install -r requirements.txt
```

## Usage

Use the following command to see available options for every snippet

```bash
python {snippet.py} --help
```

## Scripts

A list of scripts to reproduce the results of the paper follows:

* `seed_subsidy_allocation.py`. Produces a plot that compares the greedy, the randomized rounding algorithm and the baselines on optimizing the welfare objective. The values are also compared to the optimal solution of the fractional relaxation.
* `seed_subsidy_allocation_fairness.py`. Compares the randomized rounding solution and the corresponding fractional solutions where fairness constraints on the Gini Coefficient are present.
* `seed_subsidy_allocation_fairness_pof.py`. Similarly to `seed_subsidy_allocation_fairness.py` it plots the relation between the PoF and the upper bound on the value of the corresponding Gini Coefficient
* `seed_subsidy_allocation_sbm.py`. Plots the behaviour of PoF for a stochastic blockmodel generated from two equally-sized cliques connected with i.i.d. edges of bias `q` with one another.

The data that can be used via the `--dataset` flag are the following:

* [`german_banks`](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3771035)
* [`eba`](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2642423)
* [`venmo`](https://github.com/sa7mon/venmo-data)

## Citation

Please use the following citation when referring to the paper and the source code

```bibtex
@article{papachristou2021allocating,
  title={Allocating Stimulus Checks in Times of Crisis},
  author={Papachristou, Marios and Kleinberg, Jon},
  journal={Proceedings of The Web Conference},
  year={2022}
}
```
