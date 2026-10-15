# Stability of Binning – Closest Stable Binning

This repository contains the implementations of the algorithms presented in the paper, including **stability verification**, **radius computation**, and the **closest stable binning construction problem**. It also includes the experimental code and scripts used to generate the results and figures presented in the paper.

## Stability Verification and Radius Algorithms

The `par.py` file contains the implementations of the **stability verification algorithm** and the **radius computation algorithm** for the `COUNT` aggregation function, including all optimizations.

To run the optimized versions of these algorithms:

- `alg1_opt.py` — Runs the optimized **stability verification algorithm**.
- `alg2_opt.py` — Runs the optimized **radius computation algorithm**.
- `par2.py` — Contains the corresponding implementations for the **other aggregation functions**.

## Construction Problem

For the `COUNT` aggregation function:

- `closest.py` — Contains the implementation of the **closest stable binning algorithm**.
- `run_closest.py` — Runs the closest stable binning algorithm using the `COUNT` aggregation function.

For the other aggregation functions:

- `closest_gl.py` — Contains the implementations of the closest stable binning algorithm for the other aggregation functions.

## Plotting and Figures

The `draw.py` file contains the Python code used to generate the graphs and figures presented in the paper.

## Experiments

This repository includes the code required to reproduce the experiments presented in the paper.

The experimental scripts contain the main execution functions and allow users to modify the relevant parameters to evaluate the algorithms under different settings.

### Equal-Width, Equal-Depth, and Jenks Binning

The `base_exp.py` file contains the experiments used to compare our algorithm with **equal-width binning**, **equal-depth binning**, and **Jenks binning** on the Census dataset.

These experiments also include an evaluation of **data reconstruction**.

### Runtime and Distance Experiments

The following scripts are used to evaluate runtime and distance under different parameter settings:

- `changing_c.py` — Runs experiments with varying values of `c` for the `COUNT` aggregation function.
- `changing_k.py` — Runs experiments with varying values of `k` for the `COUNT` aggregation function.
- `changing_eps.py` — Runs experiments with varying values of `ε` for the `COUNT` aggregation function.
- `exp.py` — Runs the corresponding experiments for the other aggregation functions.

### Runtime Comparison

The `m.py` file contains the code used to generate datasets with different domain sizes and compare the runtime performance of the **graph-based algorithm** and the **filtering-based algorithm**.

## Results

The `results_csv/` directory contains the experimental results in CSV format for all experiments presented in the paper.

These files can be used to reproduce the figures by running the `draw.py` script.
