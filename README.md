# Naloxone Kit Allocation Optimization — Rhode Island

**Optimizing the geographic distribution of naloxone kits across Rhode Island to minimize projected opioid overdose deaths, using a machine learning metamodel trained on a stochastic microsimulation.**

> **Note:** This repository is adapted from a private research codebase. It includes the full ML/optimization pipeline; the microsimulation model itself is a separate, proprietary research codebase and isn't included — see "Reproducibility" below for exactly what that means per notebook.

---

## Overview

Naloxone is a life-saving medication that can reverse opioid overdoses in minutes. With limited supply, the question of *where* to allocate naloxone kits across cities and towns is a high-stakes resource allocation problem. Rhode Island — one of the states hardest hit by the opioid crisis — serves as the study area for this work.

Evaluating any single allocation strategy requires running a computationally expensive stochastic microsimulation. This makes brute-force or even heuristic search over the full allocation space intractable. The core contribution of this project is a **metamodel-based optimization framework**: a neural network surrogate trained to emulate the simulation's input-output relationship, enabling efficient optimization over the full combinatorial space.

This work is part of a forthcoming research paper (citation to be added upon publication).

---

## Methodology

```
Sobol Sequence         Stochastic            Neural Network         Constrained
Sampling           →   Microsimulation   →   Metamodel          →   Optimization
(Allocation Vectors)   (Projected Deaths)    (Fast Surrogate)       (Minimize Deaths)
```

1. **Sample Generation** — Sobol sequence sampling (via `scipy.stats.qmc`) produces a diverse set of naloxone allocation vectors across RI's 39 cities and towns, subject to a fixed statewide budget constraint (50,000 kits).

2. **Simulation** — Each allocation vector is evaluated through a calibrated stochastic microsimulation that models opioid use trajectories, overdose events, and naloxone intervention at the individual level. Simulation jobs are parallelized on an HPC cluster via SLURM.

3. **Metamodel Training** — A fully connected neural network (5 layers, ReLU activations) is trained on the simulation input-output pairs to learn the mapping from allocation → projected overdose deaths. The trained metamodel serves as a fast, differentiable surrogate for the simulation.

4. **Optimization** — The metamodel is used as the objective function in `scipy.optimize.minimize` (trust-constr method) with an equality constraint preserving the total kit budget. Multiple restarts from diverse initial points guard against local minima.

---

## Key Results

| Approach | Projected Overdose Deaths (2026) |
|---|---|
| Status Quo (Current Distribution) | 362.93 |
| Greedy Algorithm | 342.79 |
| **Metamodel Optimization** | **342.26 – 343.03** |

The metamodel-based approach achieves performance comparable to a greedy algorithm while searching over the full combinatorial allocation space — and provides a generalizable framework applicable to other resource allocation problems in public health.

---

## Technical Stack

- **Simulation:** R (microsimulation model — proprietary, not included)
- **Metamodel & Optimization:** Python (NumPy, SciPy, scikit-learn, joblib)
- **Sampling:** Sobol sequence sampling via `scipy.stats.qmc`
- **Compute:** SLURM-based HPC for simulation parallelization
- **Analysis:** Jupyter notebooks, pandas, matplotlib, seaborn

---

## Repository Structure

```
├── notebooks/
│   ├── 00_status_quo_distribution.ipynb  # Derive the current distribution baseline
│   ├── 01_generate_samples.ipynb         # Sobol sequence allocation sampling
│   ├── 02_obtain_simulation_outputs.ipynb # Aggregate HPC simulation results
│   ├── 03_train_metamodel.ipynb          # Train and validate neural network surrogate
│   ├── 04_optimize.ipynb                 # Run constrained optimization over metamodel
│   ├── 05_find_optimal_solutions.ipynb   # Extract and evaluate optimal allocations
│   ├── 06_greedy_benchmark.ipynb         # Greedy algorithm baseline comparison
│   └── 07_compare_results.ipynb          # Final comparison across all approaches
│
├── src/
│   └── optimization.py                   # Batch optimization with SLURM integration
│
├── data/
│   ├── inputs/                           # RI demographic data (real input)
│   └── generated/                        # Sample sets and starting points produced by the notebooks
│
├── results/
│   ├── figures/                          # Visualization outputs
│   └── outputs/                          # CSV results for each approach
│
└── environment.yml                       # Conda environment specification
```

---

## Reproducibility

The stochastic microsimulation model used to generate training data is the intellectual property of the original simulation authors and is not included in this repository. The notebooks here cover all downstream steps — metamodel training, optimization, benchmarking, and results analysis — which represent the novel methodological contributions of this work.

Reproducibility varies by notebook, and each one states its own dependencies up front:

- `01_generate_samples.ipynb` and `07_compare_results.ipynb` run fully locally with `seed=42` and the data included in this repo.
- `00_status_quo_distribution.ipynb` depends on the microsimulation model's own input data, which lives outside this repo — it's included as a read-through of the methodology.
- `02_obtain_simulation_outputs.ipynb`, and portions of `04_optimize.ipynb`, `05_find_optimal_solutions.ipynb`, and `06_greedy_benchmark.ipynb` depend on batch outputs from a university HPC cluster. Those cells are clearly marked and provided for documentation of the exact procedure, not local execution.

---

## About

**Ryan Kuntz**
BA, Applied Mathematics — Brown University
MS, Data Analytics — Georgia Institute of Technology

This project represents my contributions as second author on a forthcoming paper applying machine learning and optimization to opioid crisis intervention policy. My work focused on the metamodel architecture, the optimization pipeline, and the comparative analysis framework.

[LinkedIn](https://www.linkedin.com/in/ryan-kuntz-8621502a3/)

---

## Status

✅ **Complete** — The ML/optimization pipeline shown here (metamodel training, optimization, benchmarking, and results analysis) is finished and reflects the methodology and results described above. The proprietary microsimulation component is intentionally excluded; see "Reproducibility" for what that means for running this repo locally.
