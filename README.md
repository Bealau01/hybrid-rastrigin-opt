**Hybrid Rastrigin Optimization**

This repository implements a two-phase hybrid optimization framework designed to minimize the Rastrigin benchmark function. The approach combines global exploration capabilities of metaheuristics with the local refinement precision of gradient-based methods.


The optimization process is divided into two sequential stages:

1. Metaheuristic Phase (Global Search): Utilizing PSO (Particle Swarm Optimization) and CBO (Consensous Based Optimization) to navigate the multi-modal landscape of the Rastrigin function and avoid local minima.

2. Gradient-Based Phase (Local Search): Fine-tuning the results using SGD (Stochastic Gradient Descent) and Adam.

For a comprehensive analysis of the methodology, hyperparameter tuning, and experimental results, please refer to the full technical report "progetto_MOS".
