# Batch optimization script, run in parallel on an HPC cluster via SLURM
# (one job per batch, selected by SLURM_ARRAY_TASK_ID) as part of
# 04_optimize.ipynb's local-optimization step. It loads the metamodel trained
# in 03_train_metamodel.ipynb (model_weights.pkl, scaler.pkl — not included in
# this public repo, see the root README) and the starting points generated in
# 04_optimize.ipynb (x0_list.npy), then runs one gradient-based optimization
# per starting point. Each job's results are combined back in 04_optimize.ipynb
# into results/outputs/Local_Minima.csv.

import numpy as np
import joblib
import os
import csv
from scipy.optimize import minimize

# Load model weights and scaler
weights = joblib.load('model_weights.pkl')
scaler = joblib.load('scaler.pkl')

# Load list of 1000 x0 vectors
x0_list = np.load('x0_list.npy', allow_pickle=True)

def relu(x):
    return np.maximum(0, x)

# Mirrors the architecture trained in 03_train_metamodel.ipynb: 5 Dense layers
# (256 -> 128 -> 64 -> 32 -> 1) with ReLU activations; Dropout layers carry no
# weights, so they don't appear here.
def forward_pass(x, weights):
    x = scaler.transform([x])[0]

    w1, b1 = weights[0], weights[1]
    x = relu(np.dot(x, w1) + b1)

    w2, b2 = weights[2], weights[3]
    x = relu(np.dot(x, w2) + b2)

    w3, b3 = weights[4], weights[5]
    x = relu(np.dot(x, w3) + b3)

    w4, b4 = weights[6], weights[7]
    x = relu(np.dot(x, w4) + b4)

    w5, b5 = weights[8], weights[9]
    output = np.dot(x, w5) + b5

    return output[0]

def objective(x):
    return forward_pass(x, weights)

def run_batch_optimization(batch_id, batch_size=10):
    input_dim = weights[0].shape[0]
    total_tasks = len(x0_list)
    start_idx = (batch_id - 1) * batch_size
    end_idx = min(start_idx + batch_size, total_tasks)

    if start_idx >= total_tasks:
        raise IndexError(f"Batch ID {batch_id} is out of range.")

    results = []
    for i in range(start_idx, end_idx):
        x0 = x0_list[i]

        constraints = {
            'type': 'eq',
            'fun': lambda x: np.sum(x) - 50000
        }

        bounds = [(0, None) for _ in range(input_dim)]

        result = minimize(
            objective,
            x0,
            method='trust-constr',
            bounds=bounds,
            constraints=constraints
        )

        results.append([result.fun] + result.x.tolist())

    # Save all 10 results in one CSV file
    csv_filename = f'result_batch_{batch_id}.csv'
    with open(csv_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(results)

if __name__ == '__main__':
    batch_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 1))
    run_batch_optimization(batch_id)

