import math
import multiprocessing
import numpy as np
import pandas as pd

from base import create_solver

real_lambdas = [6.21200099234, 15.76823502672, 28.31967901332]
N_range = (1, 201)


def vary_N(real_lambda):
    low = math.floor(real_lambda)
    high = math.ceil(real_lambda)
    lambdas = []
    for N in range(*N_range):
        if N % 10 == 0:
            print(f'lambda={real_lambda}, N={N}')
        solver = create_solver(N, min(N, 50), N)
        try:
            lambdas.append(solver.find_lambda(low, high))
        except Exception:
            lambdas.append(np.nan)
    return lambdas


def main():
    with multiprocessing.Pool(3) as pool:
        lambda1_by_N, lambda2_by_N, lambda3_by_N = pool.map(
            vary_N, real_lambdas)
    df = pd.DataFrame({
        'N': np.arange(*N_range),
        'lambda1': np.array(lambda1_by_N),
        'lambda2': np.array(lambda2_by_N),
        'lambda3': np.array(lambda3_by_N),
    })
    df.to_csv('equispacing_min_50.py', index=False)


if __name__ == '__main__':
    main()
