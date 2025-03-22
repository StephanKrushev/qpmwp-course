# Import standard libraries
import os
import sys
import timeit # To compute runtimes
from typing import Optional
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp




# Import third-party libraries


# Import local modules
#project_root = os.path.dirname(os.path.dirname(os.getcwd()))   # Change this path if needed

project_root =os.getcwd()
print(project_root)


src_path = os.path.join(project_root, 'qpmwp-course/src')
sys.path.append(project_root)

print(src_path)

sys.path.append(src_path)
from estimation.covariance import Covariance
from estimation.expected_return import ExpectedReturn
from optimization.constraints import Constraints
from optimization.optimization import Optimization, Objective
from optimization.optimization_data import OptimizationData
from optimization.quadratic_program import QuadraticProgram, USABLE_SOLVERS
print(USABLE_SOLVERS) # edit in quaratic_program.py to add more solvers or subtract if needed


USABLE_SOLVERS.remove('quadprog')
USABLE_SOLVERS.remove('daqp')
USABLE_SOLVERS.remove('osqp')
USABLE_SOLVERS.remove('qpalm')
# USABLE_SOLVERS.remove('cvxopt')

print(USABLE_SOLVERS)

def nearest_psd_eig(A):
    # Compute the eigenvalue decomposition
    eigvals, eigvecs = np.linalg.eigh(A)
    # Set negative eigenvalues to zero (or a small positive value)
    eigvals_clipped = np.clip(eigvals, a_min=0, a_max=None)
    # Reconstruct the matrix
    A_psd = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T
    return A_psd


# Set the dimensions
T = 10000  # Number of time periods
# loop over  Ns = [64, 128, 256, 512, 1024, 2048, 4096] # Number of assets

#for N in [64, 128, 256, 512, 1024, 2048, 4096]:
total_results = pd.DataFrame()
for N in [64,128, 256, 512, 1024, 2048, 4096, 8192]:
    print(f"Number of assets: {N}")
    # Generate a random mean vector and covariance matrix for the multivariate normal distribution

    # Set the random seed for reproducibility
    np.random.seed(0)

    # Generate the mean vector and covariance matrix
    print('Generating random mean vector and covariance matrix...')
    mean = np.random.randn(N) * 0.01 
    random_matrix = np.random.randn(N, N)*0.01
    cov = random_matrix @ random_matrix.T


    # Generate the Multivariate-Normal random dataset
    data = np.random.multivariate_normal(mean, cov, size=T)

    # Convert the dataset to a DataFrame for easier manipulation
    df = pd.DataFrame(data, columns=[f'Asset_{i+1}' for i in range(N)])

    # Compute the vector of expected returns (mean returns) from df
    q = np.exp(np.log(1 + df).mean(axis=0)) - 1

    # Compute the covariance matrix from df
    P = np.cov(data, rowvar=False)
    print('Make the covariance matrix positive definite...')
    if N == 4096:
        P = nearest_psd_eig(P)
    print('Next steps...')
    # Instantiate the Constraints class
    constraints = Constraints(ids = df.columns.tolist())

    # Add budget constraint
    constraints.add_budget(rhs=1, sense='=')

    # Add box constraints (i.e., lower and upper bounds)
    constraints.add_box(lower=0, upper=0.2)

    # Add linear constraints
    G = pd.DataFrame(np.zeros((3, N)), columns=constraints.ids)
    G.iloc[0, 0:30] = 1
    G.iloc[1, 30:60] = 1
    G.iloc[2, 60:N] = 1
    h = pd.Series([0.3, 0.4, 0.5])
    constraints.add_linear(G=G, sense='<=', rhs=h)

    # Extract the constraints in the format required by the solver
    GhAb = constraints.to_GhAb()

    # Loop over solvers, instantiate the quadratic program, solve it and store the results
    risk_aversion = 3
    # initialize output dictionary
    results = {}


    for solver in USABLE_SOLVERS:
        qp = QuadraticProgram(
            P = P * risk_aversion,
            q = q.to_numpy() * -1,
            G = GhAb['G'],
            h = GhAb['h'],
            A = GhAb['A'],
            b  = np.array([GhAb['b']]),
            lb = constraints.box['lower'].to_numpy(),
            ub = constraints.box['upper'].to_numpy(),
            solver = solver,
        )
        start_time = time.time()
        qp.solve()
        end_time = time.time()
        elapsed_time = end_time - start_time
        solution = qp.results.get('solution')
        print(f"{solver}: {qp.objective_value()}")
        print(f"Elapsed time: {elapsed_time}")
        # store the results in a dict of dicts
        results[solver] = {
            'N': N,
            'is_feasible': qp.is_feasible(),
            'objective_value': qp.objective_value(),
            'solution_found': solution.found,
            'primal_residual': solution.primal_residual(),
            'dual_residual': solution.dual_residual(),
            'duality_gap': solution.duality_gap(),
            'elapsed_time': elapsed_time,
        }

    # print the results in a table format
    results_df = pd.DataFrame(results).T
    print(results_df)
    #add the results to total results
    if N == 64:
        total_results = results_df
    else:
        total_results = pd.concat([total_results, results_df])

    print("\n\n")

print(total_results)
# store the results in a file
total_results.to_csv('scaling_T_N_top4.csv')



# Plot the results, using different symbols for dots for different solvers

##############

markers = ['o', 's', '^', 'D', 'v', '*', 'P', 'X']  # adjust as needed

fig, ax = plt.subplots(figsize=(10, 6))

for i, solver in enumerate(USABLE_SOLVERS):
    marker = markers[i % len(markers)]  # cycle through markers if necessary
    ax.plot(
        total_results.loc[solver, 'N'],
        total_results.loc[solver, 'elapsed_time'],
        label=solver,
        marker=marker,
        linestyle='',  # if you only want markers without connecting lines
    )

ax.set_xlabel('Number of assets')
ax.set_ylabel('Elapsed time (s)')
ax.set_title('Elapsed time vs. Number of assets')
ax.legend()

plt.show()

# export the plot to a file
fig.savefig('scaling_T_N_top4.png')

# make a log-log plot
fig, ax = plt.subplots(figsize=(10, 6))

for i, solver in enumerate(USABLE_SOLVERS):
    marker = markers[i % len(markers)]  # cycle through markers if necessary
    ax.plot(
        total_results.loc[solver, 'N'],
        total_results.loc[solver, 'elapsed_time'],
        label=solver,
        marker=marker,
        linestyle='',  # if you only want markers without connecting lines
    )

ax.set_xlabel('Number of assets')
ax.set_ylabel('Elapsed time (s)')
ax.set_title('Elapsed time vs. Number of assets')
ax.legend()
ax.set_xscale('log')
ax.set_yscale('log')

plt.show()

# save the log-log plot to a file
fig.savefig('scaling_T_N_loglog_top4.png')