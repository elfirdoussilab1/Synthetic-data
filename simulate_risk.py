# In this file, we will show that theoretical test risk matchs well the empirical one
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from rmt_results import *
from utils import *

plt.rcParams.update({"text.usetex": True,"font.family": "STIXGeneral"})#,"font.sans-serif": "Helvetica",})
# Parameters
n = 1000
m = 2000
p = 500
mu = 1.5
mu_orth = 1
beta = 0.7
epsilon = 0.3
rho = 0.1
phi = 0.8

# Vectors
vmu = np.zeros(p)
vmu[0] = mu
vmu_orth = np.zeros(p)
vmu_orth[1] = mu_orth
vmu_beta = beta * vmu + vmu_orth

# Covariance matrix
fix_seed(123)
x = np.random.randn(n, p)
cov = x.T @ x / n
eigvals, eigvectors = np.linalg.eig(cov)

batch = 10
gammas = np.logspace(-6, 3, 20)
means_practice = []
means_theory = []
for gamma in tqdm(gammas):
    # Theory
    means_theory.append(test_risk(n, m, p, vmu, vmu_beta, cov, eigvals, eigvectors, epsilon, rho, phi, gamma))

    # Empirical
    means_practice.append(empirical_risk(batch, n, m, p, vmu, vmu_beta, cov, epsilon, rho, phi, gamma, data_type = 'synthetic'))

# Plotting results
fig, ax = plt.subplots()
ax.semilogx(gammas, means_theory, label = 'Theory', color = 'purple', linewidth = 2.5)
ax.scatter(gammas, means_practice, label = 'Simulation', marker = 'D', alpha = .7, color = 'green')
ax.set_xlabel('$\gamma$')
ax.set_ylabel('Test Risk')
ax.grid(True)
ax.legend()
path = './study-plot/' + f'simulate_risk-cov-n-{n}-m-{m}-p-{p}-beta-{beta}-mu-{mu}-mu_orth-{mu_orth}-epsilon-{epsilon}-rho-{rho}-phi-{phi}.pdf'
fig.savefig(path, bbox_inches='tight')