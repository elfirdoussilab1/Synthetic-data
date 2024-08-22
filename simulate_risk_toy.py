# In this file, we will implement the toy setting described in the paper.
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from rmt_results import *
from utils import *

plt.rcParams.update({"text.usetex": True,"font.family": "STIXGeneral"})#,"font.sans-serif": "Helvetica",})
fix_seed(123)

# Model Parameters
n = 5000
m = 3000
p = 1000
mu = 1
vmu = np.random.randn(p)
vmu = vmu / np.linalg.norm(vmu) * mu

print("norm of vmu", np.linalg.norm(vmu))

# Pruning and labelling parameters
epsilon = 0.1
rho = 0.1
phi = 0.8

# Real Dataset
X_r, y_r = gaussian_mixture(n, vmu, None, real = True)

# Synthetic dataset
vmu_hat = np.sum(y_r * X_r, axis = 1) / n

# Measuring beta
beta = np.sum(vmu * vmu_hat) / mu**2

print(beta)

# Checking Test Risk
batch = 5
risk_emps = []
risk_theory = []
gammas = np.logspace(-6, 2, 20)

for gamma in tqdm(gammas):
    risk_emps.append(empirical_risk_toy(batch, n, m, p, vmu, X_r, y_r, epsilon, rho, phi, gamma))
    risk_theory.append(test_risk_toy(n, m, p, mu, epsilon, rho, phi, gamma))

# Plotting results
fig, ax = plt.subplots()
ax.semilogx(gammas, risk_theory, label = 'Theory', color = 'purple', linewidth = 2.5)
ax.scatter(gammas, risk_emps, label = 'Simulation', marker = 'D', alpha = .7, color = 'green')
ax.set_xlabel('$\gamma$')
ax.set_ylabel('Test Risk')
ax.grid(True)
ax.legend()
path = './study-plot/' + f'simulate_risk-toy-n-{n}-m-{m}-p-{p}-beta-{beta}-mu-{mu}-epsilon-{epsilon}-rho-{rho}-phi-{phi}.pdf'
fig.savefig(path, bbox_inches='tight')



