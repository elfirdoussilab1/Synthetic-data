# In this file, we will show that theoretical test accuracy matchs well the empirical one
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from rmt_results import *
from utils import *

plt.rcParams.update({"text.usetex": True,"font.family": "STIXGeneral"})#,"font.sans-serif": "Helvetica",})
# Parameters
n = 200
m = 2000
p = 500
mu = 1.5
mu_orth = 1
sigma = 0.7
beta = 0.7
epsilon = 0.3
rho = 0.1
phi = 0.8

batch = 10
gammas = np.logspace(-6, 3, 20)
means_practice = []
means_theory = []
for gamma in tqdm(gammas):
    # Theory
    means_theory.append(test_accuracy(n, m, p, mu, mu_orth, beta, sigma, epsilon, rho, phi, gamma))

    # Empirical
    means_practice.append(empirical_accuracy(batch, n, m, p, mu, mu_orth, beta, sigma, epsilon, rho, phi, gamma, data_type = 'synthetic'))

# Plotting results
fig, ax = plt.subplots()
ax.semilogx(gammas, means_theory, label = 'Theory', color = 'purple', linewidth = 3)
ax.scatter(gammas, means_practice, label = 'Simulation', marker = 'D', alpha = .7, color = 'green')
ax.set_xlabel('$\gamma$')
ax.set_ylabel('Test Accuracy')
ax.grid(True)
ax.legend()
path = './results-plot/' + f'simulate_accuracy-n-{n}-m-{m}-p-{p}-beta-{beta}-mu-{mu}-mu_orth-{mu_orth}-sigma-{sigma}-epsilon-{epsilon}-rho-{rho}-phi-{phi}.pdf'
fig.savefig(path, bbox_inches='tight')