# This file is used to generate distribution plots to validate theoretical calculus
import numpy as np
from utils import *
import matplotlib.pyplot as plt
from rmt_results import *

plt.rcParams.update({"text.usetex": True,"font.family": "STIXGeneral"})#,"font.sans-serif": "Helvetica",})

# Parameters
n = 1000
m = 2000
p = 500
mu = 1.5
mu_orth = 1
epsilon = 0.6
rho = 0.1
phi = 0.8
gamma = 1
sigma = 2

# Vectors
vmu = np.zeros(p)
vmu[0] = mu
vmu_orth = np.zeros(p)
vmu_orth[1] = mu_orth

# Isotropic covariance matrix
fix_seed(404)
cov = sigma**2 * np.eye(p)
eigvals, eigvectors = np.linalg.eig(cov)

fig, ax = plt.subplots(1, 3, figsize = (30, 6))
fontsize = 40
labelsize = 35
linewidth = 3

betas = [0.2, 0.5, 0.8]
for i, beta in enumerate(betas):
    # Generate datasets
    vmu_beta = beta * vmu + vmu_orth
    (X_real, y_real, X_s, y_tilde, vq), (X_test, y_test) = generate_data(n, m, p, vmu, vmu_beta, cov, epsilon, rho, phi)

    # Expectation of class C_1 and C_2
    mean_c2 = test_expectation(n, m, p, vmu, vmu_beta, cov, eigvals, eigvectors, epsilon, rho, phi, gamma)
    mean_c1 = - mean_c2
    expec_2 = test_expectation_2(n, m, p, vmu, vmu_beta, cov, eigvals, eigvectors, epsilon, rho, phi, gamma)
    std = np.sqrt(expec_2 - mean_c2**2)

    # Classifier 
    w = classifier_vector(X_real, y_real, X_s, y_tilde, vq, gamma)
    t1 = np.linspace(mean_c1 - 5*std, mean_c1 + 5*std, 100)
    t2 = np.linspace(mean_c2 - 5*std, mean_c2 + 5*std, 100)


    # Plot all
    ax[i].plot(t1, gaussian(t1, mean_c1, std), color = 'tab:red', linewidth= linewidth)
    ax[i].plot(t2, gaussian(t2, mean_c2, std), color = 'tab:blue', linewidth= linewidth)
    ax[i].set_xlabel('$\\mathbf{w}_q^\\top \\mathbf{x}$', fontsize = fontsize)

    # Plotting histogram
    ax[i].hist(X_test[:, (y_test < 0)].T @ w, color = 'tab:red', density = True, bins=25, alpha=.5, edgecolor = 'black')
    ax[i].hist(X_test[:, (y_test > 0)].T @ w, color = 'tab:blue', density = True, bins=25, alpha=.5, edgecolor = 'black')
    ax[i].tick_params(axis = 'x', which = 'both', labelsize = labelsize)
    ax[i].tick_params(axis = 'y', which = 'both', labelsize = labelsize)
    # Label: label = '$\mathcal{C}_2$'
    ax[i].set_title(f'$\\beta = {beta}$', fontsize = fontsize)

ax[0].set_ylabel(f'$ \\epsilon = {epsilon} $', fontsize = fontsize)

path = './study-plot' + f'/distribution-sigma-n-{n}-m-{m}-p-{p}-mu-{mu}-mu_orth-{mu_orth}-sigma-{sigma}-epsilon-{epsilon}-rho-{rho}-phi-{phi}.pdf'
fig.savefig(path, bbox_inches='tight')