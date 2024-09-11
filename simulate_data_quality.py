# In this file, we will generate a plot that shows the performance of the toy setting by varying n_hat
import numpy as np
from utils import *
import matplotlib.pyplot as plt
from rmt_results import *
from tqdm.auto import tqdm

plt.rcParams.update({"text.usetex": True,"font.family": "STIXGeneral"})#,"font.sans-serif": "Helvetica",})

# Parameters
n = 1000
mu = 0.7
gamma = 1e-1
batch = 10
p = 500
epsilon = 0.3

n_hats = [n//10, n//5, n//2, n]
pis = np.linspace(0.001, 0.95, 50)

fig, ax = plt.subplots(1, 2, figsize = (15, 4))
linewidth = 3
fontsize = 20
labelsize = 17

vmu = np.random.randn(p)
vmu = vmu / np.linalg.norm(vmu) * mu

# Real Dataset
X_r, y_r = gaussian_mixture(n, vmu, None, real = True)

# Estimating the mean
vmu_hat = np.sum(y_r * X_r, axis = 1) / n

# Estimating covariance
C = (vmu * np.ones((n, p)) ).T
cov = (y_r * X_r - C) @ (y_r * X_r - C).T / n
#Z = np.random.randn(p, n)
#cov = Z @ Z.T / n

# eigenvalues and eigenvectors
eigvals, eigvectors = np.linalg.eig(cov)

# Measuring beta
beta = np.sum(vmu * vmu_hat) / mu**2
print(beta)

for n_hat in n_hats:
    acc_oracle_th = [] # Oracle supevision: phi = 1, rho = 0
    acc_oracle_emp = []
    acc_weak_th = [] # No supervision: phi = 1, rho = 1
    acc_weak_emp = []

    for pi in tqdm(pis):
        m = pi * n / (1 - pi) # pi is the proportion of synthetic data
        m = int(m)
        #epsilon = 1 - test_accuracy(0, m, p, vmu, vmu, cov, eigvals, eigvectors, 0, 0, 1, gamma)
        #print(f'epsilon = {epsilon} for m = {m}')

        # Oracle accuracy
        acc_oracle_th.append(test_accuracy_toy(n, n_hat, m, p, mu, epsilon, 0, 1, gamma))
        #acc_oracle_emp.append(empirical_accuracy_toy(batch, n, n_hat,m, p, vmu, X_r, y_r, epsilon, 0, 1, gamma))

        # Weak accuracy
        acc_weak_th.append(test_accuracy_toy(n, n_hat, m, p, mu, epsilon, 1, 1, gamma))
        #acc_weak_emp.append(empirical_accuracy_toy(batch, n, n_hat, m, p, vmu, X_r, y_r, epsilon, 1, 1, gamma))

    # Plotting results
    # Oracle
    line, = ax[0].plot(pis, acc_oracle_th, label = f'$\hat n$ = {n_hat}', linewidth = 2.5)
    color = line.get_color()
    #ax[0].scatter(pis, acc_oracle_emp, marker = 'D', alpha = .7, color = color)
    # Weak 
    ax[1].plot(pis, acc_weak_th, label = f'$\hat n$ = {n_hat}', linewidth = 2.5, linestyle = '-.')
    #ax[1].scatter(pis, acc_weak_emp, marker = 'D', alpha = .7, color = color)

ax[0].set_ylabel('Test Accuracy', fontsize = fontsize)
ax[0].set_xlabel('$1 - \pi$', fontsize = fontsize)
ax[0].set_title('Oracle Supervision', fontsize = fontsize)

ax[1].set_xlabel('$1 - \pi$', fontsize = fontsize)
ax[1].set_title('Weak Supervision', fontsize = fontsize)
ax[1].set_ylabel('Test Accuracy', fontsize = fontsize)
ax[0].tick_params(axis = 'x', which = 'both', labelsize = labelsize)
ax[0].tick_params(axis = 'y', which = 'both', labelsize = labelsize)
ax[1].tick_params(axis = 'x', which = 'both', labelsize = labelsize)
ax[1].tick_params(axis = 'y', which = 'both', labelsize = labelsize)

ax[0].legend(fontsize = fontsize - 5)
ax[0].grid()
ax[1].grid()
path = './study-plot' + f'/toy-setting-n-{n}-p-{p}-mu-{mu}-gamma-{gamma}-batch-{batch}.pdf'
fig.savefig(path, bbox_inches='tight')
