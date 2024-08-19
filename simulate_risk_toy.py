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
batch = 1
risk_emps = []
risk_theory = []
gammas = np.logspace(-6, 2, 20)

for gamma in tqdm(gammas):
    res = 0
    for i in range(batch):
        #X, y = gaussian_mixture(n, vmu, None, real = True) # As if we have other samples ! Divide our dataset by 2
        #vmu_hat = np.sum(y_r * X_r, axis = 1) / n
        #C = (vmu_hat * np.ones((n, p)) ).T
        #cov = (y * X - C) @ (y * X - C).T / n
        # covariance
        Z = np.random.randn(p, n)
        cov = Z @ Z.T / n
        #C = (vmu_hat * np.ones((n, p)) ).T 
        #cov = (y_r * X_r - C) @ (y_r * X_r - C).T / n

        # Synthetic dataset
        (X_real, y_real, X_s, y_tilde, vq), (X_test, y_test) = generate_data(n, m, p, vmu, vmu_hat, cov, epsilon, rho, phi)
        
        # Classifier
        w = classifier_vector(X_r, y_r, X_s, y_tilde, vq, gamma)
        res += L2_loss(w, X_test, y_test)
    risk_emps.append(res / batch)
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



