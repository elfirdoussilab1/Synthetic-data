# In this file, we will prove that theory mathces empirical values using a real dataset: Amazon Review
import numpy as np
import matplotlib.pyplot as plt
from dataset import Amazon
from utils import *
from rmt_results import *
plt.rcParams.update({"text.usetex": True,"font.family": "STIXGeneral"})#,"font.sans-serif": "Helvetica",})


# Parameters
n = 1500
m = 5000
p = 400
gamma = 1
epsilon = .2
rho = .2
phi = .9

names = ['books', 'elec', 'dvd']

fix_seed(1337)

fig, ax = plt.subplots(1, 3, figsize = (30, 6))
fontsize = 40
labelsize = 35
linewidth = 3

for i, name in enumerate(names):
    data = Amazon(n, name)

    # Test Data
    X_test = data.X_test
    y_test = data.y_test
    mu = data.mu

    # Real data
    X_r = data.X_r
    y_r = data.y_r

    X_s, y_s, vmu_hat, vq, y_tilde = data.generate_synth_data(m, epsilon, rho, phi)

    # Theory
    # Expectation of class C_1 and C_2
    mean_c2 = test_expectation_toy(n, m, p, mu, epsilon, rho, phi, gamma)
    mean_c1 = - mean_c2
    expec_2 = test_expectation_2_toy(n, m, p, mu, epsilon, rho, phi, gamma)
    std = np.sqrt(expec_2 - mean_c2**2)

    # Classifier
    w = classifier_vector(X_r.T, y_r, X_s.T, y_tilde, vq, gamma)

    t1 = np.linspace(mean_c1 - 5*std, mean_c1 + 5*std, 100)
    t2 = np.linspace(mean_c2 - 5*std, mean_c2 + 5*std, 100)


    # Plot all
    ax[i].plot(t1, gaussian(t1, mean_c1, std), color = 'tab:red', linewidth= linewidth)
    ax[i].plot(t2, gaussian(t2, mean_c2, std), color = 'tab:blue', linewidth= linewidth)
    ax[i].set_xlabel('$\\mathbf{w}_q^\\top \\mathbf{x}$', fontsize = fontsize)

    # Plotting histogram
    ax[i].hist(X_test[y_test < 0] @ w, color = 'tab:red', density = True, bins=25, alpha=.5, edgecolor = 'black')
    ax[i].hist(X_test[y_test > 0] @ w, color = 'tab:blue', density = True, bins=25, alpha=.5, edgecolor = 'black')
    ax[i].tick_params(axis = 'x', which = 'both', labelsize = labelsize)
    ax[i].tick_params(axis = 'y', which = 'both', labelsize = labelsize)
    # Label: label = '$\mathcal{C}_2$'
    ax[i].set_title(f'{name.upper()}', fontsize = fontsize)

ax[0].set_ylabel(f'Density', fontsize = fontsize)

path = './study-plot' + f'/distribution-amazon-toy-n-{n}-m-{m}-p-{p}-mu-{mu}-epsilon-{epsilon}-rho-{rho}-phi-{phi}.pdf'
fig.savefig(path, bbox_inches='tight')

