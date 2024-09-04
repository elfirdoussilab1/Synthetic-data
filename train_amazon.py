# In this file, we will plot the evolution of the Test accuracy with 1 - pi on Amazon Review dataset
import numpy as np
import matplotlib.pyplot as plt
from dataset import Amazon
from utils import *
from rmt_results import *
from tqdm.auto import tqdm

plt.rcParams.update({"text.usetex": True,"font.family": "STIXGeneral"})#,"font.sans-serif": "Helvetica",})

# Parameters
n = 1000
p = 400
gamma = 1e-1
epsilon = .2
rho = 0.
phi = 1.

names = ['books', 'elec', 'kitchen']
ms = np.array([1, n//4, n//2, n, 2*n, 5*n, 10*n, 15*n])

fig, ax = plt.subplots(1, 3, figsize = (30, 6))
fontsize = 40
labelsize = 35
linewidth = 4
s = 150

seeds = [1, 2, 123, 404]

pis = ms / (n + ms)

theory = False

for i, name in enumerate(names):
    accs = []
    for seed in seeds:
        accs_seed = []
        fix_seed(seed)
        data = Amazon(n, name)

        for m in tqdm(ms):
            # Test Data
            X_test = data.X_test
            y_test = data.y_test
            mu = data.mu
            # Real data
            X_r = data.X_r
            y_r = data.y_r

            # Synthetic data
            X_s, y_s, vmu_hat, vq, y_tilde = data.generate_synth_data(m, epsilon, rho, phi)

            # Classifier
            w = classifier_vector(X_r.T, y_r, X_s.T, y_tilde, vq, gamma)
            if theory:
                accs_seed.append(test_accuracy_toy(n, m, p, mu, epsilon, rho, phi, gamma))
            else:
                accs_seed.append(accuracy(y_test, decision(w, X_test.T)))
            
        
        accs.append(accs_seed)
    
    accs = np.array(accs)
    assert accs.shape == (len(seeds), len(ms))

    # Plotting results
    ax[i].plot(pis, np.mean(accs, axis = 0), linewidth = linewidth)
    ax[i].scatter(pis, np.mean(accs, axis = 0), marker = 'D', s = s, color = 'tab:green', alpha = .7)
    ax[i].fill_between(pis,  np.mean(accs, axis = 0) - np.std(accs, axis = 0), np.mean(accs, axis = 0) + np.std(accs, axis = 0),
                    alpha = 0.3, linestyle = '-.', color = 'tab:orange')
    
    ax[i].set_title(f'{name.upper()}', fontsize = fontsize)
    ax[i].set_xlabel('$1 - \pi$', fontsize = fontsize)
    ax[i].tick_params(axis='x', which = 'both', labelsize=labelsize)
    ax[i].tick_params(axis='y', which = 'both', labelsize=labelsize)
    ax[i].grid(True)

title = 'Oracle Supervision' if rho == 0. else 'Weak Supervision'
ax[0].set_ylabel(title, fontsize = fontsize)
path = f'./study-plot/train_amazon-n-{n}-p-{p}-gamma-{gamma}-epsilon-{epsilon}-rho-{rho}-phi-{phi}-theory-{theory}.pdf'
fig.savefig(path, bbox_inches='tight')
