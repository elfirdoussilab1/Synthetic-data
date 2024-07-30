# In this file, we will produce some accuracy plots and see their evolution with m, epsilon or beta
import numpy as np
from utils import *
import matplotlib.pyplot as plt
from rmt_results import *
from tqdm.auto import tqdm

plt.rcParams.update({"text.usetex": True,"font.family": "STIXGeneral"})#,"font.sans-serif": "Helvetica",})

# Parameters
n = 100
m = 100000
p = 500
pi = n / (n + m)
sigma = 1.5
rho = 0.2
phi = 0.9
gamma = 1
mu = 1
mu_orth = 0.8

betas = [0.7, 0.8, 0.99]
linewidth = 3
fontsize = 40
labelsize = 35
s = 200

seed = 1337

fig, ax = plt.subplots(1, 3, figsize = (30, 6))
epsilons = np.linspace(0, 1, 100)

# Accuracy for training with only real data
epsilon_critik = 1 / (1 + rho / phi)
for i, beta in enumerate(betas):
    accs = []
    for epsilon in tqdm(epsilons):
        accs.append(test_accuracy(n, m, p, mu, mu_orth, beta, sigma, epsilon, rho, phi, gamma))
    
    acc_critik = test_accuracy(n, m, p, mu, mu_orth, beta, sigma, epsilon_critik, rho, phi, gamma)

    # Plotting
    ax[i].plot(epsilons, accs, linewidth = linewidth, color = 'tab:blue')
    #ax[i].plot([epsilons[0], epsilons[-1]], [acc_real, acc_real], linewidth = linewidth, color = 'tab:orange', linestyle = '-.')
    ax[i].scatter( epsilon_critik, acc_critik, color = 'tab:red', s = s, marker = 'D')
    #ax[i].scatter(m_min / (m_min + n), np.min(accs), color = 'tab:red', s = s, marker = 'D')
    sentence_critik = f'$\\bar \\varepsilon = {round(epsilon_critik, 3) } $'
    hx = 2e-2
    hy = 4e-3
    ax[i].text(epsilon_critik + hx, acc_critik - hy, sentence_critik, fontsize = fontsize - 15)
    ax[i].set_title(f'$\\beta = {beta} $', fontsize = fontsize)
    ax[i].set_xlabel('$\\varepsilon$', fontsize = fontsize)
    ax[i].tick_params(axis='x', which = 'both', labelsize=labelsize)
    ax[i].tick_params(axis='y', which = 'both', labelsize=labelsize)
    ax[i].grid()

ax[0].set_ylabel(f'Test accuracy', fontsize = fontsize)

path = './study-plot/' + f'accuracy_epsilon-n-{n}-m-{m}-p-{p}-mu-{mu}-mu_orth-{mu_orth}-sigma-{sigma}-rho-{rho}-phi-{phi}.pdf'
fig.savefig(path, bbox_inches='tight')

