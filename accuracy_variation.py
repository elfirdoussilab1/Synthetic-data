# In this file, we will produce some accuracy plots and see their evolution with m, epsilon or beta
import numpy as np
from utils import *
import matplotlib.pyplot as plt
from rmt_results import *
from tqdm.auto import tqdm

plt.rcParams.update({"text.usetex": True,"font.family": "STIXGeneral"})#,"font.sans-serif": "Helvetica",})

# Parameters
n = 500
p = 500
sigma = 1.
epsilon = 0.2
rho = 0.
phi = 1
gamma = 1
mu = 1
mu_orth = 0.8

betas = [0.7, 0.8, 0.99]
linewidth = 4
fontsize = 40
labelsize = 35
s = 250

seed = 1337

fig, ax = plt.subplots(1, 3, figsize = (30, 6))
ms = np.arange(0, 10001) * 100

# Accuracy for training with only real data
acc_real = test_accuracy(n, 0, p, mu, 0, 1, 1, epsilon, rho, phi, gamma)
for i, beta in enumerate(betas):
    accs = []
    for m in tqdm(ms):
        accs.append(test_accuracy(n, m, p, mu, mu_orth, beta, sigma, epsilon, rho, phi, gamma))
    
    # m_max, m_min
    m_max = ms[np.argmax(accs)]
    m_min = ms[np.argmin(accs)]
    ax[i].plot(ms / (ms + n), accs, linewidth = linewidth, color = 'tab:blue')
    ax[i].plot([ms[0] / (ms[0] + n), ms[-1] / (ms[-1] + n)], [acc_real, acc_real], linewidth = linewidth, color = 'tab:orange', linestyle = '-.')
    ax[i].scatter(m_max / (m_max + n) , np.max(accs), color = 'tab:green', s = s, marker = 'D')
    ax[i].scatter(m_min / (m_min + n), np.min(accs), color = 'tab:red', s = s, marker = 'D')
    sentence_max = f'$1 - \pi^* = {round(m_max / (m_max + n), 3) } $'
    sentence_min = f'$1 - \\bar \pi = {round(m_min / (m_min + n), 3)} $'
    hx = 4e-3
    hy = 4e-3
    ax[i].text(m_max / (m_max + n) + hx, np.max(accs) - hy, sentence_max, fontsize = fontsize - 10)
    ax[i].text(m_min / (m_min + n) - hx, np.min(accs) + hy, sentence_min, fontsize = fontsize - 10)
    ax[i].set_title(f'$\\beta = {beta} $', fontsize = fontsize)
    ax[i].set_xlabel('m', fontsize = fontsize)
    ax[i].tick_params(axis='x', which = 'both', labelsize=labelsize)
    ax[i].tick_params(axis='y', which = 'both', labelsize=labelsize)
    ax[i].grid()

ax[0].set_ylabel(f'Test accuracy, $\\varepsilon = {epsilon}$', fontsize = fontsize)

path = './study-plot/' + f'accuracy_var-n-{n}-p-{p}-mu-{mu}-mu_orth-{mu_orth}-sigma-{sigma}-epsilon-{epsilon}-rho-{rho}-phi-{phi}.pdf'
fig.savefig(path, bbox_inches='tight')

