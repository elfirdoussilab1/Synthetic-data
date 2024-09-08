# In this file, we will generate the safety score plots 
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"text.usetex": True,"font.family": "STIXGeneral"})#,"font.sans-serif": "Helvetica",})

fig, ax = plt.subplots(1, 2, figsize = (15, 4), sharey= True)
linewidth = 3
fontsize = 20
labelsize = 17
s = 170
n = 5000

epsilons = [0.1, 0.5]
# Proportions of synthetic data
pis = [0, 0.58, 0.74, 0.81, 0.85, 0.88]

# Scores with strong supervision
scores_strong = {epsilons[0]: [0.66, 0.76, 0.87, 0.95, 0.98, 0.99], 
                 epsilons[1]: [0.66, 0.71, 0.75, 0.82, 0.88, 0.93]}

# Scores with weak supervision
scores_weak = {epsilons[0]: [0.66, 0.70, 0.77, 0.82, 0.88, 0.93],
               epsilons[1]: [0.66, 0.66, 0.654, 0.657, 0.68, 0.66]}

for i, eps in enumerate(epsilons):
    # Each plot will contain weak vs strong supervision
    # Strong supervision
    ax[i].plot(pis, scores_strong[eps], linewidth = linewidth, color = 'tab:green', label = 'Strong supervison')
    #ax[i].scatter(pis, scores_strong[eps], edgecolors = 'tab:green', s = s, marker = 'o', facecolors='none')
    ax[i].scatter(pis, scores_strong[eps], color = 'tab:green', s = s, marker = 'o')

    # Weak supervision
    ax[i].plot(pis, scores_weak[eps], linewidth = linewidth, color = 'tab:red', label = 'Weak Supervision')
    #ax[i].scatter(pis, scores_weak[eps], edgecolors='tab:red', s = s, marker = 'o', facecolors='none')
    ax[i].scatter(pis, scores_weak[eps], color='tab:red', s = s, marker = 'o')

    ax[i].set_ylabel('Safety Score', fontsize = fontsize)
    ax[i].set_xlabel('Proportion of Synthetic data', fontsize = fontsize)
    ax[i].set_title(f'$\\varepsilon = {eps}$', fontsize = fontsize)
    ax[i].tick_params(axis = 'x', which = 'both', labelsize = labelsize)
    ax[i].tick_params(axis = 'y', which = 'both', labelsize = labelsize)
    ax[i].grid(True)

# Legend
ax[0].legend(fontsize = labelsize)

path = './study-plot' + f'/safety-scores-plot-n-{n}.pdf'
fig.savefig(path, bbox_inches='tight')
