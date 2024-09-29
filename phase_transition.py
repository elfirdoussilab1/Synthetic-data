# In this file, we will demonstrate the smooth phase transition
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from rmt_results import *
from utils import *

plt.rcParams.update({"text.usetex": True,"font.family": "STIXGeneral"})#,"font.sans-serif": "Helvetica",})
fix_seed(123)

# Model Parameters
p = 100
mu = 1
gamma = 1

# Pruning and labelling parameters
rho = 0.2
phi = 0.9

eps_crit = 1 / (1 + rho / phi)
print(eps_crit)

epsilons = np.linspace(0, eps_crit, 50)
ms = [p, 10*p, 100*p]

fig, ax = plt.subplots(figsize = (6, 4))
#fig, ax = plt.subplots()
linewidth = 3
fontsize = 20
labelsize = 17
s = 100

for m in ms:
    accs = []

    for epsilon in tqdm(epsilons):
        accs.append(test_accuracy_synth(m, p, mu, epsilon, rho, phi, gamma))
    
    ax.plot(epsilons, accs, label = '$\\frac{p}{m} = $' + str(round(p/m, 3)), linewidth = linewidth, alpha = .8)


acc_critical = test_accuracy_synth(ms[-1], p, mu, eps_crit, rho, phi, gamma)
ax.scatter([eps_crit], [acc_critical], color = 'black', marker = '*', s = s)
pos_x = eps_crit - 0.15
pos_y = acc_critical 
ax.text(pos_x, pos_y, '$\\varepsilon^* $', fontsize = labelsize)
ax.arrow(pos_x + 0.05, pos_y, 0.05, 0, head_width=0.02, head_length=0.02, fc='black', ec='black')

ax.legend(fontsize = fontsize)
ax.set_xlabel('$\\varepsilon$', fontsize = fontsize)
ax.set_ylabel('Test Accuracy', fontsize = fontsize)
ax.grid(True)
ax.tick_params(axis = 'x', which = 'both', labelsize = labelsize)
ax.tick_params(axis = 'y', which = 'both', labelsize = labelsize)

path = './study-plot/' + f'phase_transition-p-{p}-mu-{mu}-rho-{rho}-phi-{phi}.pdf'
fig.savefig(path, bbox_inches='tight')


