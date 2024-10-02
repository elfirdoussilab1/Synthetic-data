# In this file, we will observe the evolution of the test accuracy with n_hat
import numpy as np
from utils import *
import matplotlib.pyplot as plt
from rmt_results import *
from tqdm.auto import tqdm

plt.rcParams.update({"text.usetex": True,"font.family": "STIXGeneral"})#,"font.sans-serif": "Helvetica",})

# Parameters
n = 6000
m = 4000
mu = 0.8
gamma = 1
epsilon = .2
rho = .1
phi = .9

ps = [100, 500, 1000]
n_hats = np.linspace(1, n, 100)

fig, ax = plt.subplots(1, 3, figsize = (30, 6))
linewidth = 4
fontsize = 40
labelsize = 35

for i, p in enumerate(ps):
    accs = []
    for n_hat in tqdm(n_hats):
        n_hat = int(n_hat)
        acc = test_accuracy_toy(n, n_hat, m, p, mu, epsilon, rho, phi, gamma)
        accs.append(acc)
    
    ax[i].plot(n_hats, accs)
    
plt.show()


