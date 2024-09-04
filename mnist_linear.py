# In this file, we will test multi-class extension that was described in the appendix
import numpy as np
from utils import *
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from dataset import *

plt.rcParams.update({"text.usetex": True,"font.family": "STIXGeneral"})#,"font.sans-serif": "Helvetica",})

# Paramaters
n = 100
ms = [0, n//2,  2*n, 10*n, 20*n, 50*n]
#ms = [0, n//3, n//2,  2*n, 10*n, 20*n]
gamma = 10
threshold = 0.
p_estim = 0.
name = 'mnist'

# Whether to generate other non supervised data using gan or statistics
#gan = True

# plotting params
linewidth = 3
fontsize = 20
labelsize = 17

test_data = GAN_data(name, n, 0, 'cpu', train = False)
X_test = test_data.X
y_test = test_data.y.astype(int)

# One-Hot Encoding
Y_test = np.eye(10)[y_test]

fig, ax = plt.subplots(1, 2, figsize = (15, 4), sharey = True)

seeds = [1, 123, 404]

for gan in [True, False]:
    accs_oracle_all = []
    accs_weak_all = []
    for seed in seeds:
        fix_seed(seed)
        accs_oracle = []
        accs_weak = []

        for m in tqdm(ms):
        
            # Oracle Supervision
            if gan :
                train_data = MNIST_GAN(n, m, 'cpu', train = True, supervision = True, threshold= threshold)
            else:
                train_data = MNIST_generator(n, m, 'cpu', train = True, m_estim = int(p_estim * m), estimate_cov= True, supervision= True, threshold= threshold)
            #train_data = MNIST_GAN(n, m, 'cpu', train = True, supervision = True, threshold= threshold)
            X_r = train_data.X_real
            y_r = train_data.y_real.astype(int)
            X_s = train_data.X_s
            y_s = train_data.y_s.astype(int)

            # One-Hot encoding
            Y_r = np.eye(10)[y_r]
            Y_s = np.eye(10)[y_s]

            # Linear Layer
            W = multi_classifier(X_r, X_s, Y_r, Y_s, m, gamma)

            # Predictions
            acc = accuracy_multi(X_test @ W , Y_test)
            accs_oracle.append(acc)

            # Bad data
            if gan :
                train_data_weak = MNIST_GAN(n, m, 'cpu', train = True, supervision = False)
            else:
                train_data_weak = MNIST_generator(n, m, 'cpu', train = True, m_estim = int(p_estim * m), estimate_cov= True, supervision= False)
            #train_data_weak = MNIST_generator(n, m, 'cpu', train = True, m_estim = int(p_estim * m), estimate_cov= True, supervision= True, threshold= threshold)
            X_r = train_data_weak.X_real
            y_r = train_data_weak.y_real.astype(int)
            X_s = train_data_weak.X_s
            y_s = train_data_weak.y_s.astype(int)

            # One-Hot encoding
            Y_r = np.eye(10)[y_r]
            Y_s = np.eye(10)[y_s]

            # Linear Layer
            W = multi_classifier(X_r, X_s, Y_r, Y_s, m, gamma)

            # Predictions
            acc = accuracy_multi(X_test @ W , Y_test)
            accs_weak.append(acc)
        accs_oracle_all.append(accs_oracle)
        accs_weak_all.append(accs_weak)

    pis = np.array(ms) / (n + np.array(ms))

    accs_oracle_all = np.array(accs_oracle_all)
    accs_weak_all = np.array(accs_weak_all)

    # Plotting results
    # Oracle supervision
    label = 'GAN data' if gan else 'Gaussian data'
    color = 'tab:green'if gan else 'tab:red'
    ax[0].plot(pis, np.mean(accs_oracle_all, axis = 0), linewidth = linewidth, color = color, label = label)
    ax[0].scatter(pis, np.mean(accs_oracle_all, axis = 0), color = color, s= 100)
    ax[0].fill_between(pis,  np.mean(accs_oracle_all, axis = 0) - np.std(accs_oracle_all, axis = 0), np.mean(accs_oracle_all, axis = 0) + np.std(accs_oracle_all, axis = 0),
                    alpha = 0.2, linestyle = '-.', color = 'tab:orange')
    ax[0].set_xlabel('Proportion of Synthetic data', fontsize = fontsize)
    ax[0].set_ylabel('Test Accuracy', fontsize = fontsize)
    ax[0].tick_params(axis='x', which = 'both', labelsize=labelsize)
    ax[0].tick_params(axis='y', which = 'both', labelsize=labelsize)
    ax[0].set_title("Discriminator's Supervision", fontsize = fontsize)
    

    # No supervision
    ax[1].plot(pis, np.mean(accs_weak_all, axis = 0), linewidth = linewidth, color = color, label = label)
    ax[1].scatter(pis, np.mean(accs_weak_all, axis = 0), color = color, s= 100)
    ax[1].fill_between(pis,  np.mean(accs_weak_all, axis = 0) - np.std(accs_weak_all, axis = 0), np.mean(accs_weak_all, axis = 0) + np.std(accs_weak_all, axis = 0),
                    alpha = 0.2, linestyle = '-.', color = 'tab:orange')

    ax[1].set_xlabel('Proportion of Synthetic data', fontsize = fontsize)
    ax[1].tick_params(axis='x', which = 'both', labelsize=labelsize)
    ax[1].tick_params(axis='y', which = 'both', labelsize=labelsize)

    ax[1].set_title("No Supervision", fontsize = fontsize)
    

ax[0].legend(fontsize = labelsize)
ax[0].grid()
ax[1].legend(fontsize = labelsize)
ax[1].grid()

path = f'./study-plot/{name}_linear_n-{n}-gamma-{gamma}-threshold-{threshold}-p_estim-{p_estim}.pdf'
fig.savefig(path, bbox_inches='tight')
