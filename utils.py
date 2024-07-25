# In this file, we will implement util functions
import numpy as np
import random

# Seed fixing
def fix_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

# Gaussian mixture generation
def gaussian_mixture(n, vmu, sigma = 1.):
    p = len(vmu)
    y = np.ones(n)
    y[: n // 2] = -1
    Z = np.random.randn(p, n)
    X = np.outer(vmu, y) + sigma * Z # vmu.T @ y + Z
    return X, y

# Data generation: mixture of pi real data and (1 - pi) synthetic
def generate_data(n, m, p, mu, mu_orth, beta, sigma, epsilon, rho, phi):
    
    # Real data
    vmu = np.zeros(p)
    vmu[0] = mu
    X_real, y_real = gaussian_mixture(n, vmu, sigma = 1)

    # Synthetic data
    vmu_orth = np.zeros(p)
    vmu_orth[1] = mu_orth
    vmu_beta = beta * vmu + np.sqrt(1 - beta**2) * vmu_orth
    X_s, y_s = gaussian_mixture(m, vmu_beta, sigma)

    # Noise the labels of the synthetic data
    y_tilde = y_s * (2 * np.random.binomial(size = m, p = 1 - epsilon, n = 1) - 1) # p = P[X = 1], i.e 1 - p = epsilon
    
    # Pruning
    vq = np.zeros(m)
    # Indices of y_tilde = y
    n_1 = (y_tilde == y_s).sum()
    vq[y_tilde == y_s] = np.random.binomial(size = n_1, p = phi, n = 1) 
    vq[y_tilde != y_s] = np.random.binomial(size = n - n_1, p = rho, n = 1)

    # Test data
    X_test, y_test = gaussian_mixture(2*n, vmu, sigma = 1)

    return (X_real, y_real, X_s, y_tilde, vq), (X_test, y_test)

def accuracy(y, y_pred):
    acc = np.mean(y == y_pred)
    return max(acc, 1 - acc)

def L2_loss(w, X, y):
    # X of shape (p, n)
    return np.mean((X.T @ w - y)**2)

# Decision functions
g = lambda w, X : X.T @ w

decision = lambda w, X: 2* (g(w, X) >= 0) -1

def classifier_vector(X_real, y_real, X_s, y_tilde, vq, gamma):
    # X of shape (p, n)
    p, n = X_real.shape
    m = len(y_tilde)
    N = n + m
    Q = np.linalg.solve( (X_real @ X_real.T + (X_s * vq) @ X_s.T) / N + gamma * np.eye(p), np.eye(p))

    return Q @ (X_real @ y_real + (X_s * vq) @ y_tilde) / N

def empirical_accuracy(batch, n, m, p, mu, mu_orth, beta, sigma, epsilon, rho, phi, gamma, data_type = 'synthetic'):
    res = 0
    for i in range(batch):
        if 'synthetic' in data_type:
            (X_real, y_real, X_s, y_tilde, vq), (X_test, y_test) = generate_data(n, m, p, mu, mu_orth, beta, sigma, epsilon, rho, phi)
        
        elif 'amazon' in data_type:
            print("Not implemented yet!")
            return -1 # Not implemented yet

        w = classifier_vector(X_real, y_real, X_s, y_tilde, vq, gamma)
        res += accuracy(y_test, decision(w, X_test))
    
    return res / batch

def empirical_risk(batch, n, m, p, mu, mu_orth, beta, sigma, epsilon, rho, phi, gamma, data_type = 'synthetic'):
    res = 0
    for i in range(batch):
        if 'synthetic' in data_type:
            (X_real, y_real, X_s, y_tilde, vq), (X_test, y_test) = generate_data(n, m, p, mu, mu_orth, beta, sigma, epsilon, rho, phi)
        
        elif 'amazon' in data_type:
            print("Not implemented yet!")
            return -1 # Not implemented yet

        w = classifier_vector(X_real, y_real, X_s, y_tilde, vq, gamma)
        res += L2_loss(w, X_test, y_test)
    
    return res / batch

def empirical_mean(batch, n, m, p, mu, mu_orth, beta, sigma, epsilon, rho, phi, gamma, data_type = 'synthetic'):
    res = 0
    for i in range(batch):
        if 'synthetic' in data_type:
            (X_real, y_real, X_s, y_tilde, vq), (X_test, y_test) = generate_data(n, m, p, mu, mu_orth, beta, sigma, epsilon, rho, phi)
        
        elif 'amazon' in data_type:
            print("Not implemented yet!")
            return -1 # Not implemented yet

        w = classifier_vector(X_real, y_real, X_s, y_tilde, vq, gamma)
        res += np.mean(y_test * (X_test.T @ w))
    
    return res / batch

def empirical_mean_2(batch, n, m, p, mu, mu_orth, beta, sigma, epsilon, rho, phi, gamma, data_type = 'synthetic'):
    res = 0
    for i in range(batch):
        if 'synthetic' in data_type:
            (X_real, y_real, X_s, y_tilde, vq), (X_test, y_test) = generate_data(n, m, p, mu, mu_orth, beta, sigma, epsilon, rho, phi)
        
        elif 'amazon' in data_type:
            print("Not implemented yet!")
            return -1 # Not implemented yet

        w = classifier_vector(X_real, y_real, X_s, y_tilde, vq, gamma)
        res += np.mean((X_test.T @ w)**2)
    
    return res / batch

def gaussian(x, mean, std):
    return np.exp(- (x - mean)**2 / (2 * std**2)) / (std * np.sqrt(2 * np.pi))