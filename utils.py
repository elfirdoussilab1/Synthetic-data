# In this file, we will implement util functions
import numpy as np
import random

# Seed fixing
def fix_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

# Gaussian mixture generation
def gaussian_mixture(n, vmu, cov, real = True):
    p = len(vmu)
    y = np.ones(n)
    y[: n // 2] = -1
    if real:
        Z = np.random.randn(p, n)

    else: # Synthetic data
        Z = np.random.multivariate_normal(mean = np.zeros(p), cov = cov, size = n).T
    X = np.outer(vmu, y) +  Z # vmu.T @ y + Z
    return X, y

# Data generation: mixture of pi real data and (1 - pi) synthetic
def generate_data(n, m, p, vmu, vmu_beta, cov, epsilon, rho, phi):
    
    # Real data
    X_real, y_real = gaussian_mixture(n, vmu, np.eye(p), real = True)

    # Synthetic data
    X_s, y_s = gaussian_mixture(m, vmu_beta, cov, real = False)

    # Noise the labels of the synthetic data
    y_tilde = y_s * (2 * np.random.binomial(size = m, p = 1 - epsilon, n = 1) - 1) # p = P[X = 1], i.e 1 - p = epsilon
    
    # Pruning
    vq = np.zeros(m)
    # Indices of y_tilde = y
    m_1 = (y_tilde == y_s).sum()
    vq[y_tilde == y_s] = np.random.binomial(size = m_1, p = phi, n = 1) 
    vq[y_tilde != y_s] = np.random.binomial(size = m - m_1, p = rho, n = 1)

    # Test data
    X_test, y_test = gaussian_mixture(2*n, vmu, np.eye(p), real = True)

    return (X_real, y_real, X_s, y_tilde, vq), (X_test, y_test)

def accuracy(y, y_pred):
    acc = np.mean(y == y_pred)
    return max(acc, 1 - acc)
    #return acc

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

def empirical_accuracy(batch, n, m, p, vmu, vmu_beta, cov, epsilon, rho, phi, gamma, data_type = 'synthetic'):
    res = 0
    for i in range(batch):
        if 'synthetic' in data_type:
            (X_real, y_real, X_s, y_tilde, vq), (X_test, y_test) = generate_data(n, m, p, vmu, vmu_beta, cov, epsilon, rho, phi)
        
        elif 'amazon' in data_type:
            print("Coming soon !")
            return -1 # Not implemented yet

        w = classifier_vector(X_real, y_real, X_s, y_tilde, vq, gamma)
        res += accuracy(y_test, decision(w, X_test))
    
    return res / batch

def empirical_risk(batch, n, m, p, vmu, vmu_beta, cov, epsilon, rho, phi, gamma, data_type = 'synthetic'):
    res = 0
    for i in range(batch):
        if 'synthetic' in data_type:
            (X_real, y_real, X_s, y_tilde, vq), (X_test, y_test) = generate_data(n, m, p, vmu, vmu_beta, cov, epsilon, rho, phi)
        
        elif 'amazon' in data_type:
            print("Coming soon!")
            return -1 # Not implemented yet

        w = classifier_vector(X_real, y_real, X_s, y_tilde, vq, gamma)
        res += L2_loss(w, X_test, y_test)
    
    return res / batch

def empirical_mean(batch, n, m, p, vmu, vmu_beta, cov, epsilon, rho, phi, gamma, data_type = 'synthetic'):
    res = 0
    for i in range(batch):
        if 'synthetic' in data_type:
            (X_real, y_real, X_s, y_tilde, vq), (X_test, y_test) = generate_data(n, m, p, vmu, vmu_beta, cov, epsilon, rho, phi)
        
        elif 'amazon' in data_type:
            print("Coming soon!")
            return -1 # Not implemented yet

        w = classifier_vector(X_real, y_real, X_s, y_tilde, vq, gamma)
        res += np.mean(y_test * (X_test.T @ w))
    
    return res / batch

def empirical_mean_2(batch, n, m, p, vmu, vmu_beta, cov, epsilon, rho, phi, gamma, data_type = 'synthetic'):
    res = 0
    for i in range(batch):
        if 'synthetic' in data_type:
            (X_real, y_real, X_s, y_tilde, vq), (X_test, y_test) = generate_data(n, m, p, vmu, vmu_beta, cov, epsilon, rho, phi)
        
        elif 'amazon' in data_type:
            print("Coming soon!")
            return -1 # Not implemented yet

        w = classifier_vector(X_real, y_real, X_s, y_tilde, vq, gamma)
        res += np.mean((X_test.T @ w)**2)
    
    return res / batch

def gaussian(x, mean, std):
    return np.exp(- (x - mean)**2 / (2 * std**2)) / (std * np.sqrt(2 * np.pi))


######################## Toy Setting ########################
def empirical_risk_toy(batch, n, m, p, vmu, X_r, y_r, epsilon, rho, phi, gamma):
    res = 0
    #vmu_hat = np.sum(y_r * X_r, axis = 1) / n

    for i in range(batch):
        Z = np.random.randn(p, n)
        cov = Z @ Z.T / n
        # Synthetic dataset
        (X_real, y_real, X_s, y_tilde, vq), (X_test, y_test) = generate_data(n, m, p, vmu, vmu, cov, epsilon, rho, phi)
        
        # Classifier
        w = classifier_vector(X_r, y_r, X_s, y_tilde, vq, gamma)
        res += L2_loss(w, X_test, y_test)
    return res / batch

def empirical_accuracy_toy(batch, n, m, p, vmu, X_r, y_r, epsilon, rho, phi, gamma):
    res = 0
    #vmu_hat = np.sum(y_r * X_r, axis = 1) / n

    for i in range(batch):
        #Z = np.random.randn(p, n)
        #cov = Z @ Z.T / n
        C = (vmu * np.ones((n, p)) ).T
        cov = (y_r * X_r - C) @ (y_r * X_r - C).T / n
        # Synthetic dataset
        (X_real, y_real, X_s, y_tilde, vq), (X_test, y_test) = generate_data(n, m, p, vmu, vmu, cov, epsilon, rho, phi)
        
        # Classifier
        w = classifier_vector(X_r, y_r, X_s, y_tilde, vq, gamma)
        res += accuracy(y_test, decision(w, X_test))
    return res / batch

    