# In this file, we will state the results of our theoretical calculations gotten using RMT
import numpy as np
from scipy import integrate
import utils
from math import sqrt
import cmath

# Root of a thrid order polynomial
def solution(a, b, c, d):
    # Depressing the cubic
    p = (3 * a * c - b ** 2) / (3 * a ** 2)
    q = (2 * b ** 3 - 9 * a * b * c + 27 * a ** 2 * d) / (27 * a ** 3)

    # Solving for t using Cardano's method
    delta = (q / 2) ** 2 + (p / 3) ** 3

    u = (-q / 2 + cmath.sqrt(delta)) ** (1 / 3)
    v = (-q / 2 - cmath.sqrt(delta)) ** (1 / 3)

    t1 = u + v
    t2 = -(u + v) / 2 + cmath.sqrt(3) * (u - v) / 2j
    t3 = -(u + v) / 2 - cmath.sqrt(3) * (u - v) / 2j

    # Back-substitute to find x
    x1 = t1 - b / (3 * a)
    x2 = t2 - b / (3 * a)
    x3 = t3 - b / (3 * a)

    return x1, x2, x3

# Delta
def Delta(n, m, eigvals, epsilon, rho, phi, gamma): # verified
    alpha = phi * (1 - epsilon) + rho * epsilon
    N = n + m
    pi = n / N
    delta = 0
    delta_s = 0

    while True:
        arr = gamma + pi / (1 + delta) + alpha * (1 - pi) * eigvals / (1 + delta_s) 
        delta_up =  np.sum(1 / arr) / N
        delta_s_up = alpha * np.sum(eigvals / arr) / N
        if abs(delta_up - delta) < 1e-12 and abs(delta_s_up - delta_s) < 1e-12:
            return delta_up, delta_s_up
        else:
            delta = delta_up
            delta_s = delta_s_up


def ddelta(n, m, p, sigma, epsilon, rho, phi, gamma): # Old one
    alpha = phi * (1 - epsilon) + rho * epsilon
    N = n + m
    eta = p / N
    pi = n / N
    delta = 0
    for i in range(50):
        delta = eta / (gamma + pi / (1 + delta) + alpha * (1 - pi) * sigma**2 / (1 + alpha * sigma**2 * delta))
        print(delta)
    return delta

def Delta_bars(n, m, p, epsilon, rho, phi, gamma):
    alpha = phi * (1 - epsilon) + rho * epsilon
    N = n + m
    eta = p / N
    pi = n / N
    delta = 1
    delta_s = 1
    delta_bar = 1

    while True:
        delta_up = pi * (1 + delta_s) * delta_bar / (alpha * (1 - pi))
        delta_s_up = alpha * delta / (1 + delta_bar)
        a = gamma + pi / (1 + delta) + alpha * (1 - pi) / ((1 + delta_s) * (1 + delta_bar))
        delta_bar_up = alpha * (1 - pi) * p / (n * (1 + delta_s) * a)

        if abs(delta_up - delta) < 1e-12 and abs(delta_s_up - delta_s) < 1e-12 and abs(delta_bar_up - delta_bar) < 1e-12:
            return delta_up, delta_s_up, delta_bar_up
        else:
            delta = delta_up
            delta_s = delta_s_up
            delta_bar = delta_bar_up

        

# Resolvent
def Q_bar(n, m, vmu, vmu_beta, cov, eigvals, epsilon, rho, phi, gamma):
    p = len(vmu)
    alpha = phi * (1 - epsilon) + rho * epsilon
    N = n + m
    pi = n / N
    delta, delta_s = Delta(n, m, eigvals, epsilon, rho, phi, gamma)

    # Q = (S_1 + S_2 + \theta I_p)^{-1}
    S_1 = pi * (np.outer(vmu, vmu) + np.eye(p)) / (1 + delta) 
    S_2 = alpha * (1 - pi) * (np.outer(vmu_beta, vmu_beta) + cov) / (1 + delta_s)

    return np.linalg.solve(S_1 + S_2 + gamma * np.eye(p), np.eye(p))

def Q_bar_bar(n, m, p, vmu, epsilon, rho, phi, gamma):
    alpha = phi * (1 - epsilon) + rho * epsilon
    N = n + m
    pi = n / N
    mu = np.linalg.norm(vmu)

    # quantities
    delta, delta_s, delta_bar = Delta_bars(n, m, p, epsilon, rho, phi, gamma)
    a = pi / (1 + delta) + alpha * (1 - pi) / (1 + delta_s)
    b = gamma + pi / (1 + delta) + alpha * (1 - pi) / ((1 + delta_s) * (1 + delta_bar))

    return (np.eye(p) - a * np.outer(vmu, vmu) / (b + a * mu**2)) / b


# Implementation of Q_bar with no matrix inversion
def Q_bar_smart(n, m, vmu, vmu_beta, eigvals, eigvectors, epsilon, rho, phi, gamma):# verified, but prefer the first one
    P = eigvectors.T # (v_1, ..., v_p)
    alpha = phi * (1 - epsilon) + rho * epsilon
    N = n + m
    pi = n / N
    delta, delta_s = Delta(n, m, eigvals, epsilon, rho, phi, gamma)

    # Q = A^{-1} - A^{-1} (zeta_1 vmu vmu^\top + zeta_2 vmu_beta vmu_beta^top) A^{-1}
    # inverse of Delta matrix
    DTA_1 = 1 / (gamma + pi / (1 + delta) + alpha * (1 - pi)* eigvals / (1 + delta_s))

    A_1 = (P * DTA_1) @ P.T
    #A_1 = P @ np.diag(DTA_1) @ P.T
    mu_A_mu = vmu @ A_1 @ vmu.T
    mu_A_mu_beta = vmu @ A_1 @ vmu_beta.T
    mu_beta_A_mu_beta = vmu_beta @ A_1 @ vmu_beta.T
    det = (1 + pi * mu_A_mu / (1 + delta)) * (1 + alpha * (1 - pi) * mu_beta_A_mu_beta / (1 + delta_s)) - alpha * pi * (1 - pi) * mu_A_mu_beta**2 / ((1 + delta)*(1 + delta_s))

    M_11 = (1 + alpha * (1 - pi) * mu_beta_A_mu_beta / (1 + delta_s)) / det
    M_12 = - sqrt(alpha * pi * (1 - pi) / ((1 + delta)*(1 + delta_s))) * mu_A_mu_beta / det
    M_22 = (1 + pi * mu_A_mu / (1 + delta)) / det
    zeta_1 = pi * M_11 / (1 + delta)
    zeta_2 = alpha * (1 - pi) * M_22 / (1 + delta_s)
    zeta_3 = sqrt(alpha * pi * (1 - pi) / ((1 + delta)*(1 + delta_s))) * M_12

    # Finally
    q_bar = A_1 - A_1 @ (zeta_1 * np.outer(vmu, vmu) + zeta_2 * np.outer(vmu_beta, vmu_beta) + zeta_3 * (np.outer(vmu, vmu_beta) + np.outer(vmu_beta, vmu))) @ A_1
    return q_bar

# Test Expectation: E[w^T x]
def test_expectation(n, m, p, vmu, vmu_beta, cov, eigvals, eigvectors, epsilon, rho, phi, gamma):
    N = n + m
    pi = n / N
    lam = phi * (1 - epsilon) - rho * epsilon
    #q_bar = Q_bar_smart(n, m, vmu, vmu_beta, eigvals, eigvectors, epsilon, rho, phi, gamma)
    q_bar = Q_bar(n, m, vmu, vmu_beta, cov, eigvals, epsilon, rho, phi, gamma)
    delta, delta_s = Delta(n, m, eigvals, epsilon, rho, phi, gamma)

    s_1 = vmu @ q_bar @ vmu.T
    s_2 = vmu_beta @ q_bar @ vmu.T
    return pi * s_1 / (1 + delta) + lam * (1 - pi) * s_2 / (1 + delta_s)

# Second order moment: E[(w^T x)^2]
def test_expectation_2(n, m, p, vmu, vmu_beta, cov, eigvals, eigvectors, epsilon, rho, phi, gamma):
    N = n + m
    pi = n / N
    lam = phi * (1 - epsilon) - rho * epsilon
    alpha = phi * (1 - epsilon) + rho * epsilon
    q_bar = Q_bar(n, m, vmu, vmu_beta, cov, eigvals, epsilon, rho, phi, gamma)
    delta, delta_s = Delta(n, m, eigvals, epsilon, rho, phi, gamma)

    # Trace identities
    arr = gamma + pi / (1 + delta) + alpha * (1 - pi) * eigvals / (1 + delta_s) 
    t_1 = np.sum(1 / arr**2) / N # first trace in lemma A.5
    t_2 = np.sum((eigvals / arr)**2) / N
    t_3 = np.sum(eigvals / arr**2) / N
    
    # Constants as and bs
    a_1 = pi * t_1 / (1 + delta)**2
    a_2 = pi * t_3 / (1 + delta)**2
    b_1 = alpha * (1 - pi) * t_3 / (1 + delta_s)**2
    b_2 = alpha * (1 - pi) * t_2 / (1 + delta_s)**2
    h = (1 - b_2) * (1 - a_1) - a_2 * b_1

    # T_1 and T_2 in the final term
    T_1 = (a_1 * (1 - b_2) + a_2 * b_1) / h
    T_2 = b_1 / (alpha * h)

    mu_q_mu = vmu @ q_bar @ vmu.T
    mu_q_mu_beta = vmu @ q_bar @ vmu_beta.T
    mu_beta_q_mu_beta =  vmu_beta @ q_bar @ vmu_beta.T
    

    exp = ((1 - b_2) * q_bar @ (np.outer(vmu, vmu) + np.eye(p)) @ q_bar + b_1 * q_bar @ (np.outer(vmu_beta, vmu_beta) + cov) @ q_bar) / h
    # Computing the sums now
    s_1 = (pi / (1 + delta))**2 * vmu @ exp @ vmu.T + (lam * (1 - pi)/ (1 + delta_s))**2 * vmu_beta @ exp @ vmu_beta.T + 2 * lam * pi * (1 - pi) * vmu @ exp @ vmu_beta.T / ((1 + delta)*(1 + delta_s))

    s_2 = T_1 * (1 - 2 * pi * mu_q_mu / (1 + delta) - 2 * lam * (1 - pi) * mu_q_mu_beta / (1 + delta_s))

    s_3 = T_2 * (alpha - 2 * (1 - pi) * lam**2 * mu_beta_q_mu_beta / (1 + delta_s) - 2 * lam * pi * mu_q_mu_beta / (1 + delta) )

    return s_1 + s_2 + s_3

# Test accuracy
def test_accuracy(n, m, p, vmu, vmu_beta, cov, eigvals, eigvectors, epsilon, rho, phi, gamma):

    # E[g] and E[g^2]
    mean = test_expectation(n, m, p, vmu, vmu_beta, cov, eigvals, eigvectors, epsilon, rho, phi, gamma)
    expec_2 = test_expectation_2(n, m, p, vmu, vmu_beta, cov, eigvals, eigvectors, epsilon, rho, phi, gamma)
    std = np.sqrt(expec_2 - mean**2)
    return 1 - integrate.quad(lambda x: utils.gaussian(x, 0, 1), abs(mean)/std, np.inf)[0]

# test Risk
def test_risk(n, m, p, vmu, vmu_beta, cov, eigvals, eigvectors, epsilon, rho, phi, gamma):
    # E[g] and E(g^2)
    mean = test_expectation(n, m, p, vmu, vmu_beta, cov, eigvals, eigvectors, epsilon, rho, phi, gamma)
    expec_2 = test_expectation_2(n, m, p, vmu, vmu_beta, cov, eigvals, eigvectors, epsilon, rho, phi, gamma)
    return expec_2 + 1 - 2 * mean    


########################### Toy Setting results #########################
def test_expectation_toy(n, m, p, mu, epsilon, rho, phi, gamma):
    # Constants
    N = n + m
    pi = n / N
    lam = phi * (1 - epsilon) - rho * epsilon
    alpha = phi * (1 - epsilon) + rho * epsilon
    delta, delta_s, delta_bar = Delta_bars(n, m, p, epsilon, rho, phi, gamma)

    # Quantities
    a = pi / (1 + delta) + alpha * (1 - pi) / (1 + delta_s)
    b = gamma + pi / (1 + delta) + alpha * (1 - pi) / ((1 + delta_s) * (1 + delta_bar))

    return (pi / (1 + delta) + lam * (1 - pi) / (1 + delta_s)) * mu**2 / (b + a * mu**2)

def test_expectation_2_toy(n, m, p, mu, epsilon, rho, phi, gamma):
    # Constants
    N = n + m
    pi = n / N
    eta = p / N
    lam = phi * (1 - epsilon) - rho * epsilon
    alpha = phi * (1 - epsilon) + rho * epsilon
    delta, delta_s, delta_bar = Delta_bars(n, m, p, epsilon, rho, phi, gamma)

    # Quantities
    a = pi / (1 + delta) + alpha * (1 - pi) / (1 + delta_s)
    b = gamma + pi / (1 + delta) + alpha * (1 - pi) / ((1 + delta_s) * (1 + delta_bar))
    h_bar = 1 - (alpha * (1 - pi) / ((1 + delta_s) * (1 + delta_bar)))**2 * (p / (n * b**2))
    
    # 1/N Tr(Q_bar**2)
    t = eta / b**2
    a_1 = pi * t / (h_bar * (1 + delta)**2) 
    a_2 = pi * t / (h_bar * (1 + delta)**2 * (1 + delta_bar)**2 )
    b_1 = alpha * (1 - pi) * t / (h_bar * (1 + delta_s)**2 * (1 + delta_bar)**2)
    b_2 = alpha * (1 - pi) * t / (h_bar * (1 + delta_s)**2 * (1 + delta_bar)**2)
    h = (1 - a_1) * (1 - b_2) - a_2 * b_1
    T_1 = (1 + delta)**2 * (a_1 * (1 - b_2) + a_2 * b_1) / (pi * h)
    T_2 = (1 + delta_s)**2 * b_1 / (alpha * (1 - pi) * h)

    mu_q_mu = mu**2 / (b + a * mu**2)
    mu_q_2_mu = mu**2 / (b + a * mu**2)**2

    # Final terms
    r_1 = (pi / (1 + delta) + lam * (1 - pi) / (1 + delta_s)) **2 / h
    s_1 = (1 + b_1 - b_2) * mu_q_mu**2 + (1 - b_2 + b_1 / (1 + delta_bar)**2) * mu_q_2_mu / h_bar

    r_2 = pi * T_1 / (1 + delta)**2
    s_2 = 1 - 2* pi * mu_q_mu / (1 + delta) - 2 * lam * (1 - pi) * mu_q_mu / (1 + delta_s)

    r_3 = (1 - pi) * T_2 / (1 + delta_s)**2
    s_3 = alpha - 2 * lam**2 * (1 - pi) * mu_q_mu / (1 + delta_s) - 2 * lam * pi * mu_q_mu / (1 + delta)

    return r_1 * s_1 + r_2 * s_2 + r_3 * s_3

def test_accuracy_toy(n, m, p, mu, epsilon, rho, phi, gamma):
    # E[g] and E[g^2]
    mean = test_expectation_toy(n, m, p, mu, epsilon, rho, phi, gamma)
    expec_2 = test_expectation_2_toy(n, m, p, mu, epsilon, rho, phi, gamma)
    std = np.sqrt(expec_2 - mean**2)
    return 1 - integrate.quad(lambda x: utils.gaussian(x, 0, 1), abs(mean)/std, np.inf)[0]

def test_risk_toy(n, m, p, mu, epsilon, rho, phi, gamma):
    # E[g] and E[g^2]
    mean = test_expectation_toy(n, m, p, mu, epsilon, rho, phi, gamma)
    expec_2 = test_expectation_2_toy(n, m, p, mu, epsilon, rho, phi, gamma)
    return expec_2 + 1 - 2 * mean   