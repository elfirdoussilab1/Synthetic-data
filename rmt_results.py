# In this file, we will state the results of our theoretical calculations gotten using RMT
import numpy as np
from scipy import integrate
import utils
from math import sqrt

# Root of a thrid order polynomial
def solution(a, b, c, d):
    s = -(-3*c/a + b**2/a**2)/(3*(sqrt(-4*(-3*c/a + b**2/a**2)**3 + (27*d/a - 9*b*c/a**2 + 2*b**3/a**3)**2)/2 + 27*d/(2*a) - 9*b*c/(2*a**2) + b**3/a**3)**(1/3)) - (sqrt(-4*(-3*c/a + b**2/a**2)**3 + (27*d/a - 9*b*c/a**2 + 2*b**3/a**3)**2)/2 + 27*d/(2*a) - 9*b*c/(2*a**2) + b**3/a**3)**(1/3)/3 - b/(3*a)
    return s

# Delta
def Delta(n, m, p, sigma, epsilon, rho, phi, gamma):
    alpha = phi * (1 - epsilon) + rho * epsilon
    N = n + m
    eta = p / N
    pi = n / N
    delta = 1

    while True:
        delta_ = eta / ( gamma + pi / (1 + delta) + alpha * (1-pi) * sigma**2 / (1 + alpha * sigma**2 * delta) )
        if abs(delta - delta_) < 1e-8:
            return delta_
        else:
            delta = delta_ 

# Resolvent
def Q_bar(n, m, vmu, vmu_orth, beta, sigma, epsilon, rho, phi, gamma):
    p = len(vmu)
    alpha = phi * (1 - epsilon) + rho * epsilon
    N = n + m
    pi = n / N
    delta = Delta(n, m, p, sigma, epsilon, rho, phi, gamma)
    theta = gamma + pi / (1 + delta) + alpha * (1 - pi) * sigma**2 / (1 + alpha * delta * sigma**2)
    vmu_beta = beta * vmu + np.sqrt(1 - beta**2) * vmu_orth

    # Q = (S_1 + S_2 + \theta I_p)^{-1}
    S_1 = pi * np.outer(vmu, vmu) / (1 + delta)
    S_2 = alpha * (1 - pi) * np.outer(vmu_beta, vmu_beta) / (1 + alpha * delta * sigma**2)

    return np.linalg.solve(S_1 + S_2 + theta * np.eye(p), np.eye(p))

# vmu^\top \bar \rmQ \vmu
def mu_Q_mu(n, m, p, mu, mu_orth, beta, sigma, epsilon, rho, phi, gamma):
    alpha = phi * (1 - epsilon) + rho * epsilon
    N = n + m
    pi = n / N
    delta = Delta(n, m, p, sigma, epsilon, rho, phi, gamma)
    theta = gamma + pi / (1 + delta) + alpha * (1 - pi) * sigma**2 / (1 + alpha * delta * sigma**2)
    delta_s = alpha * delta * sigma**2
    mu_beta_2 = (beta * mu)**2 + (1 - beta**2) * mu_orth**2

    # vmu^\top \rmr_1 \vmu
    mu_R1_mu = mu**2 * (1 - alpha * (1 - pi) * beta**2 * mu**2 / (theta * (1 + delta_s) + alpha* (1 - pi) * mu_beta_2)) / theta
    return mu_R1_mu / (1 + pi * mu_R1_mu / (1 + delta))

# vmu^\top \bar \rmQ \vmu_beta
def mu_Q_mu_beta(n, m, p, mu, mu_orth, beta, sigma, epsilon, rho, phi, gamma):
    alpha = phi * (1 - epsilon) + rho * epsilon
    N = n + m
    pi = n / N
    delta = Delta(n, m, p, sigma, epsilon, rho, phi, gamma)
    theta = gamma + pi / (1 + delta) + alpha * (1 - pi) * sigma**2 / (1 + alpha * delta * sigma**2)
    delta_s = alpha * delta * sigma**2
    mu_beta_2 = (beta * mu)**2 + (1 - beta**2) * mu_orth**2

    # vmu^\top R_2 \vmu_beta
    mu_R2_mu_beta = beta * (1 + delta) * mu**2 / (theta * (1 + delta) + pi * mu**2)
    # vmu_beta^\top R_2 \vmu_beta
    mu_beta_R2_mu_beta = (mu_beta_2 - pi * (beta * mu**2)**2 / (theta * (1 + delta) + pi * mu**2))/ theta

    return mu_R2_mu_beta / (1 + alpha * (1 - pi) * mu_beta_R2_mu_beta/ (1 + delta_s))


# E[g(x)]
def test_expectation(n, m, p, mu, mu_orth, beta, sigma, epsilon, rho, phi, gamma):
    alpha = phi * (1 - epsilon) + rho * epsilon
    lam = phi * (1 - epsilon) - rho * epsilon
    N = n + m
    pi = n / N
    delta = Delta(n, m, p, sigma, epsilon, rho, phi, gamma)
    delta_s = alpha * delta * sigma**2
    
    mu_q_mu = mu_Q_mu(n, m, p, mu, mu_orth, beta, sigma, epsilon, rho, phi, gamma)
    mu_q_mu_beta = mu_Q_mu_beta(n, m, p, mu, mu_orth, beta, sigma, epsilon, rho, phi, gamma)
    return pi * mu_q_mu / (1 + delta)  + lam * (1 - pi) * mu_q_mu_beta / (1 + delta_s)

