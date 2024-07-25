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
def Q_bar():
    pass