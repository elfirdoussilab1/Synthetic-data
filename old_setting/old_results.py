# vmu^\top \bar \rmQ \vmu: verified!
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

# vmu^\top \bar \rmQ \vmu_beta: verified!
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


# E[g(x)]: verified!
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

# vmu_beta^\top \rmQ vmu_\beta: verified!
def mu_beta_Q_mu_beta(n, m, p, mu, mu_orth, beta, sigma, epsilon, rho, phi, gamma):
    alpha = phi * (1 - epsilon) + rho * epsilon
    N = n + m
    pi = n / N
    delta = Delta(n, m, p, sigma, epsilon, rho, phi, gamma)
    theta = gamma + pi / (1 + delta) + alpha * (1 - pi) * sigma**2 / (1 + alpha * delta * sigma**2)
    delta_s = alpha * delta * sigma**2
    mu_beta_2 = (beta * mu)**2 + (1 - beta**2) * mu_orth**2

    # mu_beta_R2_mu_beta
    mu_beta_R2_mu_beta = (mu_beta_2 - pi * (beta * mu**2)**2 / (theta * (1 + delta) + pi * mu**2))/ theta

    return mu_beta_R2_mu_beta / (1 + alpha * (1 - pi) * mu_beta_R2_mu_beta / (1 + delta_s))

# vmu^\top \rmQ^2 \vmu: verified!
def mu_Q_2_mu(n, m, p, mu, mu_orth, beta, sigma, epsilon, rho, phi, gamma):
    alpha = phi * (1 - epsilon) + rho * epsilon
    N = n + m
    pi = n / N
    delta = Delta(n, m, p, sigma, epsilon, rho, phi, gamma)
    theta = gamma + pi / (1 + delta) + alpha * (1 - pi) * sigma**2 / (1 + alpha * delta * sigma**2)
    delta_s = alpha * delta * sigma**2
    mu_beta_2 = (beta * mu)**2 + (1 - beta**2) * mu_orth**2

    # vmu^\top \rmr_1 \vmu
    mu_R1_mu = mu**2 * (1 - alpha * (1 - pi) * beta**2 * mu**2 / (theta * (1 + delta_s) + alpha* (1 - pi) * mu_beta_2)) / theta

    # vmu^\top R_1^2 \vmu
    zeta = theta * (1 + delta_s) + alpha * (1 - pi) * mu_beta_2
    mu_R1_2_mu = mu**2 / theta**2 + alpha * (1 - pi) * (beta * mu**2)**2 * (alpha * (1 - pi) * mu_beta_2 / zeta - 2) / (zeta * theta**2)
    return mu_R1_2_mu / (1 + pi * mu_R1_mu / (1 + delta))**2

# \vmu^\top \rmQ^2 \vmu_\beta: verified!
def mu_Q_2_mu_beta(n, m, p, mu, mu_orth, beta, sigma, epsilon, rho, phi, gamma):
    # quantities
    alpha = phi * (1 - epsilon) + rho * epsilon
    N = n + m
    pi = n / N
    delta = Delta(n, m, p, sigma, epsilon, rho, phi, gamma)
    theta = gamma + pi / (1 + delta) + alpha * (1 - pi) * sigma**2 / (1 + alpha * delta * sigma**2)
    delta_s = alpha * delta * sigma**2
    mu_beta_2 = (beta * mu)**2 + (1 - beta**2) * mu_orth**2

    # mu_R1_mu
    mu_R1_mu = mu**2 * (1 - alpha * (1 - pi) * beta**2 * mu**2 / (theta * (1 + delta_s) + alpha* (1 - pi) * mu_beta_2)) / theta

    # mu_beta_R2_mu_beta
    mu_beta_R2_mu_beta = (mu_beta_2 - pi * (beta * mu**2)**2 / (theta * (1 + delta) + pi * mu**2))/ theta

    # mu_beta_R2_R1_mu
    zeta = theta * (1 + delta) + pi * mu**2
    zeta_beta = theta * (1 + delta_s) + alpha * (1 - pi) * mu_beta_2

    mu_beta_R2_R1_mu = beta * mu**2 * (1 - alpha * (1 - pi) * mu_beta_2 / zeta_beta - pi * mu**2 / zeta + alpha * pi * (1 - pi) * (beta * mu**2)**2 / (zeta * zeta_beta)) / theta**2

    return mu_beta_R2_R1_mu / ( (1 + pi * mu_R1_mu / (1 + delta)) * (1 + alpha * (1 - pi) * mu_beta_R2_mu_beta / (1 + delta_s)) )

# \vmu_\beta^\top \rmQ^2 \vmu_\beta: verified!
def mu_beta_Q_2_mu_beta(n, m, p, mu, mu_orth, beta, sigma, epsilon, rho, phi, gamma):
    # quantities
    alpha = phi * (1 - epsilon) + rho * epsilon
    N = n + m
    pi = n / N
    delta = Delta(n, m, p, sigma, epsilon, rho, phi, gamma)
    theta = gamma + pi / (1 + delta) + alpha * (1 - pi) * sigma**2 / (1 + alpha * delta * sigma**2)
    delta_s = alpha * delta * sigma**2
    mu_beta_2 = (beta * mu)**2 + (1 - beta**2) * mu_orth**2

    # mu_beta_R2_mu_beta
    mu_beta_R2_mu_beta = (mu_beta_2 - pi * (beta * mu**2)**2 / (theta * (1 + delta) + pi * mu**2))/ theta

    # mu_beta_R2_2_mu_beta
    zeta = theta * (1 + delta) + pi * mu**2
    mu_beta_R2_2_mu_beta = mu_beta_2 / theta**2 + pi * (beta * mu**2)**2 * (pi * mu**2 / zeta - 2) / (zeta * theta**2)

    return mu_beta_R2_2_mu_beta / (1 + alpha * (1 - pi) * mu_beta_R2_mu_beta / (1 + delta_s))**2

# E[g(x)^2]
def test_expectation_2(n, m, p, mu, mu_orth, beta, sigma, epsilon, rho, phi, gamma):
    # quantities
    alpha = phi * (1 - epsilon) + rho * epsilon
    lam = phi * (1 - epsilon) - rho * epsilon
    N = n + m
    pi = n / N
    eta = p / N
    delta = Delta(n, m, p, sigma, epsilon, rho, phi, gamma)
    theta = gamma + pi / (1 + delta) + alpha * (1 - pi) * sigma**2 / (1 + alpha * delta * sigma**2)
    delta_s = alpha * delta * sigma**2
    mu_beta_2 = (beta * mu)**2 + (1 - beta**2) * mu_orth**2

    # Matrix quantities
    mu_q_mu = mu_Q_mu(n, m, p, mu, mu_orth, beta, sigma, epsilon, rho, phi, gamma)
    mu_q_mu_beta = mu_Q_mu_beta(n, m, p, mu, mu_orth, beta, sigma, epsilon, rho, phi, gamma)
    mu_beta_q_mu_beta = mu_beta_Q_mu_beta(n, m, p, mu, mu_orth, beta, sigma, epsilon, rho, phi, gamma)

    mu_q_2_mu = mu_Q_2_mu(n, m, p, mu, mu_orth, beta, sigma, epsilon, rho, phi, gamma)
    mu_beta_q_2_mu_beta = mu_beta_Q_2_mu_beta(n, m, p, mu, mu_orth, beta, sigma, epsilon, rho, phi, gamma)
    mu_q_2_mu_beta = mu_Q_2_mu_beta(n, m, p, mu, mu_orth, beta, sigma, epsilon, rho, phi, gamma)

    # as and bs
    a_1 = pi * eta / (theta * (1 + delta))**2
    a_2 = pi * eta * sigma**2 / (theta * (1 + delta))**2
    b_1 = alpha * (1 - pi) * eta * sigma**2 / (theta * (1 + delta_s))**2
    b_2 = alpha * (1 - pi) * eta * sigma**4 / (theta * (1 + delta_s))**2

    h = (1 - b_2) * (1 - a_1) - a_2 * b_1
    T = eta * (1 - b_2 + b_1 * sigma**2) / (h * theta**2)
    # vmu^\top E \vmu
    mu_E_mu = ((1 - b_2) * (mu_q_mu**2 + mu_q_2_mu) + b_1 * (mu_q_mu_beta**2 + mu_q_2_mu * sigma**2))/h
    mu_E_mu_beta = ((1 - b_2) * (mu_q_mu * mu_q_mu_beta + mu_q_2_mu_beta) + b_1 * (mu_q_mu_beta * mu_beta_q_mu_beta + mu_q_2_mu_beta * sigma**2)) / h
    mu_beta_E_mu_beta = ((1 - b_2) * (mu_q_mu_beta**2 + mu_beta_q_2_mu_beta) + b_1 * (mu_beta_q_mu_beta**2 + mu_beta_q_2_mu_beta * sigma**2)) / h

    res = pi**2 * mu_E_mu / (1 + delta)**2 + (lam * (1 - pi))**2 * mu_beta_E_mu_beta / (1 + delta_s)**2 + 2 * lam * pi * (1 - pi) * mu_E_mu_beta / ((1 + delta) * (1 + delta_s))

    res += pi * T * ( 1 - 2 * pi * mu_q_mu / (1 + delta)  - 2 * lam * (1 - pi) * mu_q_mu_beta / (1 + delta_s) ) / (1 + delta)**2
    res += (1 - pi) * T * sigma**2 * (alpha - 2 * (1 - pi) * (lam**2)  * mu_beta_q_mu_beta / (1 + delta_s) - 2 * lam * pi * mu_q_mu_beta / (1 + delta) ) / (1 + delta_s)**2

    return res