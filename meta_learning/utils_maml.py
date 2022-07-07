import numpy as np


def es_grad(f, theta, n, sigma):
    """
    Returns Monte Carlo ES Gradient
    """

    return sum([
        f(theta + sigma * g)[0] * g
        for g in [np.random.normal(size=theta.shape)]
    ]) / (n * sigma)