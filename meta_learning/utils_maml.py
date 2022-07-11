import numpy as np

from copy import deepcopy


def es_grad(f, theta, n, sigma):
    """
    Returns Monte Carlo ES Gradient
    """

    return sum([
        f(theta + sigma * g)[0] * g
        for g in [np.random.normal(size=len(theta.get_flattened_weights())) for _ in range(n)]
    ]) / (n * sigma)


def get_reward(this, task, random_vector):
    """
    Returns the reward on a given task with a random pertubation on the current policy
    Can be overriden to change gradient estimator, or the process all-together
    """

    new_theta = deepcopy(this.theta)
    
    d = es_grad(task, new_theta + this.sigma * random_vector, this.K, this.sigma)
    return task(new_theta + this.sigma * random_vector + this.alpha * d)[0]