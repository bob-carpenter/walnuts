import numpy as np
import scipy.stats as sps

def standard_normal_lpdf(q):
    return -0.5 * np.dot(q, q)
def standard_normal_grad(q):
    return -q

def correlated_normal_lpdf(q):
    rho = 0.5
    return -0.5 * q[0]**2 - 0.5 / (1 - rho**2) * (q[1] - rho * q[0])**2
def correlated_normal_grad(q):
    rho = 0.5
    return np.array([-q[0] + rho * q[1],
                         (-q[1] + rho * q[0]) / (1 - rho**2)])

def rosenbrock_lpdf(q):
    return -0.5 * q[0]**2 - 0.5 * (q[1] - q[0]**2)**2
def rosenbrock_grad(q):
    return np.array([-q[0] + 2 * q[0] * q[1] - 2 * q[0]**3,
                         q[0]**2 - q[1]])

def funnel_lpdf(q):
    return -0.5 * q[0]**2 / 9 - 0.5 * np.dot(q[1:], q[1:]) / np.exp(0.5 * q[0])
def funnel_grad(q):
    grad = np.empty(q.size)
    grad[0] = -(q[0] / 9 - 0.25 * np.dot(q[1:], q[1:]) / np.exp(0.5 * q[0]))
    grad[1:] = -1 / np.exp(0.5 * q[0]) * q[1:]
    return grad

