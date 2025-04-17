from scipy.special import softmax
from tqdm import trange
import numpy as np
import warnings

def disable_runtime_warnings():
    """Filters out `RuntimeWarning` messages.
    
    Return:
        None
    """
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    
def uturn(theta_rho1, theta_rho2, inv_mass):
    """Return `True` if there is a U-turn between the positions and momentums.

    There is a U-turn between `(theta1, rho1) = theta_rho1` and
    `(theta2, rho2) = theta_rho2` if `dot(theta2 - theta1, rho_bk)
    < 0` or `dot(theta2 - theta1, rho_fw) < 0`.

    Args:
        theta_rho1: The starting state.
        theta_rho2: The ending state.
        inv_mass: The diagonal of the inverse mass matrix, used as a Euclidean metric in the U-turn condition.
    Returns:
        `True` if there is a U-turn between the states.
    """
    theta1, rho1 = theta_rho1
    theta2, rho2 = theta_rho2
    diff = inv_mass * (theta2 - theta1)
    return np.dot(rho1, diff) < 0 or np.dot(rho2, diff) < 0


def sub_uturn(orbit, start, end, inv_mass):
    """Return `True` if there is a sub-U-turn in `orbit[start:end]`.

    An orbit has a sub-U-turn if (a) the orbit has a U-turn, (b) the
    first half of the orbit has a sub-U-turn, or (c) the second half of
    the orbit has a sub-U-turn. The inverse mass matrix is used as a Euclidean
    metric in the calculation (see the `uturn()` function for a definition).

    Args:
        orbit: A list of pairs representing position and momentum in phase space of size `2^K` for some `K >= 0`.
        start: The first position to consider in the orbit.
        end: The last position to consider in the orbit.
        inv_mass: The diagonal of the inverse mass matrix.
    Returns:
        `True` if there is a sub-U-turn between the ends of the orbit.
    """
    size = end - start
    if size >= 2 and uturn(orbit[start], orbit[end - 1], inv_mass):
        return True
    if size >= 4:
        mid = start + size // 2
        return sub_uturn(orbit, start, mid, inv_mass) or sub_uturn(
            orbit, mid, end, inv_mass
        )
    return False


def leapfrog(grad, theta, rho, step_size, inv_mass, num_steps):
    """Return the result of multiple leapfrog steps from `(theta, rho)`.

    Args:
        grad: The gradient function for the target log density.
        theta: The initial position.
        rho: The initial momentum.
        step_size: The interval of time discretization of the dynamics simulator.
        inv_mass: The diagonal of the inverse mass matrix.
        num_steps: The number of leapfrog steps to take.
    Returns:
        A pair `(theta, rho)` consisting of the final position and final momentum.
    """
    half_step_size = 0.5 * step_size
    step_inv_mass = step_size * inv_mass
    rho = rho + half_step_size * grad(theta)
    for _ in range(num_steps - 1):
        theta = theta + step_inv_mass * rho
        rho = rho + step_size * grad(theta)
    theta = theta + step_inv_mass * rho
    rho = rho + half_step_size * grad(theta)
    return theta, rho

def potential_energy(theta, logp):
    """Return the potential energy for the specified position.

    The potential energy is defined to be the negative log density, `-logp(theta)`.

    Args:
        theta: The position.
        logp: The target log density.
    Returns:
        The potential energy.
    """
    return -logp(theta)


def kinetic_energy(rho, inv_mass):
    """Return the kinetic energy for the specified momentum.

    The kinetic energy is defined to be `normal(rho | 0, inv(inv_mass))`,
    which for a fixed diagonal of the inverse mass matrix `inv_mass`,
    works out to `rho * inv_mass * rho`.

    Args:
        rho: The momentum.
        inv_mass: The diagonal of the inverse mass matrix.
    Returns:
        The kinetic energy.
    """
    return 0.5 * np.dot(inv_mass, rho**2)


def H(theta, rho, logp, inv_mass):
    """Return the Hamiltonian for the specified position and momentum.

    The Hamiltonian is the sum of the potential energy of the position
    plus the kinetic energy of the momentum.

    Args:
        theta: The position.
        rho: The momentum.
        logp: The target log density function.
        inv_mass: The diagonal of the inverse mass matrix.
    Returns:
        The Hamiltonian.
    """
    return potential_energy(theta, logp) + kinetic_energy(rho, inv_mass)


def stable_steps(theta0, rho0, logp, grad, inv_mass, macro_step, max_error):
    """Return the minimum steps into which `macro_step` must be broken to bound error by `max_error`.

    Only numbers of steps of the form `2**n` for `n <= 10` are considered. If
    `2**n` fails to conserve the error, the first element of the return will be `False`.

    Args:
        theta0: The initial position.
        rho0: The initial momentum.
        logp: The target log density function.
        grad: The gradient function for the target log density.
        inv_mass: The inverse mass matrix.
        macro_step: The largest step size considered.
        max_error: The max difference allowed among the Hamiltonians.
    Returns:
        A pair `(success, ell)` of a success flag and minimum steps preserving stability.
    """
    for n in range(11):
        theta, rho = theta0, rho0
        ell = 2**n
        step_size = macro_step / ell
        H_min = H_max = H(theta, rho, logp, inv_mass)
        half_step_size = 0.5 * step_size
        step_inv_mass = step_size * inv_mass
        rho = rho + half_step_size * grad(theta)
        for _ in range(ell - 1):
            theta = theta + step_inv_mass * rho
            g = grad(theta)
            rho = rho + half_step_size * g
            H_current = H(theta, rho, logp, inv_mass)
            H_min, H_max = min(H_min, H_current), max(H_max, H_current)
            rho = rho + half_step_size * g
        theta = theta + step_inv_mass * rho
        rho = rho + half_step_size * grad(theta)
        H_current = H(theta, rho, logp, inv_mass)
        H_min, H_max = min(H_min, H_current), max(H_max, H_current)
        if H_max - H_min <= max_error:
            return True, ell
    return False, ell


def choose_micro_steps(rng, ell_stable):
    """Generate a random step size around `ell_stable`

    The distribution is uniform among `ell_stable // 2`, `ell_stable`, and `ell_stable * 2`.

    Args:
        rng (np.Generator): A random number generator
        ell_stable (int > 0): The minimum number of steps preserving Hamiltonian below threshold.
    """
    return rng.choice([ell_stable // 2, ell_stable, ell_stable * 2])


def micro_steps_logp(ell, ell_stable):
    """Return the log probability of the given step size given the stable step size.

    The distribution is uniform among `ell_stable // 2`, `ell_stable`, and `ell_stable * 2`.

    Args:
        ell (int > 0): The chosen number of steps.
        ell_stable (int > 0): The minimum number of steps preserving Hamiltonian below threshold.
    """
    if ell == ell_stable or ell == ell_stable // 2 or ell == ell_stable * 2:
        return -np.log(3)
    return -np.inf


def extend_orbit(
    rng,
    going_backward,
    orbit,
    weights,
    logp,
    grad,
    inv_mass,
    macro_step,
    num_macro_steps,
    max_error,
):
    """Extend the orbit a fixed number of macro steps in the given direction from the current state.

    The orbit is extended by continually doubling and weights are calculated as described in
    the paper (see

    Args:
        rng (np.Generator): A random number generator
        going_backward (bool): `True` if evolving chain backward in time
        orbit (list(np.ndarray(D, ), np.ndarray(D,))): The current orbit.
        log_weights (lis(float)): The previous log weights.
        logp (function: np.ndarray(D,) -> float): The target log density function.
        grad (function: np.ndarray(D,) -> np.ndarray(D,)): The target gradient function.
        inv_mass (np.ndarray(D,)): The diagonal of the inverse mass matrix.
        macro_step (float > 0): The macro step size for NUTS.
        num_macro_steps (int > 0): The number of macro steps to take.
        max_error (float > 0): The maximum allowable energy error between states.
    Return:
        A pair `(orbit, weights)` of the new orbit and its weights.
    """
    if going_backward:
        theta, rho = orbit[0]
        weight = weights[0]
        rho = -rho
    else:
        theta, rho = orbit[-1]
        weight = weights[-1]
    new_orbit = []
    new_weights = []
    for _ in range(num_macro_steps):
        _, ell_stable = stable_steps(
            theta, rho, logp, grad, inv_mass, macro_step, max_error
        )
        ell = choose_micro_steps(rng, ell_stable)
        # TODO(carpenter): if ell = ell_stable, should already have theta, rho
        theta_next, rho_next = leapfrog(
            grad, theta, rho, macro_step / ell, inv_mass, ell
        )
        _, ell_stable_next = stable_steps(
            theta_next, -rho_next, logp, grad, inv_mass, macro_step, max_error
        )
        p_theta_rho_next = -H(theta_next, rho_next, logp, inv_mass)
        p_theta_rho = -H(theta, rho, logp, inv_mass)
        weight_next = (
            p_theta_rho_next
            - p_theta_rho
            + micro_steps_logp(ell, ell_stable_next)
            - micro_steps_logp(ell, ell_stable)
            + weight
        )
        new_orbit.append((theta_next, rho_next))
        new_weights.append(weight_next)
        theta, rho, weight = theta_next, rho_next, weight_next
    if going_backward:
        new_orbit, new_weights = new_orbit[::-1], new_weights[::-1]
    return new_orbit, new_weights


def walnuts_step(rng, theta, logp, grad, inv_mass, macro_step, max_nuts_depth, max_error):
    """Return the next state from WALNUTS given the current state `theta`.

    Sequentially drawing from WALNUTS given the previous draw forms a Markov chain, the stationary
    distribution of which has an unnormalized log density function `logp`.

    WALNUTS uses NUTS to determine the number of macro steps.  The step size within
    each macro step is locally adapted. The implementation follows the pseudocode in the
    appendix of (Bou-Rabee 2025).

    References:
        Nawaf Bou-Rabee, Bob Carpenter, Tore Selland Kleppe, and Sifan Liu. 2025.
        The within-orbit adaptive step-length no-U-turn sampler.

    Args:
        rng (np.random.Generator): A NumPy random number generator.
        theta (1D array_like (D,)): The current state vector.
        logp (function (D,) -> float): A continuously differentiable target log density function
        grad (function (D,) -> (D,)): The gradient function of the target log density
        inv_mass (1D array_like (D,)): The diagonal of the inverse mass matrix
        macro_step (float > 0): The macro step size for NUTS.
        max_nuts_depth (int > 0): The maximum number of doublings in NUTS evolution.
        max_error (float > 0): The maximum error in energy allowed between micro steps making up a macro step.
    Raises:
        ValueError: The shapes of `theta` and `inv_mass` do not match, or arguments are out of range.
    Returns:
        The next state vector in the Markov chain.
    """
    theta = np.array(theta)
    inv_mass = np.array(inv_mass)
    if theta.ndim != 1:
        raise ValueError("theta not a vector")
    if inv_mass.ndim != 1:
        raise ValueError("inv_mass not a vector")
    if theta.size != inv_mass.size:
        raise ValueError("size mismatch between theta and inv_mass")
    if not macro_step > 0:
        raise ValueError("non-positive macro_step")
    if not max_nuts_depth > 0:
        raise ValueError("non-positive max_nuts_depth")
    if not max_error > 0:
        raise ValueError("non-positive max_error")

    L_mass = inv_mass**-0.5
    D = theta.size

    rho = L_mass * rng.normal(size=D)  # rho ~ mvnormal(0, inverse(inv_mass))
    orbit, log_weights = [(theta, rho)], [-H(theta, rho, logp, inv_mass)]
    theta_selected = theta
    for depth in range(max_nuts_depth):
        num_macro_steps = 2**depth
        going_backward = bool(rng.binomial(1, 0.5))
        orbit_ext, log_weights_ext = extend_orbit(
            rng,
            going_backward,
            orbit,
            log_weights,
            logp,
            grad,
            inv_mass,
            macro_step,
            num_macro_steps,
            max_error,
        )
        if sub_uturn(orbit_ext, 0, num_macro_steps, inv_mass):
            break
        accept_prob = min(1.0, np.exp(sum(log_weights_ext) - sum(log_weights)))
        accept = bool(rng.binomial(1, accept_prob))
        if accept:
            p = softmax(log_weights_ext)
            if np.isnan(p).any():
                theta_selected, _ = rng.choice(orbit_ext)
            else:
                theta_selected, _ = rng.choice(orbit_ext, p=p)
        orbit = orbit_ext + orbit if going_backward else orbit + orbit_ext
        if uturn(orbit[0], orbit[-1], inv_mass):
            break
        log_weights = (
            log_weights_ext + log_weights
            if going_backward
            else log_weights + log_weights_ext
        )
    return theta_selected


def walnuts(
    rng,
    theta_init,
    logp,
    grad,
    inv_mass,
    macro_step,
    max_nuts_depth,
    max_error,
    iter_warmup,
    iter_sample,
):
    """Return a Markov chain of samples using the WALNUTS transition operator.

    Args:
        rng (np.random.Generator): A NumPy random number generator.
        theta_init (array_like (D,)): The starting state vector.
        logp (function (D,) -> float): A continuously differentiable target log density function.
        grad (function (D,) -> (D,)): The gradient of the log density.
        inv_mass (array_like float (D,)): The diagonal of the inverse mass matrix.
        macro_step (float > 0): The macro step size.
        max_nuts_depth (int > 0): The maximum number of doublings per transition.
        max_error (float > 0): Maximum error in Hamiltonian for adaptive step size.
        iter_warmup (int >= 0): The number of warmup transitions to discard.
        iter_sample (int > 0): The number of posterior draws to return.

    Returns:
        A NumPy array of shape (iter_sample, D) with one row per draw.
    """
    disable_runtime_warnings()
    theta = np.array(theta_init)
    D = theta.size
    draws = np.empty((iter_sample, D))
    for i in trange(iter_warmup + iter_sample, desc="WALNUTS"):
        theta = walnuts_step(
            rng,
            theta,
            logp,
            grad,
            inv_mass,
            macro_step,
            max_nuts_depth,
            max_error,
        )
        if i >= iter_warmup:
            draws[i - iter_warmup] = theta
    return draws
