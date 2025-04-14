from scipy.special import softmax

def uturn(theta_rho1, theta_rho2, inv_mass):
    """Return `True` if there is a U-turn between the positions and momentums.

    There is a U-turn between `(theta1, rho1)` and `(theta2, rho2)` 
    if `dot(theta_fw - theta_bk, rho_bk) < 0` `dot(theta_fw - theta_bk, rho_fw) < 0`.

    Args:
        theta_rho1: The starting state.
        theta_rho2: The ending state.
        inv_mass: The diagonal of the inverse mass matrix, used as a Euclidean metric in the U-turn condition.
    Returns:
        `True` if there is a U-turn between the states.
    """
    theta1, rho1 = theta_rho1
    theta2, rho2 = theta_rho2
    diff = inv_mass * (theta1 - theta2)
    return np.dot(rho1, diff) < 0 or np.dot(rho2, diff) < 0


def sub_uturn(orbit, start, end, inv_mass):
    """Return `True` if there is a sub-U-turn in `orbit[start:end]`.

    An orbit has a sub-U-turn if (a) the orbit has a U-turn, (b) the
    first half of the orbit has a sub-U-turn, or (c) the second half of
    the orbit has a sub-U-turn. The inverse mass matrix is used as a Euclidean
    metric in the calculation (see the `uturn()` function for a definition).

    Args:
        orbit: A list of pairs representing position and momentum in phase space of size `2^K`for some `K >= 0`.
        start: The first position to consider in the orbit.
        end: The last position to consider in the orbit.
        inv_mass: The diagonal of the diagonal inverse mass matrix.
    Returns:
        `True` if there is a sub-U-turn between the ends of the orbit.
    """
    size = end - start
    if size >= 2 and uturn(orbit[start], orbit[end - 1], inv_mass):
        return True
    if size >= 4:
        mid = start + size // 2
        return (sub_uturn(orbit, start, mid, inv_mass) or
                     or sub_uturn(orbit, mid, end, inv_mass))
    return False


def leapfrog_step(grad, theta, rho, step_size, inv_mass):
    """Return the result of a single leapfrog step from `(theta, rho)`.
    
    Args:
        grad: The gradient function for the target log density.
        theta: The initial position.
        rho: The initial momentum.
        step_size: The interval of time discretization of the dynamics simulator.
        inv_mass: The diagonal of the diagonal inverse mass matrix
    Returns:
        A pair `(theta_f, rho_f)` consisting of the final position and final momentum.
    """
    half_step_size = 0.5 * step_size
    rho_half = rho + half_step_size * grad(theta)
    theta_full = theta + step_size * inv_mass * rho_half
    rho_full = rho_half + half_step_size * grad(theta_full)
    return theta_full, rho_full

def potential_energy(theta, logp):
    """Return the potential energy for the specified position.

    The potential energy is defined to be the negative log density, `-logp(theta)`.

    Args:
        theta: The position.
        logp: The target log density.
    Returns:
        The potnetial energy.
    """
    return -logp(theta)


def kinetic_energy(rho, inv_mass):
    """Return the kinetic energy for the specified momentum.

    The kinetic energy is defined to be `normal(rho | 0, inv(inv_mass))`,
    which for a fixed diagonal of the inverse mass matrix `inv_mass`,
    works out to `rho * inv_mass * rho`.

    Args:
        rho: The momentum.
        inv_mass: The diagonal of the diagonal inverse mass matrix.
    Returns:
        The kinetic energy.
    """
    return 0.5 * np.dot(rho, inv_mass * rho)
 

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
        A pair `(success, ell)` of minimum number of steps `ell` and a flag `success`
        which is `True` if `ell` is stable.
    """    
    for n in range(11):
        theta, rho = theta0, rho0
        ell = 2**n
        step_size = macro_step / ell
        H_min = H_max = H(theta, rho, logp, inv_mass)
        for j in range(ell):
            theta, rho = leapfrog_step(grad, theta, rho, step_size, inv_mass)
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
        ell_stable (float > 0): The maximum step size preserving Hamiltonian below threshold.
    """
    return rng.choice([ell_stable // 2, ell_stable, ell_stable * 2])

def micro_steps_logp(ell, ell_stable):
    """Return the log probability of the given step size given the stable step size.

    The distribution is uniform among `ell_stable // 2`, `ell_stable`, and `ell_stable * 2`. 

    Args:
        ell (float > 0): The chosen step size.
        ell_stable (float > 0): The maximum step size preserving Hamiltonian below threshold.
    """
    if ell == ell_stable or ell == ell_stable // 2 or ell == ell_stable * 2:
        return -np.log(3)
    return np.NINF

def extend_orbit_forward(rng, going_backward, theta, rho, weight, logp, grad, inv_mass,
                             macro_step, num_macro_steps, max_error):
    """Extend the orbit a fixed number of macro steps in the given direction from the current state.
    
    The orbit is extended by continually doubling and weights are calculated as described in
    the paper (see 
    
    Args: 
        rng (np.Generator): A random number generator
        going_backward (bool): `True` if evolving chain backward in time
        theta (np.ndarray (D,)): The previous position.
        rho (np.ndarray (D,)): The previous momentum.
        weight (float): The previous log weight.
        logp (function: np.ndarray(D,) -> float): The target log density function.
        grad (function: np.ndarray(D,) -> np.ndarray(D,)): The target gradient function.
        inv_mass (np.ndarray(D,)): The diagonal of the diagonal inverse mass matrix.
        macro_step (float > 0): The macro step size for NUTS.
        num_macro_steps (int > 0): The number of macro steps to take.
        max_error (float > 0): The maximum allowable energy error between states.
    Return: 
        A pair `(orbit, weights)` of the new orbit and its weights.
    """
    if going_backward:
        rho = -rho
    new_orbit = []
    new_weights = []
    for _ in range(num_macro_steps):
        ell_stable = stable_steps(theta, rho, logp, grad, inv_mass, macro_step, max_error)
        ell = choose_micro_steps(rng, ell_stable)
        theta_next, rho_next = leapfrog_step(theta, rho, logp, grad, inv_mass, ell, macro_step / ell)
        ell_return = stable_steps(theta_next, rho_next, logp, grad, inv_mass, macro_step, max_error)
        ell_stable_back = stable_steps(theta_next, -rho_next, logp, grad, inv_mass, macro_step, max_error)
        weight_next = (
            -H(theta_next, rho_next) - -H(theta, rho)
            + micro_steps_logp(ell, ell_stable_return) - micro_steps_logp(ell, ell_stable_back)
            + weight
        )
        new_orbit.append((theta_next, rho_next))
        new_weights.append(weight_next)
        theta, rho, weight = theta_next, rho_next, weight_next
    if going_backward:
        new_orbit, new_weights = np.reverse(new_orbit), np.reverse(new_weights)
    return new_orbit, new_weights
    
def walnuts(rng, theta, logp, grad, inv_mass, macro_step, max_nuts_depth, max_error):
    """Return the next state from WALNUTS given the currrent state `theta`.
    
    Sequentially drawing from WALNUTS given the previous draw forms a Markov chain, the stationy
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
    if theta.shape != inv_mass.shape:
        raise ValueError("shape mismatch between theta and inv_mass")
    if theta.shape[0] != theta.size:
        raise ValueError("theta not a vector")
    if inv_mass.shape[0] != inv_mass.size:
        raise ValueError("inv_mass not a vector")
    if not macro_step > 0:
        raise ValueError("non-positive macro_step")
    if not max_nuts_depth > 0:
        raise ValueError("non-positive max_nuts_depth")
    if not max_error > 0:
        raise ValueError("non-positive max_error")

    L_mass = inv_mass**-0.5
    D = theta.shape

    rho = L_mass * rng.normal(size=D)  # rho ~ mvnormal(0, inverse(inv_mass))
    orbit, log_weights = [(theta, rho)], [-H(theta, rho, logp, inv_mass)]
    theta_selected = theta
    for depth in range(max_nuts_depth):
        num_macro_steps = 2**depth
        going_backward = rng.binomial(1, 0.5)
        orbit_ext, log_weights_ext = extend_orbit(rng, going_backward, orbit, log_weights, 
                                                      logp, grad, inv_mass, macro_step, num_macro_steps, max_error)
        if sub_uturn(orbit_ext, 0, num_macro_steps, inv_mass):
            break
        accept_prob = max(1.0, np.exp(sum(log_weights_ext) - sum(log_weights)))
        accept = rng.binomial(1, accept_prob)
        if accept:
            theta_selected, _ = rng.choice(orbit_ext, p=softmax(log_weights_ext))
        orbit = np.concatenate([orbit_ext, orbit] if going_backward else [orbit, orbit_ext])
        if uturn(orbit[0], orbit[-1], inv_mass):
            break
        log_weights = np.concatenate([log_weights_ext, log_weights] if going_backward else [log_weights, log_weights_ext])
    return theta_selected
