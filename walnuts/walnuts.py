# inv_mass is going to be diagonal

def uturn_state_(theta_rho_bk, theta_rho_fw, inv_mass):
    theta_bk, rho_bk = theta_rho_bk
    theta_fw, rho_fw = theta_rho_fw
    diff = inv_mass * (theta_fw - theta_bk)
    return np.dot(rho_bk, diff) < 0 or np.dot(rho_fw, diff) < 0


def sub_uturn_idx_(orbit, start, end, inv_mass):
    size = end - start
    if size < 2:
        return False
    mid = start + size // 2
    if uturn_state_(orbit[start], orbit[end - 1], inv_mass):
        return True
    return size >= 4 and (
        sub_uturn_idx_(orbit, start, mid, inv_mass)
        or sub_uturn_idx_(orbit, mid, end, inv_mass)
    )


def uturn(orbit, inv_mass):
    """Return `True` if there is a U-turn between the ends of the orbit.

    A U-turn between positions `(theta_bk, rho_bk)` and `(theta_fw, rho_fw)` occurs if
    `dot(theta_fw - theta_bk, rho_bk) < 0` or `dot(theta_fw - theta_bk, rho_fw) < 0`.

    Args:
        orbit: A list of pairs representing position and momentum in phase space.
        inv_mass: The diagonal of the inverse mass matrix, used as a metric in the U-turn condition.
    Returns:
        `True` if there is a U-turn between the ends of the orbit.
    """
    return orbit.size >= 2 and uturn_state_(orbit[0], orbit[-1], inv_mass)


def sub_uturn(orbit, inv_mass):
    """Return `True` if the orbit has a U-turn or one its half orbits has a sub-U-turn.

    A an orbit has a sub-U-turn if (a) the orbit has a U-turn, (b) the
    first half of the orbit has a sub-U-turn, or (c) the second half of
    the orbit has a sub-U-turn.

    Args:
        orbit: A list of pairs representing position and momentum in phase space of size `2^K`for some `K >= 0`.
        inv_mass: The diagonal of the inverse mass matrix, used as a metric in the U-turn condition.
    Returns:
        `True` if there is a sub-U-turn between the ends of the orbit.
    """
    return sub_uturn_idx_(orbit, 0, orbit.size, inv_mass)


def leapfrog_step(grad, theta, rho, step_size, inv_mass):
    """Return the result of a single leapfrog step from `(theta, rho)`.
    
    Args:
        grad: The gradient function for the target log density.
        theta: The initial position.
        rho: The initial momentum.
        step_size: The interval of time discretization of the dynamics simulator.
        inv_mass: The diagonal of the inverse mass matrix
    Returns:
        A pair `(theta_f, rho_f)` consisting of the final position and final momentum.
    """
    half_step_size = 0.5 * step_size
    rho_half = rho + half_step_size * grad(theta)
    theta_full = theta + step_size * inv_mass * rho_half
    rho_full = rho_half + half_step_size * grad(theta)
    return theta_full, rho_full

def kinetic_energy(rho, inv_mass):
    """Return the kinetic energy for the specified momentum.

    The kinetic energy is defined to be normal(rho | 0, inv(inv_mass)),
    which for a fixed diagonal of the inverse mass matrix `inv_mass`,
    works out to `rho * inv_mass * rho`.

    Args:
        rho: The momentum.
        inv_mass: The diagonal of the inverse mass matrix.
    Returns:
        The kinetic energy.
    """
    return 0.5 * np.dot(rho, (inv_mass * rho))

def potential_energy(theta, log_p):
    """Return the potential energy for the specified position.

    The potential energy is the negative log density, `-log_p(theta)`.

    return -log_p(theta)

def H(theta, rho, log_p, inv_mass):
    return potential_energy(theta, log_p) + kinetic_energy(rho, inv_mass)

def stable_steps(theta0, rho0, log_p, grad, inv_mass, macro_step, max_error):
    for n in range(11):
        theta, rho = theta0, rho0
        ell = 2**n
        step_size = macro_step / ell
        H_min = H_max = H(theta, rho, log_p, inv_mass)
        for j in range(ell):
            theta, rho = leapfrog_step(grad, theta, rho, step_size, inv_mass)
            H_current = H(theta, rho, log_p, inv_mass)
            H_min, H_max = min([H_min, H_current]), max([H_max, H_current])
        if H_max - H_min <= max_error:
            return ell
    return ell
        
