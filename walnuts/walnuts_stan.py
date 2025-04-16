import bridgestan as bs
import walnuts as wn

def walnuts_stan(
    rng,
    theta_init,
    stan_file,
    data_file,
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
        theta_init (array_like float (D,)): The starting state vector.
        stan_file (str): The path to a file containing a Stan program.
        data_file (str): The path to a file containing JSON-formatted data.
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
    stan_model = bs.StanModel(stan_file, data_file)

    def logp(theta):
        return stan_model.log_density(theta)
    
    def grad(theta):
        _, grad = stan_model.log_density_gradient(theta)
        return grad

    return wn.walnuts(rng, theta_init, logp, grad, inv_mass, macro_sep, max_Nuts_depth, max_error, iter_warmup, iter_sample)
