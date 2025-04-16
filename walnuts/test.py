import numpy as np
import pandas as pd
from plotnine import ggplot, aes, geom_point, theme_minimal, labs

from walnuts import walnuts, walnuts_chain
from targets import standard_normal_lpdf, standard_normal_grad


def main():
    rng = np.random.default_rng(seed=123)
    D = 2
    theta_init = np.zeros(D)
    inv_mass = np.ones(D)
    macro_step = 1.0
    max_nuts_depth = 5
    max_error = 0.1
    iter_warmup = 100
    iter_sample = 1000

    draws = walnuts_chain(
        rng,
        theta_init,
        standard_normal_lpdf,
        standard_normal_grad,
        inv_mass,
        macro_step,
        max_nuts_depth,
        max_error,
        iter_warmup,
        iter_sample,
    )

    means = draws.mean(axis=0)
    stds = draws.std(axis=0)

    print("Posterior means:", means)
    print("Posterior stds:", stds)

    df = pd.DataFrame({
        "x1": draws[:, 0],
        "x2": draws[:, 1]
    })

    p = (
        ggplot(df, aes("x1", "x2"))
        + geom_point(alpha=0.5, size=1)
        + theme_minimal()
        + labs(title="WALNUTS Samples from Standard Normal", x="x1", y="x2")
    )
    print(p)


if __name__ == "__main__":
    main()
