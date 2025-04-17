import numpy as np
import pandas as pd
from plotnine import ggplot, aes, geom_point, ggtitle, scale_y_continuous
from walnuts import walnuts
from targets import standard_normal_lpdf, standard_normal_grad
from targets import funnel_lpdf, funnel_grad


def main():
    rng = np.random.default_rng(seed=123)
    D = 2
    theta_init = np.zeros(D)
    inv_mass = np.ones(D)
    macro_step = 10.0
    max_nuts_depth = 10
    max_error = 0.001
    iter_warmup = 0
    iter_sample = 500

    draws = walnuts(
        rng,
        theta_init,
        funnel_lpdf,
        funnel_grad,
        inv_mass,
        macro_step,
        max_nuts_depth,
        max_error,
        iter_warmup,
        iter_sample,
    )

    means = draws.mean(axis=0)
    stds = draws.std(axis=0)

    print("Posterior means:\n", means)
    print("\n")
    print("Posterior stds:\n", stds)

    df = pd.DataFrame({
        "log_scale": draws[:, 0],
        "coefficient": draws[:, 1]
    })

    plot = (
        ggplot(df, aes("coefficient", "log_scale"))
        + geom_point(alpha=0.5, size=1)
        + ggtitle("WALNUTS draws")
        + scale_y_continuous(limits=[-12, 12], breaks = [-12, -9, -6, -3, 0, 3, 6, 9, 12])
    )
    plot.show()


if __name__ == "__main__":
    main()
