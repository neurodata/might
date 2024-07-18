import numpy as np


def linear(n, p, noise=False, coeffs=None):
    x = np.random.normal(size=(n, p))
    eps = np.random.normal(size=(n, p))
    if coeffs is None:
        coeffs = np.array([np.exp(-0.0022 * (i + 30)) for i in range(p)])
    y = x * coeffs + noise * eps

    return x, y


def exponential(n, p, noise=False, coeffs=None):
    x = np.random.normal(scale=3, size=(n, p))
    eps = np.random.normal(scale=3, size=(n, p))
    if coeffs is None:
        coeffs = np.array([np.exp(-0.022 * (i + 52)) for i in range(p)])
    y = np.exp(x * coeffs) - 1 + noise * eps

    return x, y


def cubic(n, p, noise=False, coeffs=None):
    x = np.random.normal(size=(n, p))
    eps = np.random.normal(size=(n, p))
    if coeffs is None:
        coeffs = np.array([np.exp(-0.031 * (i + 25)) for i in range(p)])

    x_coeffs = x * coeffs
    y = x_coeffs**3 + x_coeffs**2 + x_coeffs + noise * eps

    return x, y


def step(n, p, noise=False, coeffs=None):
    x = np.random.normal(size=(n, p))
    if coeffs is None:
        coeffs = np.array([np.exp(-0.0457 * (i + 10)) for i in range(p)])
    eps = np.random.normal(size=(n, p))

    x_coeff = ((x * coeffs) > 0.5) * 1
    y = x_coeff + noise * eps

    return x, y


def quadratic(n, p, noise=False, coeffs=None):
    x = np.random.normal(size=(n, p))
    if coeffs is None:
        coeffs = np.array([np.exp(-0.0325 * (i + 24)) for i in range(p)])
    eps = np.random.normal(size=(n, p))

    x_coeffs = x * coeffs
    y = x_coeffs**2 + noise * eps

    return x, y


def w_shaped(n, p, noise=False, coeffs=None):
    x = np.random.normal(scale=30, size=(n, p))
    if coeffs is None:
        coeffs = np.array([np.exp(-0.2735 * (i + 10)) for i in range(p)])
    eps = np.random.normal(scale=30, size=(n, p))
    x_coeffs = x * coeffs
    y = ((x_coeffs**4) - 7 * x_coeffs**2) + noise * eps

    return x, y


def logarithmic(n, p, noise=False, coeffs=None):
    rng = np.random.default_rng()
    if coeffs is None:
        coeffs = np.array([np.exp(-0.072 * (i + 10)) for i in range(p)])

    x = rng.normal(size=(n, p))
    eps = rng.normal(size=(n, p))

    y = np.log((x * coeffs + 1) ** 2) + noise * eps

    return x, y


def fourth_root(n, p, noise=False, coeffs=None):
    x = np.random.normal(size=(n, p))
    eps = np.random.normal(size=(n, p))
    if coeffs is None:
        coeffs = np.array([np.exp(-0.25 * (i + 50)) for i in range(p)])

    x_coeffs = x * coeffs
    y = 10 * np.abs(x_coeffs) ** 0.25 + noise * eps

    return x, y


def _sin(n, p, noise=False, period=4 * np.pi, coeffs=None):
    rng = np.random.default_rng()

    if period == 4 * np.pi and coeffs is None:
        coeffs = np.array([np.exp(-0.0095 * (i + 50)) for i in range(p)])
    elif period == 16 * np.pi and coeffs is None:
        coeffs = np.array([np.exp(-0.015 * (i + 50)) for i in range(p)])
    x = rng.normal(size=(n, p))
    eps = rng.normal(size=(n, p))

    y = np.sin(x * coeffs * period) + noise * eps

    return x, y


def sin_four_pi(n, p, noise=False, coeffs=None):
    return _sin(n, p, noise=noise, period=4 * np.pi, coeffs=coeffs)


def sin_sixteen_pi(n, p, noise=False, coeffs=None):
    return _sin(n, p, noise=noise, period=16 * np.pi, coeffs=coeffs)


def _square_diamond(n, p, noise=False, low=-1, high=1, period=-np.pi / 2, coeffs=None):
    u = np.random.uniform(low, high, size=(n, p))
    v = np.random.uniform(low, high, size=(n, p))
    eps = np.random.uniform(low, high, size=(n, p))
    if coeffs is None:
        coeffs = np.array([np.exp(-0.0042 * (i + 10)) for i in range(p)])

    x = u * np.cos(period) + v * np.sin(period)
    y = -u * coeffs * np.sin(period) + v * coeffs * np.cos(period) + eps * noise

    return x, y


def square(n, p, noise=False, low=-1, high=1, coeffs=None):
    return _square_diamond(
        n, p, noise=noise, low=low, high=high, period=-np.pi / 8, coeffs=coeffs
    )


def two_parabolas(n, p, noise=False, prob=0.5, coeffs=None):
    x = np.random.normal(size=(n, p))
    if coeffs is None:
        coeffs = np.array([np.exp(-0.00145 * (i + 50)) for i in range(p)])
    u = np.random.binomial(1, prob, size=(n, 1))
    eps = np.random.normal(size=(n, p))

    x_coeffs = x * coeffs
    y = (x_coeffs**2) * (u - 0.5) + noise * eps

    return x, y


def diamond(n, p, noise=False, low=-1, high=1, coeffs=None):
    return _square_diamond(
        n, p, noise=noise, low=low, high=high, period=-np.pi / 4, coeffs=coeffs
    )


def multimodal_independence(n, p, prob=0.5, sep1=3, sep2=2):
    rng = np.random.default_rng()

    sig = np.identity(p)
    u = rng.multivariate_normal(np.zeros(p), sig, size=n, method="cholesky")
    v = rng.multivariate_normal(np.zeros(p), sig, size=n, method="cholesky")
    u_2 = rng.binomial(1, prob, size=(n, p))
    v_2 = rng.binomial(1, prob, size=(n, p))

    x = u / sep1 + sep2 * u_2 - 1
    y = v / sep1 + sep2 * v_2 - 1

    return x, y


SIMS = {
    "linear": linear,
    "exponential": exponential,
    "cubic": cubic,
    "step": step,
    "quadratic": quadratic,
    "w_shaped": w_shaped,
    "logarithmic": logarithmic,
    "fourth_root": fourth_root,
    "sin_four_pi": sin_four_pi,
    "sin_sixteen_pi": sin_sixteen_pi,
    "square": square,
    "two_parabolas": two_parabolas,
    "diamond": diamond,
    "multimodal_independence": multimodal_independence,
}


def indep_sim(sim, n, p, **kwargs):
    if sim not in SIMS.keys():
        raise ValueError(
            "sim_name must be one of the following: {}".format(list(SIMS.keys()))
        )
    else:
        sim = SIMS[sim]

    return sim(n, p, **kwargs)
