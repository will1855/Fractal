from __future__ import annotations
import math
import numpy as np
import matplotlib.pyplot as plt


# Sampling

def sample_truncated_powerlaw_steps(
    rng: np.random.Generator,
    n: int,
    mu: float,
    xmin: float,
    xmax: float,
) -> np.ndarray:
    """
    Sample step lengths from truncated power law:
        p(l) ∝ l^{-mu},  xmin <= l <= xmax
    """
    if mu <= 1.0:
        raise ValueError("mu must be > 1")
    if xmin <= 0 or xmax <= xmin:
        raise ValueError("Need 0 < xmin < xmax")

    u = rng.random(n)
    a = xmin ** (1.0 - mu)
    b = xmax ** (1.0 - mu)
    x_pow = a + u * (b - a)
    return x_pow ** (1.0 / (1.0 - mu))


def sample_exponential_steps(
    rng: np.random.Generator,
    n: int,
    scale: float,
    xmin: float = 0.0,
) -> np.ndarray:
    """
    Brownian-style control: exponential step lengths.
    """
    if scale <= 0:
        raise ValueError("scale must be > 0")
    return xmin + rng.exponential(scale=scale, size=n)


# Walk generation

def random_walk_2d(
    rng: np.random.Generator,
    steps: np.ndarray,
) -> np.ndarray:
    theta = rng.uniform(0.0, 2.0 * math.pi, size=len(steps))
    dx = steps * np.cos(theta)
    dy = steps * np.sin(theta)

    pos = np.zeros((len(steps) + 1, 2), dtype=float)
    pos[1:, 0] = np.cumsum(dx)
    pos[1:, 1] = np.cumsum(dy)
    return pos


def levy_walk_2d(
    seed: int = 7,
    n_steps: int = 8000,
    mu: float = 2.0,
    xmin: float = 0.25,
    xmax: float = 40.0,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    steps = sample_truncated_powerlaw_steps(rng, n_steps, mu=mu, xmin=xmin, xmax=xmax)
    pos = random_walk_2d(rng, steps)
    return pos, steps


def brownian_walk_2d(
    seed: int = 7,
    n_steps: int = 8000,
    scale: float = 2.0,
    xmin: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    steps = sample_exponential_steps(rng, n_steps, scale=scale, xmin=xmin)
    pos = random_walk_2d(rng, steps)
    return pos, steps


# CCDF
def empirical_ccdf(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns sorted x and empirical CCDF P(X >= x).
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    x = x[x > 0]
    x = np.sort(x)

    n = len(x)
    if n == 0:
        raise ValueError("No positive values for CCDF")

    ccdf = 1.0 - (np.arange(n) / n)
    return x, ccdf


# MLE fitting
def mle_powerlaw_mu(x: np.ndarray, xmin: float) -> float:
    """
    Continuous power-law MLE above xmin:
        mu_hat = 1 + n / sum(log(x/xmin))
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    x = x[x >= xmin]

    if len(x) < 5:
        raise ValueError("Need more samples above xmin")

    return 1.0 + len(x) / np.sum(np.log(x / xmin))


def mle_exponential_lambda(x: np.ndarray, xmin: float) -> float:
    """
    Exponential MLE on [xmin, inf):
        lambda_hat = 1 / mean(x - xmin)
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    x = x[x >= xmin]

    if len(x) < 5:
        raise ValueError("Need more samples above xmin")

    shifted = x - xmin
    mean_shifted = np.mean(shifted)
    if mean_shifted <= 0:
        raise ValueError("Mean(x - xmin) must be > 0")

    return 1.0 / mean_shifted


def loglik_powerlaw(x: np.ndarray, mu: float, xmin: float) -> float:
    """
    Log-likelihood for continuous power law on [xmin, inf).
    """
    x = np.asarray(x, dtype=float)
    x = x[x >= xmin]
    n = len(x)
    if n == 0 or mu <= 1.0:
        return -np.inf

    return n * np.log(mu - 1.0) + n * (mu - 1.0) * np.log(xmin) - mu * np.sum(np.log(x))


def loglik_exponential(x: np.ndarray, lam: float, xmin: float) -> float:
    """
    Log-likelihood for shifted exponential on [xmin, inf).
    """
    x = np.asarray(x, dtype=float)
    x = x[x >= xmin]
    n = len(x)
    if n == 0 or lam <= 0:
        return -np.inf

    shifted = x - xmin
    return n * np.log(lam) - lam * np.sum(shifted)


# Search efficiency experiment

def make_prey_field(
    rng: np.random.Generator,
    n_targets: int,
    field_size: float,
) -> np.ndarray:
    """
    Uniform random prey points in square [-field_size, field_size]^2
    """
    return rng.uniform(-field_size, field_size, size=(n_targets, 2))


def search_efficiency(
    pos: np.ndarray,
    prey: np.ndarray,
    detect_radius: float,
    steps: np.ndarray,
) -> tuple[int, float, float]:
    """
    Returns:
    hits = number of unique prey found
    total_distance = total path length
    hits_per_distance = efficiency
    """
    found = np.zeros(len(prey), dtype=bool)

    for p in pos:
        d2 = np.sum((prey - p) ** 2, axis=1)
        found |= (d2 <= detect_radius ** 2)

    hits = int(np.sum(found))
    total_distance = float(np.sum(steps))
    efficiency = hits / total_distance if total_distance > 0 else 0.0
    return hits, total_distance, efficiency


# Plotting helpers

def plot_walk(pos: np.ndarray, title: str) -> None:
    plt.figure()
    plt.plot(pos[:, 0], pos[:, 1], linewidth=0.7)
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.tight_layout()


def plot_ccdf(steps_levy: np.ndarray, steps_brown: np.ndarray) -> None:
    xL, cL = empirical_ccdf(steps_levy)
    xB, cB = empirical_ccdf(steps_brown)

    plt.figure()
    plt.loglog(xL, cL, label="Lévy")
    plt.loglog(xB, cB, label="Brownian")
    plt.xlabel("Step length l")
    plt.ylabel("P(L ≥ l)")
    plt.title("Step-length CCDF")
    plt.legend()
    plt.tight_layout()


def plot_search_bar(levy_eff: float, brown_eff: float) -> None:
    plt.figure()
    plt.bar(["Lévy", "Brownian"], [levy_eff, brown_eff])
    plt.ylabel("Hits per unit distance")
    plt.title("Search efficiency")
    plt.tight_layout()


# Main

def main() -> None:
    seed = 7
    n_steps = 8000

    # Lévy settings
    mu_true = 2.0
    xmin = 0.25
    xmax = 40.0

    # Brownian settings
    brown_scale = 2.0

    # --- Generate walks ---
    levy_pos, levy_steps = levy_walk_2d(
        seed=seed,
        n_steps=n_steps,
        mu=mu_true,
        xmin=xmin,
        xmax=xmax,
    )

    brown_pos, brown_steps = brownian_walk_2d(
        seed=seed + 1,
        n_steps=n_steps,
        scale=brown_scale,
        xmin=xmin,
    )

    # --- Fit distributions ---
    fit_xmin = xmin

    mu_hat_levy = mle_powerlaw_mu(levy_steps, fit_xmin)
    lam_hat_levy = mle_exponential_lambda(levy_steps, fit_xmin)

    mu_hat_brown = mle_powerlaw_mu(brown_steps, fit_xmin)
    lam_hat_brown = mle_exponential_lambda(brown_steps, fit_xmin)

    ll_pl_levy = loglik_powerlaw(levy_steps, mu_hat_levy, fit_xmin)
    ll_exp_levy = loglik_exponential(levy_steps, lam_hat_levy, fit_xmin)

    ll_pl_brown = loglik_powerlaw(brown_steps, mu_hat_brown, fit_xmin)
    ll_exp_brown = loglik_exponential(brown_steps, lam_hat_brown, fit_xmin)

    print("\n=== Lévy synthetic data ===")
    print(f"true mu         = {mu_true:.3f}")
    print(f"fitted mu       = {mu_hat_levy:.3f}")
    print(f"fitted lambda   = {lam_hat_levy:.3f}")
    print(f"loglik powerlaw = {ll_pl_levy:.3f}")
    print(f"loglik exp      = {ll_exp_levy:.3f}")
    print("better fit      =", "power law" if ll_pl_levy > ll_exp_levy else "exponential")

    print("\n=== Brownian synthetic data ===")
    print(f"fitted mu       = {mu_hat_brown:.3f}")
    print(f"fitted lambda   = {lam_hat_brown:.3f}")
    print(f"loglik powerlaw = {ll_pl_brown:.3f}")
    print(f"loglik exp      = {ll_exp_brown:.3f}")
    print("better fit      =", "power law" if ll_pl_brown > ll_exp_brown else "exponential")

    # --- Search efficiency ---
    rng = np.random.default_rng(123)
    prey = make_prey_field(rng, n_targets=300, field_size=300.0)
    detect_radius = 5.0

    levy_hits, levy_dist, levy_eff = search_efficiency(levy_pos, prey, detect_radius, levy_steps)
    brown_hits, brown_dist, brown_eff = search_efficiency(brown_pos, prey, detect_radius, brown_steps)

    print("\n=== Search efficiency ===")
    print(f"Lévy    : hits={levy_hits}, distance={levy_dist:.2f}, hits/dist={levy_eff:.6f}")
    print(f"Brownian: hits={brown_hits}, distance={brown_dist:.2f}, hits/dist={brown_eff:.6f}")

    # --- Plots ---
    plot_walk(levy_pos, f"Lévy walk (mu={mu_true})")
    plot_walk(brown_pos, "Brownian walk")
    plot_ccdf(levy_steps, brown_steps)
    plot_search_bar(levy_eff, brown_eff)

    plt.show()


if __name__ == "__main__":
    main()