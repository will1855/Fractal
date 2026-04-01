"""
Quantitative evaluation pipeline for levy_walk_model.py.

What it does:
- Generates N Lévy and Brownian walks
- Fits power-law and exponential models
- Compares log-likelihoods
- Computes search efficiency
- Writes results to CSV
- Saves summary plots

Run:
  python eval_levy_pipeline.py --n 30 --seed0 7 --out out_levy_eval
"""

from __future__ import annotations

import os
import csv
import argparse
from typing import List, Dict

import numpy as np
import matplotlib.pyplot as plt

import levy_walk_model as L


# ----------------------------
# Plot helpers
# ----------------------------

def plot_hist(vals, title, xlabel, out_path, bins=20):
    arr = np.asarray(vals, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return

    plt.figure()
    plt.hist(arr, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_model_preference(rows, out_path):
    levy_wins = sum(1 for r in rows if r["levy_pref_powerlaw"] == 1)
    brown_wins = sum(1 for r in rows if r["brown_pref_exp"] == 1)

    plt.figure()
    plt.bar(["Levy -> power-law", "Brownian -> exp"], [levy_wins, brown_wins])
    plt.title("Model preference across seeds")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed0", type=int, default=7)
    ap.add_argument("--out", type=str, default="out_levy_eval")

    # model params
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--mu", type=float, default=2.0)
    ap.add_argument("--xmin", type=float, default=0.25)
    ap.add_argument("--xmax", type=float, default=40.0)
    ap.add_argument("--scale", type=float, default=2.0)

    args = ap.parse_args()

    N = int(args.n)
    seed0 = int(args.seed0)
    out_dir = args.out

    os.makedirs(out_dir, exist_ok=True)
    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    rows: List[Dict] = []

    for i in range(N):
        seed = seed0 + i
        print(f"[{i+1}/{N}] seed={seed}")

        # Generate walks
        levy_pos, levy_steps = L.levy_walk_2d(
            seed=seed,
            n_steps=args.steps,
            mu=args.mu,
            xmin=args.xmin,
            xmax=args.xmax,
        )

        brown_pos, brown_steps = L.brownian_walk_2d(
            seed=seed,
            n_steps=args.steps,
            scale=args.scale,
            xmin=args.xmin,
        )

        # Fit models: Levy data
        mu_hat_levy = L.mle_powerlaw_mu(levy_steps, args.xmin)
        lam_hat_levy = L.mle_exponential_lambda(levy_steps, args.xmin)

        ll_power_levy = L.loglik_powerlaw(levy_steps, mu_hat_levy, args.xmin)
        ll_exp_levy = L.loglik_exponential(levy_steps, lam_hat_levy, args.xmin)

        levy_pref_powerlaw = int(ll_power_levy > ll_exp_levy)

        # Fit models: Brownian data
        mu_hat_brown = L.mle_powerlaw_mu(brown_steps, args.xmin)
        lam_hat_brown = L.mle_exponential_lambda(brown_steps, args.xmin)

        ll_power_brown = L.loglik_powerlaw(brown_steps, mu_hat_brown, args.xmin)
        ll_exp_brown = L.loglik_exponential(brown_steps, lam_hat_brown, args.xmin)

        brown_pref_exp = int(ll_exp_brown > ll_power_brown)

        # Search efficiency
        rng = np.random.default_rng(seed)
        prey = rng.uniform(-50, 50, size=(200, 2))

        levy_hits, levy_dist, levy_eff = L.search_efficiency(
            levy_pos, prey, detect_radius=1.0, steps=levy_steps
        )

        brown_hits, brown_dist, brown_eff = L.search_efficiency(
            brown_pos, prey, detect_radius=1.0, steps=brown_steps
        )

        row = {
            "seed": seed,
            "mu_hat_levy": float(mu_hat_levy),
            "lambda_hat_levy": float(lam_hat_levy),
            "loglik_powerlaw_levy": float(ll_power_levy),
            "loglik_exp_levy": float(ll_exp_levy),
            "levy_pref_powerlaw": int(levy_pref_powerlaw),

            "mu_hat_brown": float(mu_hat_brown),
            "lambda_hat_brown": float(lam_hat_brown),
            "loglik_powerlaw_brown": float(ll_power_brown),
            "loglik_exp_brown": float(ll_exp_brown),
            "brown_pref_exp": int(brown_pref_exp),

            "levy_efficiency": float(levy_eff),
            "brownian_efficiency": float(brown_eff),
            "levy_hits": int(levy_hits),
            "brownian_hits": int(brown_hits),
        }

        rows.append(row)

    # ----------------------------
    # Write CSV
    # ----------------------------
    csv_path = os.path.join(out_dir, "levy_metrics.csv")
    if rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    # ----------------------------
    # Plots
    # ----------------------------
    plot_hist(
        [r["mu_hat_levy"] for r in rows],
        "Estimated power-law exponent for Levy walks",
        "mu",
        os.path.join(plots_dir, "mu_hat_levy_hist.png"),
    )

    plot_hist(
        [r["levy_efficiency"] for r in rows],
        "Levy search efficiency",
        "Efficiency",
        os.path.join(plots_dir, "levy_efficiency_hist.png"),
    )

    plot_hist(
        [r["brownian_efficiency"] for r in rows],
        "Brownian search efficiency",
        "Efficiency",
        os.path.join(plots_dir, "brownian_efficiency_hist.png"),
    )

    plot_model_preference(
        rows,
        os.path.join(plots_dir, "model_preference.png"),
    )

    print("\nDone.")
    print("Wrote:")
    print(" ", csv_path)
    print(" ", plots_dir)


if __name__ == "__main__":
    main()