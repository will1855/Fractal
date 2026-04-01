"""
Quantitative evaluation pipeline for lightning_dbm_model.py.

What it does:
- Generates N lightning trees (seeds seed0..seed0+N-1)
- Renders pretty images and binary masks
- Writes structural metrics + box-counting dimension to CSV
- Saves representative outputs for the first seed
- Saves summary plots across seeds

Run:
  python eval_lightning_pipeline.py --n 30 --seed0 7 --out out_lightning_eval
"""
from __future__ import annotations

import os
import csv
import argparse
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import lightning_dbm_model as L


# ----------------------------
# Plot helpers
# ----------------------------

def plot_hist(vals, title: str, xlabel: str, out_path: str, bins: int = 20):
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


def plot_box_count_rep(rep_box, out_path: str):
    if rep_box is None or len(rep_box["x"]) == 0:
        return

    x = np.array(rep_box["x"], dtype=float)
    y = np.array(rep_box["y"], dtype=float)

    if len(x) >= 6:
        x_fit = x[1:-1]
        y_fit = y[1:-1]
    else:
        x_fit = x
        y_fit = y

    m, c = np.polyfit(x_fit, y_fit, 1)

    plt.figure()
    plt.plot(x, y, marker="o", linestyle="-")
    plt.plot(x_fit, m * x_fit + c, linestyle="--")
    plt.xlabel("log(1/ε)")
    plt.ylabel("log N(ε)")
    plt.title(f"Box-counting (representative): slope D ≈ {m:.3f}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_branch_length_distribution_all(all_branch_lengths, out_path: str):
    vals = []
    for ls in all_branch_lengths:
        vals.extend(ls)

    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return

    plt.figure()
    plt.hist(vals, bins=40)
    plt.title("Lightning branch length distribution (all seeds)")
    plt.xlabel("Branch segment length (grid edges)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_counts_scatter(rows, out_path: str):
    if not rows:
        return

    seeds = [r["seed"] for r in rows]
    terminals = [r["n_terminals"] for r in rows]
    junctions = [r["n_junctions"] for r in rows]

    plt.figure()
    plt.plot(seeds, terminals, marker="o", linestyle="-", label="terminals")
    plt.plot(seeds, junctions, marker="o", linestyle="-", label="junctions")
    plt.xlabel("Seed")
    plt.ylabel("Count")
    plt.title("Terminal / junction counts across seeds")
    plt.legend()
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
    ap.add_argument("--out", type=str, default="out_lightning_eval")

    # simulation params
    ap.add_argument("--grid", type=int, default=384)
    ap.add_argument("--max_steps", type=int, default=14000)
    ap.add_argument("--eta", type=float, default=2.25)
    ap.add_argument("--relax_iters", type=int, default=60)
    ap.add_argument("--relax_every", type=int, default=2)
    ap.add_argument("--downward_bias", type=float, default=0.65)
    ap.add_argument("--retire_prob", type=float, default=0.0)

    # render params
    ap.add_argument("--render_size", type=int, default=1024)

    args = ap.parse_args()

    N = int(args.n)
    seed0 = int(args.seed0)
    out_dir = args.out

    os.makedirs(out_dir, exist_ok=True)
    masks_dir = os.path.join(out_dir, "masks")
    plots_dir = os.path.join(out_dir, "plots")
    reps_dir = os.path.join(out_dir, "representative")
    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(reps_dir, exist_ok=True)

    rows = []
    all_branch_lengths = []
    rep_box = None

    for i in range(N):
        seed = seed0 + i
        print(f"[{i+1}/{N}] seed={seed}")

        channel, parent, hit_ground, root, hit_cell = L.grow_lightning_dbm_v2(
            seed=seed,
            grid=args.grid,
            max_steps=args.max_steps,
            eta=args.eta,
            relax_iters=args.relax_iters,
            relax_every=args.relax_every,
            downward_bias=args.downward_bias,
            retire_prob=args.retire_prob,
        )

        graph = L.build_undirected_graph_from_parent(parent)
        terminals, junctions, path_nodes = L.classify_graph_nodes(graph)
        branch_lengths = L.extract_branch_lengths(graph)
        all_branch_lengths.append(branch_lengths)

        tortuosity = np.nan
        main_path_length = np.nan
        main_straight = np.nan
        trunk_set = None
        branch_depths = None
        main_path = []

        if hit_ground and hit_cell is not None:
            trunk_set = L.trunk_path_to_root(hit_cell, parent)
            branch_depths = L.branch_depths_from_trunk(parent, trunk_set)
            tortuosity, main_path, main_path_length, main_straight = L.compute_main_channel_tortuosity(
                hit_cell, parent
            )

        pretty_path = os.path.join(masks_dir, f"seed_{seed:04d}_pretty.png")
        mask_path = os.path.join(masks_dir, f"seed_{seed:04d}_mask.png")

        pretty_img, mask_img = L.render_lightning_graph(
            parent,
            grid=args.grid,
            render_size=args.render_size,
            margin=30,
            trunk_set=trunk_set,
            branch_depths=branch_depths,
            trunk_width=6,
            branch_width=4,
            width_decay=0.82,
            save_pretty_path=pretty_path,
            save_mask_path=mask_path,
        )

        if seed == seed0:
            raw_mask_path = os.path.join(reps_dir, "lightning_raw_mask.png")
            pretty_rep_path = os.path.join(reps_dir, "lightning_pretty.png")
            mask_rep_path = os.path.join(reps_dir, "lightning_mask_hr.png")

            Image.fromarray((channel.astype(np.uint8) * 255)).save(raw_mask_path)
            Image.fromarray(pretty_img).save(pretty_rep_path)
            Image.fromarray((mask_img * 255).astype(np.uint8)).save(mask_rep_path)

            if len(main_path) >= 2:
                overlay_path = os.path.join(reps_dir, "lightning_main_channel_overlay.png")
                L.render_main_path_overlay(
                    pretty_img,
                    main_path,
                    grid=args.grid,
                    render_size=args.render_size,
                    margin=30,
                    overlay_width=8,
                    save_path=overlay_path,
                )

        D, r2, bx, by = L.box_count_fractal_dimension(mask_img)
        if seed == seed0:
            rep_box = {"x": bx, "y": by}

        row = {
            "seed": seed,
            "hit_ground": int(hit_ground),
            "n_cells": int(channel.sum()),
            "n_graph_nodes": int(len(graph)),
            "n_edges": int(len(parent) - 1),
            "n_terminals": int(len(terminals)),
            "n_junctions": int(len(junctions)),
            "n_path_nodes": int(len(path_nodes)),
            "n_branch_segments": int(len(branch_lengths)),
            "branch_len_mean": float(np.mean(branch_lengths)) if len(branch_lengths) else 0.0,
            "branch_len_median": float(np.median(branch_lengths)) if len(branch_lengths) else 0.0,
            "branch_len_max": int(np.max(branch_lengths)) if len(branch_lengths) else 0,
            "tortuosity": float(tortuosity) if np.isfinite(tortuosity) else "",
            "main_path_length": float(main_path_length) if np.isfinite(main_path_length) else "",
            "main_straight_length": float(main_straight) if np.isfinite(main_straight) else "",
            "D_box": float(D),
            "box_r2": float(r2),
            "pretty_png": os.path.relpath(pretty_path, out_dir),
            "mask_png": os.path.relpath(mask_path, out_dir),
        }
        rows.append(row)

    # -----------------------------
    # Write CSV
    # -----------------------------
    csv_path = os.path.join(out_dir, "lightning_metrics.csv")
    if rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    # -----------------------------
    # Summary plots
    # -----------------------------
    plot_box_count_rep(rep_box, os.path.join(plots_dir, "box_counting_representative.png"))
    plot_branch_length_distribution_all(
        all_branch_lengths,
        os.path.join(plots_dir, "branch_length_distribution_all.png"),
    )
    plot_counts_scatter(rows, os.path.join(plots_dir, "terminal_junction_counts.png"))

    plot_hist(
        [r["tortuosity"] for r in rows if r["tortuosity"] != ""],
        "Main-channel tortuosity across seeds",
        "Tortuosity",
        os.path.join(plots_dir, "tortuosity_hist.png"),
        bins=20,
    )

    plot_hist(
        [r["D_box"] for r in rows],
        "Box-counting fractal dimension across seeds",
        "D",
        os.path.join(plots_dir, "box_dimension_hist.png"),
        bins=20,
    )

    plot_hist(
        [r["branch_len_mean"] for r in rows],
        "Mean branch length across seeds",
        "Mean branch length",
        os.path.join(plots_dir, "branch_len_mean_hist.png"),
        bins=20,
    )

    plot_hist(
        [r["n_terminals"] for r in rows],
        "Terminal count across seeds",
        "Terminal count",
        os.path.join(plots_dir, "terminal_count_hist.png"),
        bins=20,
    )

    plot_hist(
        [r["n_junctions"] for r in rows],
        "Junction count across seeds",
        "Junction count",
        os.path.join(plots_dir, "junction_count_hist.png"),
        bins=20,
    )

    print("\nDone.")
    print("Wrote:")
    print(" ", csv_path)
    print(" ", plots_dir)
    print(" ", reps_dir)


if __name__ == "__main__":
    main()