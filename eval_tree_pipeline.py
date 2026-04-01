"""
Quantitative evaluation pipeline for fractal_tree_model.py.

What it does:
- Generates N trees (seeds seed0..seed0+N-1)
- Renders 3 orthographic silhouette views each (yaw = 0, 45, 90 deg)
- Saves binary masks (PNG) for each seed/view
- Writes silhouette metrics + box-counting dimension to CSV
- Computes Strahler order distribution from geometry
- Validates pipe model: radius vs downstream tip count (log-log fit)
- Saves key plots (PNG)

Run:
  python eval_tree_pipeline.py --n 30 --seed0 7 --out out_tree_eval
"""
from __future__ import annotations

import os
import math
import csv
import argparse
from types import SimpleNamespace
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

import importlib.util
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
TREE_PATH = HERE / "fractal_tree_model.py"

spec = importlib.util.spec_from_file_location("treemodel", TREE_PATH)
T = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules[spec.name] = T
spec.loader.exec_module(T)


# ----------------------------
# Compatibility layer
# ----------------------------

def build_tree(seed: int):
    """
    Build tree and convert it to a simple node/edge representation
    compatible with the original evaluation code.
    """
    params = T.Params(seed=seed)
    root = T.build_tree(params)

    nodes = []
    edges = []

    def walk(node, parent_id=None):
        my_id = len(nodes)
        nodes.append(SimpleNamespace(
            id=my_id,
            pos=np.array(node.pos, dtype=float),
            end=np.array(node.end if node.end is not None else node.pos, dtype=float),
            depth=int(node.depth),
            radius=float(node.radius if node.radius is not None else 0.0),
        ))
        if parent_id is not None:
            edges.append(SimpleNamespace(
                a=parent_id,
                b=my_id,
                radius=float(node.radius if node.radius is not None else 0.0),
                start=np.array(node.pos, dtype=float),
                end=np.array(node.end if node.end is not None else node.pos, dtype=float),
            ))
        for ch in node.children:
            walk(ch, my_id)

    walk(root, None)
    return root, nodes, edges, params


# ----------------------------
# Rendering (geometry -> mask)
# ----------------------------

def yaw_rotate(points: np.ndarray, yaw_deg: float) -> np.ndarray:
    a = math.radians(float(yaw_deg))
    c, s = math.cos(a), math.sin(a)
    R = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=float)
    return points @ R.T


def render_mask(nodes, edges, params, yaw_deg: float, out_png: str,
                W: int = 1024, H: int = 1024, margin: int = 60) -> np.ndarray:
    """
    Returns a boolean mask (H,W) where True = tree pixels.
    Writes PNG to out_png.

    Orthographic projection after yaw rotation around the vertical axis.
    """
    img = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(img)

    if not edges:
        os.makedirs(os.path.dirname(out_png), exist_ok=True)
        img.save(out_png)
        return np.zeros((H, W), dtype=bool)

    # Rotate all segment endpoints for this yaw
    segs = []
    pts_all = []
    for ed in edges:
        p0 = np.array(ed.start, dtype=float) * params.world_scale
        p1 = np.array(ed.end, dtype=float) * params.world_scale
        p0r = yaw_rotate(p0[None, :], yaw_deg)[0]
        p1r = yaw_rotate(p1[None, :], yaw_deg)[0]
        segs.append((p0r, p1r, float(ed.radius)))
        pts_all.append(p0r)
        pts_all.append(p1r)

    pts_all = np.array(pts_all, dtype=float)
    xs = pts_all[:, 0]
    ys = pts_all[:, 1]

    minx, maxx = xs.min(), xs.max()
    miny, maxy = ys.min(), ys.max()
    bw = max(1e-9, maxx - minx)
    bh = max(1e-9, maxy - miny)

    scale = min((W - 2 * margin) / bw, (H - 2 * margin) / bh) * 0.95

    cx = 0.5 * (minx + maxx)
    cy = 0.5 * (miny + maxy)

    projected = []
    for p0r, p1r, r in segs:
        x0 = W * 0.5 + (p0r[0] - cx) * scale
        x1 = W * 0.5 + (p1r[0] - cx) * scale
        y0 = H * 0.5 - (p0r[1] - cy) * scale
        y1 = H * 0.5 - (p1r[1] - cy) * scale
        th = max(1, int(round(r * params.line_width_scale * 0.65)))
        projected.append((th, x0, y0, x1, y1))

    # draw thick-to-thin so trunk sits underneath twigs
    projected.sort(key=lambda t: -t[0])
    for th, x0, y0, x1, y1 in projected:
        draw.line((x0, y0, x1, y1), fill=255, width=th)

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    img.save(out_png)
    mask = np.array(img) > 0
    return mask


# ----------------------------
# Silhouette metrics
# ----------------------------

def bbox_from_mask(mask: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return (0, 0, 0, 0)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return (x0, y0, x1, y1)


def silhouette_metrics(mask: np.ndarray) -> Dict[str, float]:
    H, W = mask.shape
    area = float(mask.sum()) / float(H * W)

    x0, y0, x1, y1 = bbox_from_mask(mask)
    bw = max(1, x1 - x0 + 1)
    bh = max(1, y1 - y0 + 1)
    aspect = float(bw) / float(bh)

    ys, xs = np.where(mask)
    if len(xs) == 0:
        return {
            "area_frac": 0.0,
            "bbox_w": 0.0,
            "bbox_h": 0.0,
            "aspect": 0.0,
            "cx_norm": 0.0,
            "cy_norm": 0.0,
            "elong": 0.0,
        }

    cx = float(xs.mean()) / float(W)
    cy = float(ys.mean()) / float(H)

    X = xs.astype(np.float64) - xs.mean()
    Y = ys.astype(np.float64) - ys.mean()
    cov_xx = float((X * X).mean())
    cov_yy = float((Y * Y).mean())
    cov_xy = float((X * Y).mean())
    tr = cov_xx + cov_yy
    det = cov_xx * cov_yy - cov_xy * cov_xy
    disc = max(0.0, tr * tr - 4.0 * det)
    l1 = 0.5 * (tr + math.sqrt(disc))
    l2 = 0.5 * (tr - math.sqrt(disc))
    elong = float(l1 / max(1e-9, l2))

    return {
        "area_frac": area,
        "bbox_w": float(bw) / float(W),
        "bbox_h": float(bh) / float(H),
        "aspect": aspect,
        "cx_norm": cx,
        "cy_norm": cy,
        "elong": elong,
    }


# ----------------------------
# Box-counting dimension
# ----------------------------

def box_count_dimension(mask: np.ndarray, eps_list: List[int]) -> Tuple[float, Dict[str, List[float]]]:
    H, W = mask.shape
    Ns = []
    inv_eps = []
    for eps in eps_list:
        h2 = int(math.ceil(H / eps) * eps)
        w2 = int(math.ceil(W / eps) * eps)
        pad = np.zeros((h2, w2), dtype=bool)
        pad[:H, :W] = mask
        blocks = pad.reshape(h2 // eps, eps, w2 // eps, eps)
        occ = blocks.any(axis=(1, 3))
        N = int(occ.sum())
        if N <= 0:
            continue
        Ns.append(N)
        inv_eps.append(1.0 / float(eps))

    x = np.log(inv_eps)
    y = np.log(Ns)
    if len(x) < 2:
        return 0.0, {"x": [], "y": []}

    slope = float(np.polyfit(x, y, 1)[0])
    return slope, {"x": x.tolist(), "y": y.tolist()}


# ----------------------------
# Geometry metrics: Strahler + pipe scaling
# ----------------------------

def build_children(edges) -> Dict[int, List[int]]:
    children: Dict[int, List[int]] = {}
    for e in edges:
        children.setdefault(e.a, []).append(e.b)
    return children


def compute_tips(children: Dict[int, List[int]]) -> Dict[int, int]:
    memo: Dict[int, int] = {}

    def dfs(u: int) -> int:
        if u in memo:
            return memo[u]
        ch = children.get(u, [])
        if not ch:
            memo[u] = 1
            return 1
        s = 0
        for v in ch:
            s += dfs(v)
        memo[u] = s
        return s

    nodes = set(children.keys())
    for vs in children.values():
        nodes.update(vs)
    for u in list(nodes):
        dfs(u)
    return memo


def strahler_orders(children: Dict[int, List[int]], root: int = 0) -> Dict[int, int]:
    memo: Dict[int, int] = {}

    def dfs(u: int) -> int:
        if u in memo:
            return memo[u]
        ch = children.get(u, [])
        if not ch:
            memo[u] = 1
            return 1
        orders = [dfs(v) for v in ch]
        m = max(orders)
        k = sum(1 for o in orders if o == m)
        memo[u] = m + 1 if k >= 2 else m
        return memo[u]

    dfs(root)
    return memo


def incoming_radius(edges) -> Dict[int, float]:
    acc: Dict[int, List[float]] = {}
    for e in edges:
        acc.setdefault(e.b, []).append(float(e.radius))

    out = {}
    for k, vals in acc.items():
        if len(vals) == 1 and vals[0] > 0 and math.isfinite(vals[0]):
            out[k] = float(vals[0])
    return out


def fit_pipe_scaling(tips: Dict[int, int], r_in: Dict[int, float]) -> Tuple[float, float]:
    xs, ys = [], []

    for n, nt in tips.items():
        r = r_in.get(n, None)
        if r is None or nt <= 0 or r <= 0:
            continue

        lx = math.log(float(nt))
        ly = math.log(float(r))
        if not (math.isfinite(lx) and math.isfinite(ly)):
            continue
        xs.append(lx)
        ys.append(ly)

    if len(xs) < 5:
        return 0.0, 0.0

    x = np.array(xs, dtype=np.float64)
    y = np.array(ys, dtype=np.float64)

    if np.allclose(x, x[0]):
        return 0.0, 0.0

    a, b = np.polyfit(x, y, 1)
    yhat = a * x + b
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(a), float(r2)


# ----------------------------
# Plot helpers
# ----------------------------

def plot_strahler_distribution(all_counts: List[Dict[int, int]], out_png: str):
    orders = sorted(set(k for c in all_counts for k in c.keys()))
    means, sds = [], []
    for o in orders:
        vals = [c.get(o, 0) for c in all_counts]
        means.append(np.mean(vals))
        sds.append(np.std(vals))

    plt.figure()
    plt.errorbar(orders, means, yerr=sds, marker="o")
    plt.xlabel("Strahler order")
    plt.ylabel("Node count")
    plt.title("Strahler order distribution (mean ± SD over seeds)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

    plt.figure()
    plt.plot(orders, np.log10(np.maximum(1e-9, means)), marker="o")
    plt.xlabel("Strahler order")
    plt.ylabel("log10(count)")
    plt.title("Strahler order distribution (log scale)")
    plt.tight_layout()
    plt.savefig(os.path.splitext(out_png)[0] + "_log.png", dpi=200)
    plt.close()


def plot_pipe_fit_hist(pipe_fits: List[Tuple[float, float]], out_dir: str):
    a = np.array([x for x, _ in pipe_fits], dtype=float)
    r2 = np.array([y for _, y in pipe_fits], dtype=float)

    plt.figure()
    plt.hist(a, bins=20)
    plt.xlabel("Pipe scaling exponent α (log r vs log tips)")
    plt.ylabel("Frequency")
    plt.title("Pipe-model scaling exponent across seeds")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pipe_alpha_hist.png"), dpi=200)
    plt.close()

    plt.figure()
    plt.hist(r2, bins=20)
    plt.xlabel("R² of pipe scaling fit")
    plt.ylabel("Frequency")
    plt.title("Pipe-model fit quality across seeds")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pipe_r2_hist.png"), dpi=200)
    plt.close()


def plot_rep_pipe_scatter(tips: Dict[int, int], r_in: Dict[int, float], out_png: str, seed: int):
    xs, ys = [], []
    for n, nt in tips.items():
        r = r_in.get(n, None)
        if r is None or nt <= 0 or r <= 0:
            continue
        xs.append(math.log(nt))
        ys.append(math.log(r))

    if len(xs) < 5:
        return

    x = np.array(xs)
    y = np.array(ys)
    a, b = np.polyfit(x, y, 1)

    plt.figure()
    plt.scatter(x, y, s=12)
    xx = np.linspace(x.min(), x.max(), 200)
    plt.plot(xx, a * xx + b, linestyle="--")
    plt.xlabel("log(tips downstream)")
    plt.ylabel("log(radius)")
    plt.title(f"Pipe scaling (seed {seed}): α≈{a:.3f}")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed0", type=int, default=7)
    ap.add_argument("--out", type=str, default="out_tree_eval")
    args = ap.parse_args()

    N = int(args.n)
    seed0 = int(args.seed0)
    out_dir = args.out

    os.makedirs(out_dir, exist_ok=True)
    masks_dir = os.path.join(out_dir, "masks")
    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    views = [0.0, 45.0, 90.0]
    eps_list = [1, 2, 4, 8, 16, 32, 64, 128, 256]

    sil_rows = []
    tree_rows = []
    all_strahler_counts = []
    pipe_fits = []
    rep_box = None
    rep_tips = None
    rep_rin = None

    for i in range(N):
        seed = seed0 + i
        print(f"[{i+1}/{N}] seed={seed}")
        root, nodes, edges, params = build_tree(seed)

        children = build_children(edges)
        tips = compute_tips(children)
        orders = strahler_orders(children, root=0)
        r_in = incoming_radius(edges)
        a_pipe, r2_pipe = fit_pipe_scaling(tips, r_in)
        pipe_fits.append((a_pipe, r2_pipe))

        counts: Dict[int, int] = {}
        for _, o in orders.items():
            counts[o] = counts.get(o, 0) + 1
        all_strahler_counts.append(counts)

        tree_rows.append({
            "seed": seed,
            "pipe_alpha": a_pipe,
            "pipe_r2": r2_pipe,
            "max_strahler": max(counts.keys()) if counts else 0,
            "n_edges": len(edges),
            "n_nodes": len(nodes),
        })

        for v in views:
            png_path = os.path.join(masks_dir, f"seed_{seed:04d}_yaw_{int(v):03d}.png")
            mask = render_mask(nodes, edges, params, v, png_path)
            sm = silhouette_metrics(mask)
            D, pts = box_count_dimension(mask, eps_list)
            sil_rows.append({
                "seed": seed,
                "yaw_deg": v,
                "mask_png": os.path.relpath(png_path, out_dir),
                "D_box": D,
                **sm,
            })
            if seed == seed0 and int(v) == int(views[0]):
                rep_box = pts
                rep_tips = tips
                rep_rin = r_in

    sil_csv = os.path.join(out_dir, "silhouette_metrics.csv")
    tree_csv = os.path.join(out_dir, "tree_metrics.csv")

    if sil_rows:
        with open(sil_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(sil_rows[0].keys()))
            w.writeheader()
            w.writerows(sil_rows)

    if tree_rows:
        with open(tree_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(tree_rows[0].keys()))
            w.writeheader()
            w.writerows(tree_rows)

    if rep_box and rep_box["x"]:
        x = np.array(rep_box["x"])
        y = np.array(rep_box["y"])
        m, c = np.polyfit(x, y, 1)
        plt.figure()
        plt.plot(x, y, marker="o", linestyle="-")
        plt.plot(x, m * x + c, linestyle="--")
        plt.xlabel("log(1/ε)")
        plt.ylabel("log N(ε)")
        plt.title(f"Box-counting (representative): slope D ≈ {m:.3f}")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "box_counting_representative.png"), dpi=200)
        plt.close()

    plot_strahler_distribution(all_strahler_counts, os.path.join(plots_dir, "strahler_distribution.png"))
    plot_pipe_fit_hist(pipe_fits, plots_dir)
    if rep_tips is not None and rep_rin is not None:
        plot_rep_pipe_scatter(rep_tips, rep_rin, os.path.join(plots_dir, "pipe_scaling_scatter_rep.png"), seed0)

    print("Done.")
    print("Wrote:")
    print(" ", sil_csv)
    print(" ", tree_csv)
    print(" ", plots_dir)


if __name__ == "__main__":
    main()
