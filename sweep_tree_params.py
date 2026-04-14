"""
sweep_tree_params.py

Parameter sensitivity sweep for fractal_tree_model.py.

Purpose:
- Vary one tree parameter across a list of values
- Run multiple seeds per value
- Measure structural outputs
- Save raw + summary CSVs
- Save plots and representative example renders

Example:
  python sweep_tree_params.py --param phi_a_deg --values 18 24 30 36 42 --seeds 5 --seed0 7 --out out_tree_sweep_phi_a
"""

from __future__ import annotations

import os
import csv
import math
import argparse
import pathlib
import importlib.util
import sys
from types import SimpleNamespace
from typing import Dict, List, Tuple, Any

import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt


# Dynamic import of tree model

HERE = pathlib.Path(__file__).resolve().parent
TREE_PATH = HERE / "fractal_tree_model.py"

spec = importlib.util.spec_from_file_location("treemodel", TREE_PATH)
T = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules[spec.name] = T
spec.loader.exec_module(T)


# Helpers: build tree with parameter overrides

def build_tree_with_overrides(seed: int, overrides: Dict[str, Any]):
    """
    Build a tree using T.Params, overriding exactly one parameter (or more if desired).
    Converts output into a simple node/edge representation for rendering and metrics.
    """
    params = T.Params(seed=seed)

    for k, v in overrides.items():
        if not hasattr(params, k):
            raise AttributeError(f"Params has no attribute '{k}'")
        setattr(params, k, v)

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
            children=node.children,
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


# Rendering helpers (adapted from eval_tree_pipeline.py)

def yaw_rotate(points: np.ndarray, yaw_deg: float) -> np.ndarray:
    a = math.radians(float(yaw_deg))
    c, s = math.cos(a), math.sin(a)
    R = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=float)
    return points @ R.T


def render_mask(nodes, edges, params, yaw_deg: float, out_png: str,
                W: int = 1024, H: int = 1024, margin: int = 60) -> np.ndarray:
    """
    Orthographic silhouette projection after yaw rotation.
    Writes PNG and returns boolean mask.
    """
    img = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(img)

    if not edges:
        os.makedirs(os.path.dirname(out_png), exist_ok=True)
        img.save(out_png)
        return np.zeros((H, W), dtype=bool)

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

    projected.sort(key=lambda t: -t[0])
    for th, x0, y0, x1, y1 in projected:
        draw.line((x0, y0, x1, y1), fill=255, width=th)

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    img.save(out_png)
    return np.array(img) > 0


# Metrics

def bbox_from_mask(mask: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return (0, 0, 0, 0)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


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
        }

    cx = float(xs.mean()) / float(W)
    cy = float(ys.mean()) / float(H)

    return {
        "area_frac": area,
        "bbox_w": float(bw) / float(W),
        "bbox_h": float(bh) / float(H),
        "aspect": aspect,
        "cx_norm": cx,
        "cy_norm": cy,
    }


def box_count_fractal_dimension(mask: np.ndarray, box_sizes=None):
    m = mask.astype(bool)
    H, W = m.shape

    if box_sizes is None:
        max_pow = int(np.floor(np.log2(min(H, W))))
        box_sizes = [2 ** k for k in range(1, max_pow + 1)]

    Ns = []
    eps = []

    for s in box_sizes:
        h = (H // s) * s
        w = (W // s) * s
        if h == 0 or w == 0:
            continue

        cropped = m[:h, :w]
        blocks = cropped.reshape(h // s, s, w // s, s)
        occ = blocks.any(axis=(1, 3))
        N = np.count_nonzero(occ)

        if N > 0:
            Ns.append(N)
            eps.append(s)

    Ns = np.array(Ns, dtype=float)
    eps = np.array(eps, dtype=float)

    if len(Ns) < 2:
        return float("nan"), float("nan"), np.array([]), np.array([])

    x = np.log(1.0 / eps)
    y = np.log(Ns)

    if len(x) >= 6:
        x_fit = x[1:-1]
        y_fit = y[1:-1]
    else:
        x_fit = x
        y_fit = y

    A = np.vstack([x_fit, np.ones_like(x_fit)]).T
    D, c = np.linalg.lstsq(A, y_fit, rcond=None)[0]

    y_pred = D * x_fit + c
    ss_res = np.sum((y_fit - y_pred) ** 2)
    ss_tot = np.sum((y_fit - np.mean(y_fit)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    return float(D), float(r2), x, y


def geometry_children_from_edges(nodes, edges) -> Dict[int, List[int]]:
    ch = {n.id: [] for n in nodes}
    for e in edges:
        ch[e.a].append(e.b)
    return ch


def strahler_orders(nodes, edges) -> Dict[int, int]:
    children = geometry_children_from_edges(nodes, edges)
    order = {n.id: 1 for n in nodes}

    for nid in sorted(order.keys(), reverse=True):
        ch = children[nid]
        if not ch:
            order[nid] = 1
        else:
            vals = [order[c] for c in ch]
            m = max(vals)
            order[nid] = m + 1 if vals.count(m) >= 2 else m

    return order


def branch_lengths(edges) -> np.ndarray:
    vals = []
    for e in edges:
        vals.append(float(np.linalg.norm(np.asarray(e.end) - np.asarray(e.start))))
    return np.array(vals, dtype=float)


def leaf_counts_from_root(root) -> Tuple[Dict[int, int], Dict[int, int]]:
    """
    Returns:
      downstream_tip_count per traversal index
      depth map per traversal index
    """
    idx_map = {}
    rev = []

    def assign(node):
        i = len(rev)
        idx_map[id(node)] = i
        rev.append(node)
        for ch in node.children:
            assign(ch)

    assign(root)

    tip_counts = {}
    depths = {}

    def walk(node):
        i = idx_map[id(node)]
        depths[i] = int(node.depth)
        if not node.children:
            tip_counts[i] = 1
            return 1
        s = 0
        for ch in node.children:
            s += walk(ch)
        tip_counts[i] = s
        return s

    walk(root)
    return tip_counts, depths, rev


def pipe_model_fit(root):
    """
    Simple log-log fit:
    radius ~ a * (downstream tips)^b
    """
    tip_counts, depths, rev = leaf_counts_from_root(root)

    xs = []
    ys = []

    for i, node in enumerate(rev):
        r = getattr(node, "radius", None)
        n_tips = tip_counts[i]
        if r is None or r <= 0 or n_tips <= 0:
            continue
        xs.append(math.log(float(n_tips)))
        ys.append(math.log(float(r)))

    if len(xs) < 3:
        return {"pipe_slope": float("nan"), "pipe_r2": float("nan"), "n": 0}

    x = np.array(xs, dtype=float)
    y = np.array(ys, dtype=float)

    A = np.vstack([x, np.ones_like(x)]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    y_pred = m * x + c

    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    return {"pipe_slope": float(m), "pipe_r2": float(r2), "n": int(len(x))}


# Parameter parsing

INT_PARAMS = {"d_max", "seed"}
FLOAT_PARAMS = {
    "l0", "w0", "ra", "rb", "rwa", "rwb",
    "phi_a_deg", "phi_b_deg",
    "cont_pitch_a_deg", "cont_pitch_b_deg",
    "golden_angle_deg", "azimuth_jitter_deg",
    "p_survive", "jitter_pitch_deg", "jitter_cont_deg",
    "pipe_p", "w_min", "world_scale", "line_width_scale",
}

def cast_param_value(param: str, s: str):
    if param in INT_PARAMS:
        return int(float(s))
    if param in FLOAT_PARAMS:
        return float(s)
    raise ValueError(f"Parameter '{param}' not recognized for sweeping.")


# Plot helpers

def save_errorbar_plot(summary_rows, param, metric_mean, metric_std, title, ylabel, out_path):
    xs = np.array([r["value"] for r in summary_rows], dtype=float)
    ys = np.array([r[metric_mean] for r in summary_rows], dtype=float)
    es = np.array([r[metric_std] for r in summary_rows], dtype=float)

    plt.figure()
    plt.errorbar(xs, ys, yerr=es, marker="o", linestyle="-", capsize=4)
    plt.xlabel(param)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_example_grid(example_pngs: List[Tuple[str, str]], out_path: str):
    """
    example_pngs: list of (label, png_path), one per parameter value
    """
    imgs = []
    labels = []
    for label, path in example_pngs:
        if os.path.exists(path):
            imgs.append(Image.open(path).convert("L"))
            labels.append(label)

    if not imgs:
        return

    n = len(imgs)
    W, H = imgs[0].size
    pad = 20
    label_h = 40

    canvas = Image.new("L", (n * (W + pad) + pad, H + label_h + 2 * pad), 255)

    for i, img in enumerate(imgs):
        x = pad + i * (W + pad)
        y = pad + label_h
        canvas.paste(img, (x, y))

    plt.figure(figsize=(3 * n, 4))
    plt.imshow(np.array(canvas), cmap="gray")
    plt.axis("off")

    for i, label in enumerate(labels):
        x = pad + i * (W + pad) + W / 2
        y = pad + 12
        plt.text(x, y, label, ha="center", va="center", fontsize=11)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


# Main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--param", type=str, required=True,
                    help="Single Params field to sweep, e.g. phi_a_deg, ra, d_max")
    ap.add_argument("--values", nargs="+", required=True,
                    help="Values for the swept parameter")
    ap.add_argument("--seeds", type=int, default=5,
                    help="Number of seeds per parameter value")
    ap.add_argument("--seed0", type=int, default=7)
    ap.add_argument("--out", type=str, required=True)

    ap.add_argument("--yaw_deg", type=float, default=45.0,
                    help="Silhouette yaw used for sweep metrics/examples")
    ap.add_argument("--img_size", type=int, default=1024)

    args = ap.parse_args()

    param = args.param
    values = [cast_param_value(param, v) for v in args.values]

    out_dir = args.out
    plots_dir = os.path.join(out_dir, "plots")
    examples_dir = os.path.join(out_dir, "examples")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(examples_dir, exist_ok=True)

    raw_rows = []
    summary_rows = []
    example_pngs = []

    for val in values:
        print(f"\n=== {param} = {val} ===")
        value_rows = []

        rep_png = None

        for i in range(args.seeds):
            seed = args.seed0 + i
            overrides = {param: val}

            root, nodes, edges, params = build_tree_with_overrides(seed=seed, overrides=overrides)

            png_path = os.path.join(
                examples_dir,
                f"{param}_{str(val).replace('.', 'p')}_seed_{seed}.png"
            )
            mask = render_mask(
                nodes, edges, params,
                yaw_deg=args.yaw_deg,
                out_png=png_path,
                W=args.img_size,
                H=args.img_size,
            )

            sil = silhouette_metrics(mask)
            D, box_r2, _, _ = box_count_fractal_dimension(mask)
            orders = strahler_orders(nodes, edges)
            max_strahler = max(orders.values()) if orders else 0
            bl = branch_lengths(edges)
            pipe = pipe_model_fit(root)

            row = {
                "param": param,
                "value": val,
                "seed": seed,
                "yaw_deg": args.yaw_deg,
                "n_nodes": len(nodes),
                "n_edges": len(edges),
                "branch_len_mean": float(np.mean(bl)) if len(bl) else float("nan"),
                "branch_len_median": float(np.median(bl)) if len(bl) else float("nan"),
                "branch_len_max": float(np.max(bl)) if len(bl) else float("nan"),
                "D_box": D,
                "box_r2": box_r2,
                "max_strahler": int(max_strahler),
                "pipe_slope": pipe["pipe_slope"],
                "pipe_r2": pipe["pipe_r2"],
                **sil,
                "png": os.path.relpath(png_path, out_dir),
            }
            raw_rows.append(row)
            value_rows.append(row)

            if rep_png is None:
                rep_png = png_path

            print(f"  seed={seed}  D={D:.3f}  aspect={sil['aspect']:.3f}  maxS={max_strahler}")

        # summary for this parameter value
        def mean_of(key):
            arr = np.array([r[key] for r in value_rows], dtype=float)
            arr = arr[np.isfinite(arr)]
            return float(np.mean(arr)) if len(arr) else float("nan")

        def std_of(key):
            arr = np.array([r[key] for r in value_rows], dtype=float)
            arr = arr[np.isfinite(arr)]
            return float(np.std(arr, ddof=0)) if len(arr) else float("nan")

        summary = {
            "param": param,
            "value": val,
            "n_runs": len(value_rows),
            "mean_D_box": mean_of("D_box"),
            "std_D_box": std_of("D_box"),
            "mean_area_frac": mean_of("area_frac"),
            "std_area_frac": std_of("area_frac"),
            "mean_aspect": mean_of("aspect"),
            "std_aspect": std_of("aspect"),
            "mean_n_edges": mean_of("n_edges"),
            "std_n_edges": std_of("n_edges"),
            "mean_max_strahler": mean_of("max_strahler"),
            "std_max_strahler": std_of("max_strahler"),
            "mean_pipe_r2": mean_of("pipe_r2"),
            "std_pipe_r2": std_of("pipe_r2"),
        }
        summary_rows.append(summary)

        if rep_png is not None:
            example_pngs.append((f"{param}={val}", rep_png))

    # Write CSVs
    raw_csv = os.path.join(out_dir, "raw_runs.csv")
    with open(raw_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(raw_rows[0].keys()))
        writer.writeheader()
        writer.writerows(raw_rows)

    summary_csv = os.path.join(out_dir, "summary.csv")
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    # Plots
    save_errorbar_plot(
        summary_rows, param,
        "mean_D_box", "std_D_box",
        title=f"Fractal dimension vs {param}",
        ylabel="Box-counting fractal dimension",
        out_path=os.path.join(plots_dir, f"D_vs_{param}.png"),
    )

    save_errorbar_plot(
        summary_rows, param,
        "mean_aspect", "std_aspect",
        title=f"Aspect ratio vs {param}",
        ylabel="Silhouette aspect ratio",
        out_path=os.path.join(plots_dir, f"aspect_vs_{param}.png"),
    )

    save_errorbar_plot(
        summary_rows, param,
        "mean_n_edges", "std_n_edges",
        title=f"Branch count vs {param}",
        ylabel="Edge count",
        out_path=os.path.join(plots_dir, f"edges_vs_{param}.png"),
    )

    save_example_grid(
        example_pngs,
        out_path=os.path.join(plots_dir, f"example_grid_{param}.png"),
    )

    print("\nDone.")
    print("Wrote:")
    print(" ", raw_csv)
    print(" ", summary_csv)
    print(" ", plots_dir)


if __name__ == "__main__":
    main()