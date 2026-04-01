import math
import os
from collections import deque

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw


# Box counting
def box_count_fractal_dimension(mask: np.ndarray, box_sizes=None):
    """
    mask: 2D boolean / 0-1 array
    returns D, r2, x, y
    """
    m = mask.astype(bool)
    H, W = m.shape

    if box_sizes is None:
        max_pow = int(np.floor(np.log2(min(H, W))))
        box_sizes = [2**k for k in range(1, max_pow + 1)]

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

    return D, r2, x, y


# Fast vectorised Laplace solver
def solve_laplace_jacobi(V, fixed, fixed_vals, iters=50):
    """
    Vectorised Jacobi relaxation.
    """
    V = V.copy()

    for _ in range(iters):
        Vn = V.copy()

        Vn[1:-1, 1:-1] = 0.25 * (
            V[:-2, 1:-1] + V[2:, 1:-1] + V[1:-1, :-2] + V[1:-1, 2:]
        )

        # simple side boundary handling
        Vn[:, 0] = Vn[:, 1]
        Vn[:, -1] = Vn[:, -2]
        Vn[0, :] = Vn[1, :]

        # re-apply fixed potentials
        Vn[fixed] = fixed_vals[fixed]
        V = Vn

    return V


# Lightning simulation
def grow_lightning_dbm_v2(
    seed=7,
    grid=384,
    max_steps=14000,
    eta=2.25,
    relax_iters=60,
    relax_every=2,
    downward_bias=0.65,
    retire_prob=0.00
):
    """
    DBM-like cloud->ground growth.
    Returns:
    channel mask
    parent dict mapping cell -> parent cell
    hit_ground
    root
    hit_cell
    """
    rng = np.random.default_rng(seed)
    H = W = grid

    channel = np.zeros((H, W), dtype=bool)

    # Potential field
    V = np.zeros((H, W), dtype=float)
    fixed = np.zeros((H, W), dtype=bool)
    fixed_vals = np.zeros((H, W), dtype=float)

    # Ground electrode
    fixed[H - 1, :] = True
    fixed_vals[H - 1, :] = 0.0
    V[H - 1, :] = 0.0

    # Root seed at cloud
    root = (0, W // 2)
    channel[root] = True
    fixed[root] = True
    fixed_vals[root] = 1.0
    V[root] = 1.0

    # parent map stores graph
    parent = {root: None}

    # active tips
    tips = {root}
    neigh4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def frontier_candidates_with_parents(tips_set):
        cand_parent = {}
        for tip in tips_set:
            y, x = tip
            for dy, dx in neigh4:
                yy, xx = y + dy, x + dx
                if 0 <= yy < H and 0 <= xx < W and not channel[yy, xx]:
                    cand_parent.setdefault((yy, xx), tip)
        return cand_parent

    hit_ground = False
    hit_cell = None

    for t in range(max_steps):
        if t % 200 == 0:
            print("growth step:", t)

        if t % relax_every == 0:
            V = solve_laplace_jacobi(V, fixed, fixed_vals, iters=relax_iters)

        cand_parent = frontier_candidates_with_parents(tips)
        if not cand_parent:
            break

        cands = list(cand_parent.keys())

        weights = []
        for (y, x) in cands:
            y0 = max(0, y - 1)
            y1 = min(H - 1, y + 1)
            x0 = max(0, x - 1)
            x1 = min(W - 1, x + 1)

            dVy = V[y1, x] - V[y0, x]
            dVx = V[y, x1] - V[y, x0]
            g = math.sqrt(dVx * dVx + dVy * dVy) + 1e-12

            down = 1.0 + downward_bias * (y / (H - 1))
            w = (g ** eta) * down
            weights.append(w)

        weights = np.array(weights, dtype=float)
        if (not np.isfinite(weights).all()) or weights.sum() <= 0:
            idx = rng.integers(0, len(cands))
        else:
            weights /= weights.sum()
            idx = rng.choice(len(cands), p=weights)

        new_cell = cands[idx]
        par = cand_parent[new_cell]

        channel[new_cell] = True
        parent[new_cell] = par

        fixed[new_cell] = True
        fixed_vals[new_cell] = 1.0
        V[new_cell] = 1.0

        tips.add(new_cell)

        if len(tips) > 1 and retire_prob > 0:
            dead = set()
            for tip in tips:
                if tip == new_cell:
                    continue
                if rng.random() < retire_prob:
                    dead.add(tip)
            tips -= dead

        cleaned = set()
        for tip in tips:
            y, x = tip
            alive = False
            for dy, dx in neigh4:
                yy, xx = y + dy, x + dx
                if 0 <= yy < H and 0 <= xx < W and not channel[yy, xx]:
                    alive = True
                    break
            if alive:
                cleaned.add(tip)
        tips = cleaned

        if new_cell[0] >= H - 2:
            hit_ground = True
            hit_cell = new_cell
            break

    return channel, parent, hit_ground, root, hit_cell


# Graph / order utilities
def build_children_map(parent):
    children = {}
    for node, par in parent.items():
        children.setdefault(node, [])
        if par is not None:
            children.setdefault(par, []).append(node)
    return children


def trunk_path_to_root(hit_cell, parent):
    """
    Return set of cells on main leader path from hit cell back to root.
    """
    trunk = set()
    cur = hit_cell
    while cur is not None:
        trunk.add(cur)
        cur = parent[cur]
    return trunk


def trunk_path_list(hit_cell, parent):
    """
    Ordered list from root to hit cell.
    """
    path = []
    cur = hit_cell
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path


def branch_depths_from_trunk(parent, trunk_set):
    """
    Approximate branch order:
    trunk edges = depth 0
    edges off trunk increase depth by BFS in child graph.
    """
    children = build_children_map(parent)
    depth_map = {}

    for n in trunk_set:
        depth_map[n] = 0

    q = deque()
    for n in trunk_set:
        for ch in children.get(n, []):
            if ch not in trunk_set:
                depth_map[ch] = 1
                q.append(ch)

    while q:
        u = q.popleft()
        for ch in children.get(u, []):
            if ch not in depth_map:
                depth_map[ch] = depth_map[u] + 1
                q.append(ch)

    return depth_map


# Lightning metric utilities
def build_undirected_graph_from_parent(parent):
    """
    Convert parent map to undirected adjacency graph.
    """
    graph = {}
    for node in parent:
        graph.setdefault(node, set())

    for node, par in parent.items():
        if par is not None:
            graph[node].add(par)
            graph.setdefault(par, set()).add(node)

    return graph


def classify_graph_nodes(graph):
    """
    degree 1 -> terminal
    degree 2 -> path
    degree >=3 -> junction
    """
    terminals = []
    junctions = []
    path_nodes = []

    for node, nbrs in graph.items():
        d = len(nbrs)
        if d == 1:
            terminals.append(node)
        elif d == 2:
            path_nodes.append(node)
        elif d >= 3:
            junctions.append(node)

    return terminals, junctions, path_nodes


def extract_branch_lengths(graph):
    """
    Collapse chains of degree-2 nodes into branch segments.
    Returns list of segment lengths in graph-edge units.
    """
    visited_edges = set()
    lengths = []

    for node in graph:
        deg = len(graph[node])

        # segment starts at terminals/junctions/root-like nodes
        if deg != 2:
            for nbr in graph[node]:
                edge = tuple(sorted((node, nbr)))
                if edge in visited_edges:
                    continue

                length = 1
                prev = node
                curr = nbr
                visited_edges.add(edge)

                while len(graph[curr]) == 2:
                    next_nodes = [n for n in graph[curr] if n != prev]
                    if not next_nodes:
                        break

                    nxt = next_nodes[0]
                    next_edge = tuple(sorted((curr, nxt)))
                    if next_edge in visited_edges:
                        break

                    visited_edges.add(next_edge)
                    prev = curr
                    curr = nxt
                    length += 1

                lengths.append(length)

    return lengths


def compute_main_channel_tortuosity(hit_cell, parent):
    """
    Main leader tortuosity:
    path length / straight-line distance
    using root->hit_cell trunk path.
    """
    path = trunk_path_list(hit_cell, parent)

    if len(path) < 2:
        return 1.0, path, 0.0, 0.0

    path_length = len(path) - 1

    y0, x0 = path[0]
    y1, x1 = path[-1]
    straight = math.sqrt((y1 - y0) ** 2 + (x1 - x0) ** 2)

    tortuosity = path_length / straight if straight > 0 else 1.0
    return tortuosity, path, path_length, straight


# Rendering

def cell_center_to_render_xy(cell, grid, render_size, margin=40):
    """
    Map grid cell center to render canvas coords.
    """
    y, x = cell
    usable = render_size - 2 * margin
    xx = margin + (x + 0.5) / grid * usable
    yy = margin + (y + 0.5) / grid * usable
    return xx, yy


def render_lightning_graph(
    parent,
    grid,
    render_size=2048,
    margin=40,
    trunk_set=None,
    branch_depths=None,
    trunk_width=8,
    branch_width=5,
    width_decay=0.82,
    save_pretty_path="lightning_pretty.png",
    save_mask_path="lightning_mask_hr.png"
):
    """
    Render lightning from parent-child graph to:
    - pretty grayscale image
    - binary mask image
    """
    img_pretty = Image.new("L", (render_size, render_size), 0)
    img_mask = Image.new("L", (render_size, render_size), 0)

    draw_pretty = ImageDraw.Draw(img_pretty)
    draw_mask = ImageDraw.Draw(img_mask)

    edges = []
    for child, par in parent.items():
        if par is not None:
            edges.append((par, child))

    def edge_key(edge):
        par, ch = edge
        if trunk_set is not None and par in trunk_set and ch in trunk_set:
            return 0
        return 1

    edges.sort(key=edge_key)

    for par, ch in edges:
        x0, y0 = cell_center_to_render_xy(par, grid, render_size, margin)
        x1, y1 = cell_center_to_render_xy(ch, grid, render_size, margin)

        if trunk_set is not None and par in trunk_set and ch in trunk_set:
            w = trunk_width
            gray = 255
        else:
            d = 1
            if branch_depths is not None:
                d = max(branch_depths.get(ch, 1), 1)
            w = max(1, int(round(branch_width * (width_decay ** (d - 1)))))
            gray = max(150, 255 - 18 * d)

        draw_pretty.line((x0, y0, x1, y1), fill=gray, width=w)
        draw_mask.line((x0, y0, x1, y1), fill=255, width=max(1, min(3, w)))

    img_pretty.save(save_pretty_path)
    img_mask.save(save_mask_path)

    pretty_np = np.array(img_pretty)
    mask_np = (np.array(img_mask) > 0).astype(np.uint8)

    return pretty_np, mask_np


def render_main_path_overlay(
    pretty_img,
    path,
    grid,
    render_size=2048,
    margin=40,
    overlay_width=10,
    save_path="lightning_main_channel_overlay.png"
):
    """
    Overlay main leader path on top of pretty render.
    """
    img = Image.fromarray(pretty_img).convert("RGB")
    draw = ImageDraw.Draw(img)

    if len(path) >= 2:
        pts = [cell_center_to_render_xy(cell, grid, render_size, margin) for cell in path]
        flat_pts = [(x, y) for (x, y) in pts]
        draw.line(flat_pts, fill=(255, 80, 80), width=overlay_width)

    img.save(save_path)
    return np.array(img)


# Plot helpers

def show_image(img, title):
    plt.figure(figsize=(8, 10))
    plt.imshow(img, cmap="gray", origin="upper")
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def show_box_count(mask):
    D, r2, x, y = box_count_fractal_dimension(mask)

    plt.figure(figsize=(6.5, 4.5))
    plt.plot(x, y, marker="o")

    if len(x) >= 6:
        x_fit = x[1:-1]
        y_fit = y[1:-1]
    else:
        x_fit = x
        y_fit = y

    A = np.vstack([x_fit, np.ones_like(x_fit)]).T
    Dfit, cfit = np.linalg.lstsq(A, y_fit, rcond=None)[0]
    plt.plot(x_fit, Dfit * x_fit + cfit, linestyle="--")

    plt.xlabel("log(1/ε)")
    plt.ylabel("log N(ε)")
    plt.title(f"Box-counting: D ≈ {Dfit:.6f}, R² ≈ {r2:.6f}")
    plt.tight_layout()
    plt.show()

    print(f"Box-counting fractal dimension D ≈ {Dfit:.6f}, R^2 ≈ {r2:.6f}")


def show_branch_length_hist(branch_lengths):
    if len(branch_lengths) == 0:
        print("No branch lengths to plot.")
        return

    plt.figure(figsize=(6.5, 4.5))
    plt.hist(branch_lengths, bins=30)
    plt.xlabel("Branch segment length (grid edges)")
    plt.ylabel("Frequency")
    plt.title("Lightning branch length distribution")
    plt.tight_layout()
    plt.show()


# Main

def main():
    print("RUNNING:", os.path.abspath(__file__))

    # Simulation parameters
    seed = 7
    grid = 384
    max_steps = 14000
    eta = 2.25
    relax_iters = 60
    relax_every = 2
    downward_bias = 0.65
    retire_prob = 0.00

    # Render parameters
    render_size = 2048
    trunk_width = 8
    branch_width = 5
    width_decay = 0.82
    margin = 40

    channel, parent, hit_ground, root, hit_cell = grow_lightning_dbm_v2(
        seed=seed,
        grid=grid,
        max_steps=max_steps,
        eta=eta,
        relax_iters=relax_iters,
        relax_every=relax_every,
        downward_bias=downward_bias,
        retire_prob=retire_prob
    )

    print("Hit ground:", hit_ground)
    print("grid =", grid)
    print("occupied cells =", int(channel.sum()))
    print("graph edges =", len(parent) - 1)

    trunk_set = None
    branch_depths = None
    main_path = []
    tortuosity = None
    main_path_length = None
    main_straight_length = None

    if hit_ground and hit_cell is not None:
        trunk_set = trunk_path_to_root(hit_cell, parent)
        branch_depths = branch_depths_from_trunk(parent, trunk_set)
        tortuosity, main_path, main_path_length, main_straight_length = compute_main_channel_tortuosity(
            hit_cell, parent
        )

    pretty_img, mask_img = render_lightning_graph(
        parent,
        grid=grid,
        render_size=render_size,
        margin=margin,
        trunk_set=trunk_set,
        branch_depths=branch_depths,
        trunk_width=trunk_width,
        branch_width=branch_width,
        width_decay=width_decay,
        save_pretty_path="lightning_pretty.png",
        save_mask_path="lightning_mask_hr.png"
    )

    raw_mask = (channel.astype(np.uint8) * 255)
    Image.fromarray(raw_mask).save("lightning_raw_mask.png")

    # New structural metrics
    graph = build_undirected_graph_from_parent(parent)
    terminals, junctions, path_nodes = classify_graph_nodes(graph)
    branch_lengths = extract_branch_lengths(graph)

    print("\n=== Lightning structural metrics ===")
    print("Terminal count:", len(terminals))
    print("Junction count:", len(junctions))
    print("Path-node count:", len(path_nodes))

    if len(branch_lengths) > 0:
        print("Branch segment count:", len(branch_lengths))
        print("Mean branch length:", float(np.mean(branch_lengths)))
        print("Median branch length:", float(np.median(branch_lengths)))
        print("Max branch length:", int(np.max(branch_lengths)))
    else:
        print("Branch segment count: 0")

    if tortuosity is not None:
        print("Main-channel path length:", float(main_path_length))
        print("Main-channel straight distance:", float(main_straight_length))
        print("Main-channel tortuosity:", float(tortuosity))
    else:
        print("Main-channel tortuosity: unavailable (did not hit ground)")

    # Save overlay of main leader
    if len(main_path) >= 2:
        overlay_img = render_main_path_overlay(
            pretty_img,
            main_path,
            grid=grid,
            render_size=render_size,
            margin=margin,
            overlay_width=10,
            save_path="lightning_main_channel_overlay.png"
        )
        show_image(overlay_img, "Lightning with main leader highlighted")

    # Show outputs
    show_image(pretty_img, "Lightning v2 (rendered graph)")
    show_image(mask_img, "Lightning v2 (binary mask for measurement)")
    show_box_count(mask_img)
    show_branch_length_hist(branch_lengths)


if __name__ == "__main__":
    main()