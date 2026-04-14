import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


# Basic vector utilities

Vec3 = np.ndarray


def normalize(v: Vec3) -> Vec3:
    n = np.linalg.norm(v)
    if n < 1e-12:
        return v.copy()
    return v / n


def rotate_axis(v: Vec3, axis: Vec3, angle_rad: float) -> Vec3:
    """
    Rodrigues rotation formula.
    """
    axis = normalize(axis)
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return v * c + np.cross(axis, v) * s + axis * np.dot(axis, v) * (1.0 - c)


# Frame operations
# H = heading
# L = left
# U = up

def yaw_frame(H: Vec3, L: Vec3, U: Vec3, angle_rad: float) -> Tuple[Vec3, Vec3, Vec3]:
    """
    Rotate around local U axis.
    """
    H2 = rotate_axis(H, U, angle_rad)
    L2 = rotate_axis(L, U, angle_rad)
    return normalize(H2), normalize(L2), normalize(U)


def pitch_frame(H: Vec3, L: Vec3, U: Vec3, angle_rad: float) -> Tuple[Vec3, Vec3, Vec3]:
    """
    Rotate around local L axis.
    Positive angle lifts heading.
    """
    H2 = rotate_axis(H, L, angle_rad)
    U2 = rotate_axis(U, L, angle_rad)
    return normalize(H2), normalize(L), normalize(U2)


def roll_frame(H: Vec3, L: Vec3, U: Vec3, angle_rad: float) -> Tuple[Vec3, Vec3, Vec3]:
    """
    Rotate around local H axis.
    This is the key addition that makes branch placement genuinely 3D.
    """
    L2 = rotate_axis(L, H, angle_rad)
    U2 = rotate_axis(U, H, angle_rad)
    return normalize(H), normalize(L2), normalize(U2)


# Parameters

@dataclass
class Params:
    # Initial module sizes
    l0: float = 1.0
    w0: float = 1.0
    d_max: int = 9

    # Length scaling
    ra: float = 0.84   # apical continuation shrink
    rb: float = 0.76   # lateral branch shrink

    # Provisional width scaling
    rwa: float = 0.88
    rwb: float = 0.74

    # Branch pitch angles
    phi_a_deg: float = 30.0   # trunk - lateral pitch away from axis
    phi_b_deg: float = 22.0   # branch - sub-branch pitch

    # Small yaw bending for continuation
    cont_pitch_a_deg: float = 4.0
    cont_pitch_b_deg: float = 2.0

    # Azimuthal branch placement
    golden_angle_deg: float = 137.50776405
    azimuth_jitter_deg: float = 10.0

    # Small stochasticity
    p_survive: float = 0.78
    jitter_pitch_deg: float = 4.0
    jitter_cont_deg: float = 2.0

    # Pipe model
    pipe_p: float = 2.0
    w_min: float = 0.012

    # Rendering
    world_scale: float = 180.0
    line_width_scale: float = 20.0

    # Seed
    seed: int = 7


# Node

@dataclass
class Node:
    kind: str                     # "A" or "B"
    length: float
    width: float                  # provisional width parameter
    depth: int

    pos: Vec3
    H: Vec3
    L: Vec3
    U: Vec3

    azimuth_phase_deg: float      # phase around parent heading axis

    end: Optional[Vec3] = None
    children: List["Node"] = field(default_factory=list)
    radius: Optional[float] = None


# Random helpers

def jitter_deg(rng: np.random.Generator, mag_deg: float) -> float:
    return rng.uniform(-mag_deg, mag_deg)


def maybe_spawn(prob: float, rng: np.random.Generator) -> bool:
    return rng.random() < prob


# Child frame creation

def make_radial_child_frame(
    H: Vec3,
    L: Vec3,
    U: Vec3,
    azimuth_deg: float,
    pitch_deg: float,
    rng: np.random.Generator,
    params: Params
) -> Tuple[Vec3, Vec3, Vec3]:
    """
    True 3D branch placement:
    1) roll around parent heading by azimuth
    2) pitch away from the axis
    """
    az = math.radians(azimuth_deg + jitter_deg(rng, params.azimuth_jitter_deg))
    ph = math.radians(pitch_deg + jitter_deg(rng, params.jitter_pitch_deg))

    H2, L2, U2 = roll_frame(H, L, U, az)
    H2, L2, U2 = pitch_frame(H2, L2, U2, ph)
    return H2, L2, U2


def make_continuation_frame(
    H: Vec3,
    L: Vec3,
    U: Vec3,
    pitch_deg: float,
    rng: np.random.Generator,
    params: Params
) -> Tuple[Vec3, Vec3, Vec3]:
    """
    Very mild continuation bias to stop everything going dead straight.
    """
    ph = math.radians(pitch_deg + jitter_deg(rng, params.jitter_cont_deg))
    H2, L2, U2 = pitch_frame(H, L, U, ph)
    return H2, L2, U2


# Expansion rules

def expand_A(node: Node, params: Params, rng: np.random.Generator) -> None:
    """
    Apical rule:
        A -> F [radial B] [radial B] A

    The important difference from the old planar version:
    the two B branches are placed at different azimuths around the parent axis.
    """
    node.end = node.pos + node.length * node.H

    if node.depth >= params.d_max:
        return

    d2 = node.depth + 1

    # Use a golden-angle progression around the axis.
    # Two candidate azimuths around the parent axis.
    base = node.azimuth_phase_deg
    az1 = base + params.golden_angle_deg
    az2 = az1 + params.golden_angle_deg

    # Left-ish / first lateral
    if maybe_spawn(params.p_survive, rng):
        H1, L1, U1 = make_radial_child_frame(
            node.H, node.L, node.U,
            azimuth_deg=az1,
            pitch_deg=params.phi_a_deg,
            rng=rng,
            params=params
        )
        b1 = Node(
            kind="B",
            length=node.length * params.rb,
            width=node.width * params.rwb,
            depth=d2,
            pos=node.end.copy(),
            H=H1, L=L1, U=U1,
            azimuth_phase_deg=az1
        )
        node.children.append(b1)
        expand_B(b1, params, rng)

    # Right-ish / second lateral
    if maybe_spawn(params.p_survive, rng):
        H2, L2, U2 = make_radial_child_frame(
            node.H, node.L, node.U,
            azimuth_deg=az2,
            pitch_deg=params.phi_a_deg,
            rng=rng,
            params=params
        )
        b2 = Node(
            kind="B",
            length=node.length * params.rb,
            width=node.width * params.rwb,
            depth=d2,
            pos=node.end.copy(),
            H=H2, L=L2, U=U2,
            azimuth_phase_deg=az2
        )
        node.children.append(b2)
        expand_B(b2, params, rng)

    # Continue apical axis
    H3, L3, U3 = make_continuation_frame(
        node.H, node.L, node.U,
        pitch_deg=params.cont_pitch_a_deg,
        rng=rng,
        params=params
    )
    a_next = Node(
        kind="A",
        length=node.length * params.ra,
        width=node.width * params.rwa,
        depth=d2,
        pos=node.end.copy(),
        H=H3, L=L3, U=U3,
        azimuth_phase_deg=base + 0.5 * params.golden_angle_deg
    )
    node.children.append(a_next)
    expand_A(a_next, params, rng)


def expand_B(node: Node, params: Params, rng: np.random.Generator) -> None:
    """
    Lateral rule:
        B -> F [radial B] [radial B] B
    """
    node.end = node.pos + node.length * node.H

    if node.depth >= params.d_max:
        return

    d2 = node.depth + 1

    base = node.azimuth_phase_deg
    az1 = base + params.golden_angle_deg
    az2 = az1 + params.golden_angle_deg

    # First sub-branch
    if maybe_spawn(params.p_survive, rng):
        H1, L1, U1 = make_radial_child_frame(
            node.H, node.L, node.U,
            azimuth_deg=az1,
            pitch_deg=params.phi_b_deg,
            rng=rng,
            params=params
        )
        b1 = Node(
            kind="B",
            length=node.length * params.rb,
            width=node.width * params.rwb,
            depth=d2,
            pos=node.end.copy(),
            H=H1, L=L1, U=U1,
            azimuth_phase_deg=az1
        )
        node.children.append(b1)
        expand_B(b1, params, rng)

    # Second sub-branch
    if maybe_spawn(params.p_survive, rng):
        H2, L2, U2 = make_radial_child_frame(
            node.H, node.L, node.U,
            azimuth_deg=az2,
            pitch_deg=params.phi_b_deg,
            rng=rng,
            params=params
        )
        b2 = Node(
            kind="B",
            length=node.length * params.rb,
            width=node.width * params.rwb,
            depth=d2,
            pos=node.end.copy(),
            H=H2, L=L2, U=U2,
            azimuth_phase_deg=az2
        )
        node.children.append(b2)
        expand_B(b2, params, rng)

    # Continue branch axis
    H3, L3, U3 = make_continuation_frame(
        node.H, node.L, node.U,
        pitch_deg=params.cont_pitch_b_deg,
        rng=rng,
        params=params
    )
    b_next = Node(
        kind="B",
        length=node.length * params.ra,
        width=node.width * params.rwa,
        depth=d2,
        pos=node.end.copy(),
        H=H3, L=L3, U=U3,
        azimuth_phase_deg=base + 0.5 * params.golden_angle_deg
    )
    node.children.append(b_next)
    expand_B(b_next, params, rng)


# Pipe scaling

def assign_pipe_radii(node: Node, pipe_p: float, w_min: float) -> float:
    if not node.children:
        node.radius = w_min
        return node.radius

    child_radii = [assign_pipe_radii(ch, pipe_p, w_min) for ch in node.children]
    node.radius = (sum(r ** pipe_p for r in child_radii)) ** (1.0 / pipe_p)
    return node.radius


# Segment collection

def collect_segments(node: Node, out=None):
    if out is None:
        out = []

    if node.end is not None and node.radius is not None:
        out.append({
            "kind": node.kind,
            "depth": node.depth,
            "start": node.pos.copy(),
            "end": node.end.copy(),
            "radius": node.radius,
            "length": node.length,
        })

    for ch in node.children:
        collect_segments(ch, out)

    return out


# Build tree

def build_tree(params: Params) -> Node:
    rng = np.random.default_rng(params.seed)

    # Start with a true 3D frame.
    H = np.array([0.0, 1.0, 0.0], dtype=float)
    L = np.array([-1.0, 0.0, 0.0], dtype=float)
    U = np.array([0.0, 0.0, 1.0], dtype=float)

    root = Node(
        kind="A",
        length=params.l0,
        width=params.w0,
        depth=0,
        pos=np.array([0.0, 0.0, 0.0], dtype=float),
        H=H,
        L=L,
        U=U,
        azimuth_phase_deg=0.0,
    )

    expand_A(root, params, rng)
    assign_pipe_radii(root, params.pipe_p, params.w_min)
    return root


# Plotting

def set_equal_3d_axes(ax, pts: np.ndarray):
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.5 * np.max(maxs - mins) + 1.0

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def plot_tree_3d(segments, params: Params, elev=22, azim=-70):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection="3d")

    # Plot thick-to-thin for nicer visibility
    segments_sorted = sorted(segments, key=lambda s: s["radius"], reverse=True)

    for seg in segments_sorted:
        p0 = seg["start"] * params.world_scale
        p1 = seg["end"] * params.world_scale

        xs = [p0[0], p1[0]]
        ys = [p0[1], p1[1]]
        zs = [p0[2], p1[2]]

        lw = max(0.5, seg["radius"] * params.line_width_scale)
        ax.plot(xs, ys, zs, color="black", linewidth=lw)

    pts = np.array(
        [s["start"] for s in segments] + [s["end"] for s in segments],
        dtype=float
    ) * params.world_scale

    set_equal_3d_axes(ax, pts)
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    ax.set_title("Fractal Tree")
    plt.tight_layout()
    plt.show()


# Simple orthographic projection helper for quick checking

def plot_tree_front_view(segments, params: Params):
    fig, ax = plt.subplots(figsize=(8, 8))

    segments_sorted = sorted(segments, key=lambda s: s["radius"], reverse=True)

    for seg in segments_sorted:
        p0 = seg["start"] * params.world_scale
        p1 = seg["end"] * params.world_scale

        # Front view: x vs y
        xs = [p0[0], p1[0]]
        ys = [p0[1], p1[1]]

        lw = max(0.5, seg["radius"] * params.line_width_scale)
        ax.plot(xs, ys, color="black", linewidth=lw)

    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Front View")
    plt.tight_layout()
    plt.show()


# Main

def main():
    params = Params(
        l0=1.0,
        w0=1.0,
        d_max=8,
        ra=0.84,
        rb=0.76,
        rwa=0.88,
        rwb=0.74,
        phi_a_deg=28.0,
        phi_b_deg=22.0,
        cont_pitch_a_deg=4.0,
        cont_pitch_b_deg=2.0,
        golden_angle_deg=137.50776405,
        azimuth_jitter_deg=10.0,
        p_survive=0.78,
        jitter_pitch_deg=4.0,
        jitter_cont_deg=2.0,
        pipe_p=2.0,
        w_min=0.012,
        world_scale=180.0,
        line_width_scale=20.0,
        seed=7,
    )

    root = build_tree(params)
    segments = collect_segments(root)

    print("Segments:", len(segments))
    print("Root radius:", round(root.radius, 6))

    plot_tree_3d(segments, params, elev=22, azim=-70)
    plot_tree_3d(segments, params, elev=18, azim=-20)
    plot_tree_front_view(segments, params)


if __name__ == "__main__":
    main()