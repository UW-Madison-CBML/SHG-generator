import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.interpolate import make_splprep

def sample_vector(vx, vy, pos):
    h, w = vx.shape
    y, x = pos
    if x < 0 or x >= w - 1 or y < 0 or y >= h - 1:
        return None

    x0, y0 = int(x), int(y)
    dx, dy = x - x0, y - y0

    def interp(arr):
        return (
            (1 - dx) * (1 - dy) * arr[y0, x0]
            + dx * (1 - dy) * arr[y0, x0 + 1]
            + (1 - dx) * dy * arr[y0 + 1, x0]
            + dx * dy * arr[y0 + 1, x0 + 1]
        )

    v = np.array([interp(vy), interp(vx)])
    n = np.linalg.norm(v)
    if n == 0:
        return None
    return v / n


def resolve_streamline_steps(spline_length=None, max_steps=None, default=300):
    """Max integration steps per direction from spline_length or max_steps."""
    if spline_length is not None:
        return max(1, int(spline_length))
    if max_steps is not None:
        return max(1, int(max_steps))
    return max(1, int(default))


def integrate_streamline(
    vx, vy, seed, step_size=1.0, max_steps=None, spline_length=None,
    direction=1, L_curve=0.5, rng=None,
):
    pts = []
    pos = np.array(seed, dtype=float)
    rng = np.random.default_rng() if rng is None else rng
    max_angle = 0.35 * L_curve
    n_steps = resolve_streamline_steps(spline_length=spline_length, max_steps=max_steps)

    for _ in range(n_steps):
        v1 = sample_vector(vx, vy, pos)
        if v1 is None:
            break
        mid = pos + 0.5 * step_size * direction * v1
        v2 = sample_vector(vx, vy, mid)
        if v2 is None:
            break

        if max_angle > 0:
            a = rng.normal(0.0, max_angle)
            ca, sa = np.cos(a), np.sin(a)
            vy2, vx2 = v2[0], v2[1]
            v2 = np.array([ca * vy2 - sa * vx2, sa * vy2 + ca * vx2])
            v2 = v2 / (np.linalg.norm(v2) + 1e-8)

        pos = pos + step_size * direction * v2
        pts.append(pos.copy())

    return np.array(pts)


def generate_fiber(
    vx, vy, seed, step_size=1.0, max_steps=None, spline_length=None,
    L_curve=0.5, rng=None,
):
    forward = integrate_streamline(
        vx, vy, seed, step_size, max_steps=max_steps, spline_length=spline_length,
        direction=1, L_curve=L_curve, rng=rng,
    )
    backward = integrate_streamline(
        vx, vy, seed, step_size, max_steps=max_steps, spline_length=spline_length,
        direction=-1, L_curve=L_curve, rng=rng,
    )

    pts = []
    if backward is not None and len(backward) > 0:
        pts.append(backward[::-1])
    pts.append(np.asarray(seed, dtype=float)[None, :])
    if forward is not None and len(forward) > 0:
        pts.append(forward)
    return np.vstack(pts)


def fit_spline(points, smoothing=2.0, num_samples=400, k=3):
    if len(points) < k + 2:
        return points
    x = points[:, 1]
    y = points[:, 0]
    try:
        spl, _ = make_splprep([x, y], s=smoothing, k=k)
        u_new = np.linspace(0, 1, num_samples)
        x_s, y_s = spl(u_new)
        return np.vstack([y_s, x_s]).T
    except Exception:
        return points

def per_spline_auxiliary_values(global_p, n, rng):
    p = float(np.clip(global_p, 0.0, 1.0))
    n = int(n)
    if n <= 0:
        return np.zeros(0, dtype=np.float64)
    if p <= 0.0:
        return np.zeros(n, dtype=np.float64)
    if p >= 1.0:
        return np.ones(n, dtype=np.float64)
    eps = 1e-9
    edge = 4.0 * p * (1.0 - p) + eps
    concentration = 0.35 / edge
    a = max(p * concentration, eps)
    b = max((1.0 - p) * concentration, eps)
    x = rng.beta(a, b, size=n)
    return np.clip(x.astype(np.float64), 0.0, 1.0)


def sample_seeds_from_density(D, spline_num, L_density, rng):
    D = np.asarray(D, dtype=np.float64)
    H, W = D.shape
    n = int(max(0, spline_num))
    if n == 0:
        return np.zeros((0, 2), dtype=np.int64)

    D_clipped = np.clip(D, 0.0, None)
    if np.all(D_clipped == 0):
        probs = np.full(D_clipped.size, 1.0 / D_clipped.size, dtype=np.float64)
    else:
        dens_power = 0.8 + 2.2 * float(L_density)
        probs = np.power(D_clipped.ravel(), dens_power)
        probs = probs / (probs.sum() + 1e-12)

    expected = n * probs
    counts = np.floor(expected).astype(np.int64)
    missing = int(n - counts.sum())

    if missing > 0:
        frac = expected - counts
        frac_sum = frac.sum()
        if frac_sum <= 0:
            pick = rng.choice(counts.size, size=missing, replace=False)
        else:
            pick_probs = frac / frac_sum
            pick = rng.choice(counts.size, size=missing, replace=False, p=pick_probs)
        counts[pick] += 1

    idx = np.repeat(np.arange(counts.size, dtype=np.int64), counts)
    if idx.size == 0:
        return np.zeros((0, 2), dtype=np.int64)

    rng.shuffle(idx)
    seeds = np.column_stack([idx // W, idx % W]).astype(np.int64)
    return seeds

