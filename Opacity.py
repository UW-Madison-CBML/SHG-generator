# Multi-scale SHG fiber opacity model.
# All opacity tuning knobs and calculations live here:
#   - Regional: smooth image-scale intensity clusters (high/low groupings)
#   - Per-fiber: baseline brightness offsets between individual fibers
#   - Along-fiber: SHG coherent-interference pulsation
#   - Connectivity: gap opacity from L_conn (geometry breaks, not phase)

import numpy as np
from scipy.ndimage import gaussian_filter

# Adjust opacity behaviour
OPACITY_DEFAULTS = {
    # Global brightness anchor (per-fiber baseline is scaled to this range)
    "intensity_range": (0.35, 0.40),
    # Regional clustering: L_intensity controls spread; sigma sets cluster size
    "regional_sigma_base": 4.0,
    "regional_sigma_scale": 200.0,
    "regional_log_std": 0.55,
    # Per-fiber baseline jitter (independent of region, log-normal spread)
    "fiber_log_std": 0.22,
    # Along-fiber SHG phase interference
    "shg_floor": 0.30,
    "shg_phase_cycles": (20.5, 60.0),
    # Connectivity gap modulation (uses aux_L_conn per fiber)
    "gap_depth": 0.35,
    "gap_min": 0.06,
    "gap_cycles_base": 20.0,
    "gap_cycles_scale": 4.5,
    # Post-raster tone mapping
    "overlap_damp": 8.0,
    # Normalization (percentile-based rescale to norm_target)
    "normalize": True,
    "norm_percentile": 98.0,
    "norm_target": 0.75,
    # Gamma: true power-law correction (1.0 = linear, >1 darkens, <1 brightens)
    "gamma": 1.0,
    # Contrast: S-curve strength (0 = none, 1 = strong push of darks/lights apart)
    "contrast": 0.0,
}


def _normalize_field01(field):
    lo, hi = field.min(), field.max()
    if hi - lo < 1e-8:
        return np.full_like(field, 0.5)
    return (field - lo) / (hi - lo)


def _bilinear_sample(field, row, col):
    h, w = field.shape
    if row < 0 or col < 0 or row >= h - 1 or col >= w - 1:
        r = int(np.clip(round(row), 0, h - 1))
        c = int(np.clip(round(col), 0, w - 1))
        return float(field[r, c])

    r0, c0 = int(row), int(col)
    dr, dc = row - r0, col - c0
    return float(
        (1.0 - dr) * (1.0 - dc) * field[r0, c0]
        + (1.0 - dr) * dc * field[r0, c0 + 1]
        + dr * (1.0 - dc) * field[r0 + 1, c0]
        + dr * dc * field[r0 + 1, c0 + 1]
    )


def sample_field_at_points(field, points):
    # Bilinear sample scalar field at (row, col) points, shape (N, 2).
    points = np.asarray(points, dtype=np.float64)
    n = points.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i, (row, col) in enumerate(points):
        out[i] = _bilinear_sample(field, float(row), float(col))
    return out


def make_regional_intensity_field(shape, L_intensity, rng, cfg=None):
    # Smooth regional intensity field, mean-normalized to 1.0.

    # L_intensity=0 -> flat (no regional grouping).
    # L_intensity->1 -> strong high/low intensity regions.
    cfg = {**OPACITY_DEFAULTS, **(cfg or {})}
    L_intensity = float(np.clip(L_intensity, 0.0, 1.0))

    if L_intensity <= 0.0:
        return np.ones(shape, dtype=np.float64)

    sigma = cfg["regional_sigma_base"] + cfg["regional_sigma_scale"] * L_intensity
    z = gaussian_filter(rng.normal(size=shape), sigma)
    z = (z - z.mean()) / (z.std() + 1e-8)
    log_std = cfg["regional_log_std"] * L_intensity
    field = np.exp(log_std * z)
    field /= field.mean()
    return field


class FiberOpacityTable:
    # Per-fiber opacity parameters for rasterization.

    __slots__ = ("fiber_base", "phase_offset", "n_phase", "regional_field", "regional_mean")

    def __init__(self, fiber_base, phase_offset, n_phase, regional_field):
        self.fiber_base = np.asarray(fiber_base, dtype=np.float64)
        self.phase_offset = np.asarray(phase_offset, dtype=np.float64)
        self.n_phase = np.asarray(n_phase, dtype=np.float64)
        self.regional_field = np.asarray(regional_field, dtype=np.float64)
        self.regional_mean = float(self.regional_field.mean())


def build_fiber_opacity_table(seeds, regional_field, rng, cfg=None):
    # Build per-fiber opacity table from seed locations and regional field.

    # fiber_base combines:
    #   - global intensity anchor (intensity_range midpoint)
    #   - per-fiber log-normal jitter (some fibers dimmer/brighter on average)
    # Regional modulation is applied per-stamp via sample_regional().
    cfg = {**OPACITY_DEFAULTS, **(cfg or {})}
    seeds = np.asarray(seeds, dtype=np.float64)
    n = seeds.shape[0]
    if n == 0:
        return FiberOpacityTable(
            np.zeros(0), np.zeros(0), np.zeros(0), regional_field,
        )

    lo, hi = cfg["intensity_range"]
    global_anchor = 0.5 * (lo + hi)

    fiber_log_std = cfg["fiber_log_std"]
    fiber_base = global_anchor * np.exp(fiber_log_std * rng.normal(size=n))
    fiber_base = np.clip(fiber_base, lo * 0.5, hi * 1.4)

    phase_lo, phase_hi = cfg["shg_phase_cycles"]
    phase_offset = rng.uniform(0.0, 2.0 * np.pi, size=n)
    n_phase = rng.uniform(phase_lo, phase_hi, size=n)

    return FiberOpacityTable(fiber_base, phase_offset, n_phase, regional_field)


def sample_regional(table, row, col):
    # Regional multiplier at image coordinates (mean-normalized).
    val = _bilinear_sample(table.regional_field, row, col)
    return val / (table.regional_mean + 1e-8)


def shg_along_fiber(table, fiber_i, s_frac, cfg=None):
    # Along-fiber SHG coherent-interference factor in [shg_floor, 1].
    cfg = {**OPACITY_DEFAULTS, **(cfg or {})}
    shg_floor = float(np.clip(cfg["shg_floor"], 0.0, 0.9))
    phase_offset = table.phase_offset[fiber_i]
    n_phase = table.n_phase[fiber_i]
    wave = 0.5 + 0.5 * np.cos(2.0 * np.pi * n_phase * s_frac + phase_offset)
    return shg_floor + (1.0 - shg_floor) * (wave ** 2)


def connectivity_gap(s_frac, aux_L_conn, cfg=None):
    # Connectivity gap opacity in [gap_min, 1]; separate from SHG phase.
    cfg = {**OPACITY_DEFAULTS, **(cfg or {})}
    cn = float(np.clip(aux_L_conn, 0.0, 1.0))
    n_slow = cfg["gap_cycles_base"] + cfg["gap_cycles_scale"] * cn
    conn_phase = 0.5 + 0.5 * np.cos(2.0 * np.pi * n_slow * s_frac)
    gap = 1.0 - cn * cfg["gap_depth"] * (1.0 - conn_phase ** 2.2)
    return float(np.clip(gap, cfg["gap_min"], 1.0))


def stamp_opacity(table, fiber_i, s_frac, row, col, aux_L_conn, cfg=None):
    # Combined opacity multiplier at one stamp position.

    # Multi-scale product:
    #   fiber_base[i]  — per-fiber average brightness
    #   regional       — image-region grouping at this location
    #   shg_along      — along-fiber phase pulsation
    #   gap            — connectivity breaks
    regional = sample_regional(table, row, col)
    shg = shg_along_fiber(table, fiber_i, s_frac, cfg)
    gap = connectivity_gap(s_frac, aux_L_conn, cfg)
    return table.fiber_base[fiber_i] * regional * shg * gap


def _apply_contrast(img, contrast):
    # S-curve contrast adjustment on [0, 1] data.
    # contrast=0  → identity
    # contrast=1  → strong S-curve (darks pushed toward 0, lights toward 1)
    # Uses a cubic Hermite S-curve blended with the linear identity.
    contrast = float(np.clip(contrast, 0.0, 1.0))
    if contrast < 1e-6:
        return img
    # Cubic S-curve: 3t^2 - 2t^3  (maps 0->0, 0.5->0.5, 1->1, slopes softened at ends)
    s_curve = img * img * (3.0 - 2.0 * img)
    return (1.0 - contrast) * img + contrast * s_curve


def apply_tone_map(img, cfg=None):
    
    # Post-raster tone mapping.
    # Steps (each optional / independently tunable):
    #   1. Normalize: percentile-based rescale so the bright peak lands at norm_target.
    #      Skipped when cfg["normalize"] is False — useful for absolute intensity output.
    #   2. Gamma: standard power-law (gamma=1 -> linear; >1 darkens; <1 brightens).
    #   3. Contrast: S-curve that pushes darks darker and lights lighter without
    #      clipping, independently of overall brightness.
    cfg = {**OPACITY_DEFAULTS, **(cfg or {})}

    img = np.clip(img, 0.0, None).astype(np.float64)

    if cfg["normalize"]:
        p = np.percentile(img, cfg["norm_percentile"])
        if p > 0:
            img = img / p * cfg["norm_target"]
        img = np.clip(img, 0.0, 1.0)
    else:
        # Still clip to [0, 1] so gamma/contrast are well-defined
        img = np.clip(img, 0.0, 1.0)

    gamma = float(cfg["gamma"])
    if abs(gamma - 1.0) > 1e-6 and gamma > 0:
        img = np.power(img, gamma)

    img = _apply_contrast(img, cfg["contrast"])

    return img.astype(np.float32)
