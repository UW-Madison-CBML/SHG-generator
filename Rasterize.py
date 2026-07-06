import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.interpolate import make_splprep

import Opacity as opacity


def rasterize_splines(
    H,
    W,
    splines,
    thickness=3.0,
    oversample=4.0,
    # tone mapping
    normalize=True,        # False → skip percentile rescale (raw accumulator output)
    norm_percentile=98.0,  # bright-peak percentile used when normalize=True
    norm_target=0.75,      # where that percentile lands after rescale
    gamma=1.0,             # power-law: 1=linear, >1 darkens, <1 brightens
    contrast=0.0,          # S-curve: 0=none, 1=strong darks-darker/lights-lighter
    # fiber geometry 
    out_H=None,
    out_W=None,
    intensity_seed=0,
    L_conn=0.3,
    aux_L_curve=None,
    aux_L_conn=None,
    wave_factor=1,
    # opacity model
    opacity_table=None,
    regional_field=None,
    seeds=None,
    L_intensity=0.5,
    overlap_damp=None,     # if None, uses OPACITY_DEFAULTS value
    # advanced / full override
    opacity_cfg=None,      # dict merged on top of OPACITY_DEFAULTS (lowest priority)
):
    """
    Rasterize a list of polyline splines into a grayscale image.

    Tone-mapping pipeline (applied after accumulation):
      normalize=True  → percentile rescale to norm_target, then gamma + contrast
      normalize=False → raw accumulator values clipped to [0,1], then gamma + contrast

    Parameters
    ----------
    gamma : float
        Standard power-law correction applied after normalization.
        gamma=1.0 is linear; gamma>1 darkens the image; gamma<1 brightens it.
    contrast : float in [0, 1]
        S-curve strength.  0 = no change.  1 = strong separation of darks and
        lights without clipping (mid-tones are unaffected at 0.5).
    normalize : bool
        When False the percentile rescale is skipped, so absolute accumulator
        levels are preserved (useful when compositing or checking raw intensity).
    """
    if out_H is None:
        out_H = H
    if out_W is None:
        out_W = W

    # Build the effective cfg: start from defaults, apply opacity_cfg overrides,
    # then apply explicit scalar arguments (highest priority).
    cfg = {**opacity.OPACITY_DEFAULTS, **(opacity_cfg or {})}
    cfg["normalize"] = normalize
    cfg["norm_percentile"] = norm_percentile
    cfg["norm_target"] = norm_target
    cfg["gamma"] = gamma
    cfg["contrast"] = contrast
    if overlap_damp is not None:
        cfg["overlap_damp"] = overlap_damp

    scale_y = out_H / max(H, 1)
    scale_x = out_W / max(W, 1)
    img = np.zeros((out_H, out_W), dtype=np.float32)

    rng = np.random.default_rng(intensity_seed)
    n_sp = len(splines)

    if aux_L_curve is None:
        aux_L_curve = np.zeros(n_sp, dtype=np.float64)
    else:
        aux_L_curve = np.asarray(aux_L_curve, dtype=np.float64).ravel()
        if aux_L_curve.size < n_sp:
            aux_L_curve = np.pad(aux_L_curve, (0, n_sp - aux_L_curve.size))
        aux_L_curve = aux_L_curve[:n_sp]

    if aux_L_conn is None:
        aux_L_conn = np.full(n_sp, float(L_conn), dtype=np.float64)
    else:
        aux_L_conn = np.asarray(aux_L_conn, dtype=np.float64).ravel()
        if aux_L_conn.size < n_sp:
            aux_L_conn = np.pad(aux_L_conn, (0, n_sp - aux_L_conn.size))
        aux_L_conn = aux_L_conn[:n_sp]

    if opacity_table is None:
        if regional_field is None:
            regional_field = opacity.make_regional_intensity_field(
                (H, W), L_intensity, rng, cfg,
            )
        if seeds is None:
            seeds = np.zeros((n_sp, 2), dtype=np.float64)
        opacity_table = opacity.build_fiber_opacity_table(seeds, regional_field, rng, cfg)

    stamp_damp = 1.0 / max(cfg["overlap_damp"], 1.0)

    for i, spline in enumerate(splines):
        pts = np.asarray(spline)
        if pts.ndim != 2 or pts.shape[1] != 2 or pts.size == 0:
            continue

        cv = float(np.clip(aux_L_curve[i], 0.0, 1.0))
        cn = float(aux_L_conn[i])

        pts_out = np.empty_like(pts, dtype=np.float64)
        pts_out[:, 0] = pts[:, 0] * scale_y
        pts_out[:, 1] = pts[:, 1] * scale_x

        seg = np.diff(pts_out, axis=0)
        seg_len = np.sqrt((seg**2).sum(axis=1))
        total_len = float(seg_len.sum())
        if total_len < 1e-6:
            continue
        s_vert = np.concatenate([[0.0], np.cumsum(seg_len)])

        wave_len = max(0.3, float(thickness) * 0.6)
        n_cycles = (float(np.clip(total_len / wave_len, 1.8, 9.0)) / float(wave_factor))
        wave_amp = wave_factor * 2.8 * cv

        for seg_idx, ((y0, x0), (y1, x1)) in enumerate(zip(pts_out[:-1], pts_out[1:])):
            dy = y1 - y0
            dx = x1 - x0
            sl = float(np.hypot(dy, dx))
            if sl < 1e-9:
                continue
            ty, tx = dy / sl, dx / sl
            ny_n, nx_n = -tx, ty

            s0 = s_vert[seg_idx]
            num = max(2, int(sl * oversample))
            for t in np.linspace(0.0, 1.0, num=num):
                s = s0 + t * sl
                s_frac = s / total_len
                y = y0 + t * dy
                x = x0 + t * dx
                if wave_amp > 0.0:
                    wobble = wave_amp * np.sin(2.0 * np.pi * n_cycles * s_frac)
                    y += wobble * ny_n
                    x += wobble * nx_n

                row_f = y / scale_y
                col_f = x / scale_x
                op = opacity.stamp_opacity(
                    opacity_table, i, s_frac, row_f, col_f, cn, cfg,
                )
                stamp = op * stamp_damp

                iy, ix = int(round(y)), int(round(x))
                if 0 <= iy < out_H and 0 <= ix < out_W:
                    r = int(np.ceil(thickness))
                    for ddy in range(-r, r + 1):
                        for ddx in range(-r, r + 1):
                            jy, jx = iy + ddy, ix + ddx
                            if 0 <= jy < out_H and 0 <= jx < out_W:
                                d2 = ddy * ddy + ddx * ddx
                                if d2 <= thickness * thickness:
                                    tt = d2 / (thickness * thickness + 1e-8)
                                    wgt = np.exp(-6.0 * tt * tt)
                                    img[jy, jx] += stamp * wgt

    return opacity.apply_tone_map(img, cfg)
