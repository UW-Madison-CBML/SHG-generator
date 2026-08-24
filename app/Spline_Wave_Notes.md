
# Spline Waviness Notes

I added an extra step between the `generate_fiber` function and the `fit_splines` function. This method is called `sinusoidal_fiber_offset` and it adds a sinusoidal offset to each point in the spline along its normal with a random phase offset deterimined by the rng function. Overall I tried my best to exactly copy the parameters in the old `rasterize_splines` method. Becuase the thickness parameter is used to generate the `base_wavelength` value and thickness is not part of spline generation I substituted the default value of 3.0.

Here is the change to the pipeline
```py
splines = []
for i, seed_pt in enumerate(seeds):
    raw_fiber = generate_fiber(
        Qx, Qy, seed_pt,
        step_size=1.0,
        spline_length=run_params["spline_length"],
        L_curve=run_params["L_curve"],
        susceptibility=1.0 - aux_curve[i],
        rng=rng
    )
# ---------- new offset step -----------------
offset_fiber = sinusoidal_fiber_offset(raw_fiber, wave_amp=aux_curve
# --------------------------------------------
splines.append(fit_spline(offset_fiber, num_samples=max(50, run_params["spline_length"] * 2)))

```