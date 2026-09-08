"""Fit a ratio-preserving tone curve LUT from a measured source/target pair.

Why this exists: the app's contrast slider clips at zero rather than compressing
into it. Measured on this scene, contrast 0.12 still pinned 25.8% of shadow reds
to zero while landing *shallower* blacks than the reference, which reached the
same depth with 0.1% pinned. No value of that slider gets there cleanly.

Two things make this curve behave where the slider does not:

1. It is fitted to measured luminance percentiles, so it reproduces a specific
   reference rather than applying a generic shape.
2. It is applied as a *ratio-preserving* scale - out = rgb * f(L)/L - instead of
   per channel. Channel ratios survive, so the smallest channel (red, in these
   blue shadows) is never crushed independently of the others. That also means
   saturation stays put instead of drifting up the way it does under contrast.

Monotone cubic (Fritsch-Carlson) keeps the curve from overshooting between
control points, which would band in the shadows.

Run:  python make_tone_lut.py
"""

import numpy as np

SIZE = 33
OUT = "tone-target-match.cube"

# Luminance percentiles measured with PIL/numpy over the full frame.
# source = the ungraded neural result, target = the reference grade.
PERCENTILES = [1, 5, 50, 95, 99]
SOURCE = [0.054, 0.070, 0.152, 0.350, 0.416]
TARGET = [0.031, 0.046, 0.139, 0.354, 0.432]


def monotone_cubic(x, y):
    """Fritsch-Carlson monotone cubic interpolator over sorted control points."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if np.any(np.diff(x) <= 0):
        raise ValueError("control points must be strictly increasing in x")
    h = np.diff(x)
    delta = np.diff(y) / h
    m = np.zeros_like(x)
    m[1:-1] = (delta[:-1] + delta[1:]) / 2.0
    m[0], m[-1] = delta[0], delta[-1]
    # Flatten the tangent at any local extremum. Without this the clamp below is
    # not sufficient and a non-monotone SOURCE/TARGET pair overshoots, which
    # bands in the shadows. Cannot trigger while the measured data is monotone,
    # but a refit is exactly when it would.
    for i in range(1, len(x) - 1):
        if delta[i - 1] * delta[i] <= 0:
            m[i] = 0.0
    for i in range(len(delta)):
        if delta[i] == 0:
            m[i] = m[i + 1] = 0.0
        else:
            a, b = m[i] / delta[i], m[i + 1] / delta[i]
            s = a * a + b * b
            if s > 9.0:
                t = 3.0 / np.sqrt(s)
                m[i], m[i + 1] = t * a * delta[i], t * b * delta[i]

    def f(t):
        t = np.clip(t, x[0], x[-1])
        i = np.clip(np.searchsorted(x, t) - 1, 0, len(x) - 2)
        dx = x[i + 1] - x[i]
        s = (t - x[i]) / dx
        s2, s3 = s * s, s * s * s
        return ((2 * s3 - 3 * s2 + 1) * y[i]
                + (s3 - 2 * s2 + s) * dx * m[i]
                + (-2 * s3 + 3 * s2) * y[i + 1]
                + (s3 - s2) * dx * m[i + 1])

    return f


def tone_curve():
    # Anchor at both ends so the curve stays inside the domain and keeps white.
    xs = [0.0] + SOURCE + [1.0]
    ys = [0.0] + TARGET + [1.0]
    return monotone_cubic(xs, ys)


def build():
    f = tone_curve()
    ramp = np.linspace(0.0, 1.0, SIZE)
    b, g, r = np.meshgrid(ramp, ramp, ramp, indexing="ij")
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    safe = np.maximum(lum, 1e-4)
    scale = f(safe) / safe
    # The curve lifts above L~0.315, so a saturated bright node can push a channel
    # past 1. Clipping it there would desaturate and shift hue - the opposite of
    # the point of scaling by luminance. Cap the scale instead: the node gives up
    # a little of its lift and keeps its channel ratios exactly.
    peak = np.maximum(np.maximum(r, g), b)
    scale = np.minimum(scale, 1.0 / np.maximum(peak, 1e-6))
    out = np.stack([r * scale, g * scale, b * scale], axis=-1)
    return np.clip(out, 0.0, 1.0).reshape(-1, 3)


if __name__ == "__main__":
    table = build()
    with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
        fh.write('TITLE "Tone Target Match"\n')
        fh.write(f"LUT_3D_SIZE {SIZE}\n")
        fh.write("DOMAIN_MIN 0.0 0.0 0.0\n")
        fh.write("DOMAIN_MAX 1.0 1.0 1.0\n")
        for px in table:
            fh.write(f"{px[0]:.6f} {px[1]:.6f} {px[2]:.6f}\n")
    print(f"wrote {OUT} ({SIZE}^3 = {len(table)} entries)")

    f = tone_curve()
    print("\ncurve check (source -> mapped, wanted):")
    for p, s, t in zip(PERCENTILES, SOURCE, TARGET):
        print(f"   p{p:<3} {s:.3f} -> {float(f(np.array([s]))[0]):.3f}   (wanted {t:.3f})")
