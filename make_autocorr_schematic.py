#!/usr/bin/env python3
"""
make_autocorr_schematic.py — Figure 2 for the proposal.

Demonstrates the §2.3 identity on a REAL positive-ICH slice from the RSNA cohort:
the 1-D autocorrelation of each projection equals the Radon projection of the 2-D
image autocorrelation,

    r_θ(u) = Radon[R_f](u, θ),

so the sinogram's second-order structure IS the image autocorrelation, sampled by
angle. Mechanism (Fourier-slice + Wiener-Khinchin):
    FT_1D[r_θ](ω) = |FT_1D[p_θ](ω)|² = |F(ω cosθ, ω sinθ)|²
                  = radial slice of |F|² = radial slice of FT_2D[R_f].

DATA-USE COMPLIANCE: the source is a real RSNA/Kaggle positive-ICH CT slice, but
the raw image is NOT displayed (nor is the raw, invertible sinogram). Only
non-identifiable derived statistics are shown — the 2-D autocorrelation R_f and
the power spectrum |F|² discard phase and cannot be inverted to the image; the
projection autocorrelations likewise discard per-projection phase. These are
statistics of the image, not the image.

Panels:
  A  2-D autocorrelation R_f            (real ICH slice; source image not shown)
  B  power spectrum |F|² with radial slice at θ0   (Fourier-slice mechanism)
  C  autocorrelation of each projection r_θ(u), column θ0 marked
  D  Radon transform of R_f             (projections of the autocorrelation)
"""
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.transform import radon

SUBSET = "/mnt/mlcache/subset50k"
SLICE_ID = "ID_00170833c"          # a real intraparenchymal-positive slice
N_ANGLES = 360
THETA0 = 55.0                      # the highlighted radial slice / column


def load_real_ich_slice(slice_id):
    """Load a real positive-ICH recon tensor (uint16 linear HU) and return a
    mean-subtracted float image. The image itself is used only for computation
    and is never displayed."""
    arr = np.load(f"{SUBSET}/recon/{slice_id}.npy").astype(np.float32)
    return arr - arr.mean()


def autocorr_2d_linear(f):
    n = f.shape[0]
    P = np.fft.fft2(f, s=(2 * n, 2 * n))
    ac = np.fft.ifft2(np.abs(P) ** 2).real
    ac = np.fft.fftshift(ac)
    c, h = n, n // 2
    return ac[c - h:c + h, c - h:c + h]


def proj_autocorr_sinogram(sino):
    n = sino.shape[0]
    out = np.zeros_like(sino)
    h = n // 2
    for j in range(sino.shape[1]):
        p = sino[:, j]
        full = np.correlate(p, p, mode="full")
        c = len(full) // 2
        out[:, j] = full[c - h:c - h + n]
    return out


def main():
    f = load_real_ich_slice(SLICE_ID)                        # real image (not shown)
    theta = np.linspace(0.0, 180.0, N_ANGLES, endpoint=False)

    PS = np.abs(np.fft.fftshift(np.fft.fft2(f))) ** 2        # power spectrum |F|²
    R_f = autocorr_2d_linear(f)                             # 2-D autocorrelation
    sino = radon(f, theta=theta, circle=True)               # (invertible — not shown)
    r_theta = proj_autocorr_sinogram(sino)                  # path 1
    radon_Rf = radon(R_f, theta=theta, circle=True)         # path 2

    def coln(a):
        m = a.max(axis=0, keepdims=True); m[m == 0] = 1.0
        return a / m
    c1, c2 = coln(r_theta), coln(radon_Rf)
    pear = float(np.corrcoef(c1.ravel(), c2.ravel())[0, 1])

    n = f.shape[0]
    extent = [0, 180, n, 0]

    fig, ax = plt.subplots(2, 2, figsize=(11.0, 10.6))

    # A  2-D autocorrelation (log)
    Rn = np.abs(R_f) / np.abs(R_f).max()
    ax[0, 0].imshow(np.log1p(Rn * 60), cmap="magma")
    ax[0, 0].set_title("A.  2-D autocorrelation  R_f  (log)\nreal ICH-positive slice — "
                       "source image not shown", fontsize=10.5)
    ax[0, 0].axis("off")

    # B  power spectrum with radial slice at THETA0
    PSn = np.log1p(PS / PS.max() * 1e4)
    ax[0, 1].imshow(PSn, cmap="magma")
    cy = cx = n / 2.0
    L = n * 0.48
    a = np.deg2rad(THETA0)
    ax[0, 1].plot([cx - L * np.cos(a), cx + L * np.cos(a)],
                  [cy - L * np.sin(a), cy + L * np.sin(a)],
                  "-", color="#39d0ff", lw=1.8)
    ax[0, 1].set_title("B.  Power spectrum  |F|²  (log)\nradial slice at θ₀ = 55°  "
                       "(Fourier-slice)", fontsize=10.5)
    ax[0, 1].axis("off")

    # C  projection autocorrelation sinogram, column THETA0 marked
    ax[1, 0].imshow(c1, cmap="magma", aspect="auto", extent=extent)
    ax[1, 0].axvline(THETA0, color="#39d0ff", lw=1.6, ls="--")
    ax[1, 0].set_title("C.  Autocorrelation of each projection  r_θ(u)\ncolumn θ₀ ↔ radial "
                       "slice in B", fontsize=10.5)
    ax[1, 0].set_xlabel("projection angle  θ  (deg)")
    ax[1, 0].set_ylabel("lag  u")

    # D  Radon of R_f
    ax[1, 1].imshow(c2, cmap="magma", aspect="auto", extent=extent)
    ax[1, 1].set_title("D.  Radon transform of  R_f\nprojections of the autocorrelation",
                       fontsize=10.5)
    ax[1, 1].set_xlabel("projection angle  θ  (deg)")
    ax[1, 1].set_ylabel("lag  u")

    fig.text(0.50, 0.985, "Two paths to the same sinogram:   r_θ = Radon[R_f]   "
             "(real ICH slice — Pearson r = %.3f)" % pear, ha="center", va="top",
             fontsize=12.5, color="#12305a", weight="bold")
    plt.annotate("", xy=(0.555, 0.30), xytext=(0.445, 0.30),
                 xycoords="figure fraction", textcoords="figure fraction",
                 arrowprops=dict(arrowstyle="<|-|>", color="#8a1c1c", lw=1.6))
    fig.text(0.50, 0.315, "C ≈ D", ha="center", fontsize=9.5,
             color="#8a1c1c", weight="bold")

    fig.text(0.5, 0.015,
             "Computed from a real RSNA positive-ICH CT slice; per competition data-use "
             "terms the raw image (and its invertible sinogram) are NOT shown — only "
             "non-identifiable, phase-free statistics. Each projection's 1-D autocorrelation "
             "is a radial slice of |F|² (Fourier-slice; blue line in B ↔ blue column in C); "
             "swept over angle these equal the Radon transform of the 2-D autocorrelation "
             "(D). The sinogram's second-order structure IS the object's autocorrelation, "
             "sampled by angle — the basis for reading texture and focal ICH in projection "
             "space.", ha="center", va="bottom", fontsize=8.5, wrap=True, color="#222222")

    fig.subplots_adjust(left=0.06, right=0.97, top=0.90, bottom=0.20,
                        hspace=0.32, wspace=0.16)
    fig.savefig("schematic_autocorrelation.png", dpi=160)
    print("wrote schematic_autocorrelation.png | slice=%s | Pearson r(C,D)=%.4f"
          % (SLICE_ID, pear))


if __name__ == "__main__":
    main()
