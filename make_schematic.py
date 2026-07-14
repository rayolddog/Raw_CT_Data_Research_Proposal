#!/usr/bin/env python3
"""
make_schematic.py — Figure 1 for the proposal.

Demonstrates the image <-> sinogram relationship and the central motion argument:
a focal abnormality is a clean, connected SINUSOID in projection space, whereas
patient motion (modeled as a SUDDEN JUMP partway through the scan) back-projects
into sharp STREAK artifact that obscures the same lesion in the reconstructed image.

Panels (loop: image -> Radon -> sudden motion jump -> inverse Radon):
  A  ideal image  f   (Shepp-Logan head phantom + focal hyperdensity / "metal")
  B  sinogram R[f]     (clean; lesion sinusoid overlaid)
  C  sinogram, jumped  (patient motion; step offset past the jump angle)
  D  inverse Radon of the jumped sinogram (motion streak artifact, lung window)

Everything is a real transform (skimage radon/iradon); the lesion track is the
argmax ridge of the lesion-only sinogram, offset by the same step, so the
"still coherent" claim is shown, not asserted. The reconstruction panels use a
wide, low-centered "lung" display window, which is where motion streak is most
conspicuous clinically.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import shift
from skimage.transform import radon, iradon, rescale
from skimage.data import shepp_logan_phantom

N_ANGLES = 360
JUMP_PIX = 30.0         # sudden detector displacement (rows) at the jump instant
JUMP_FRAC = 0.55        # fraction of the scan completed when the patient jumps

# "Lung" display window for the reconstructions: wide width, low center. Phantom
# units are ~[0, 1.8]; a wide low-centered window stretches faint motion streak
# (small +/- fluctuations on the dark background) into visible gray while bone
# and the lesion saturate to white — exactly the trade a lung window makes.
LUNG_CENTER, LUNG_WIDTH = 0.25, 1.2
LUNG_VMIN = LUNG_CENTER - LUNG_WIDTH / 2.0
LUNG_VMAX = LUNG_CENTER + LUNG_WIDTH / 2.0


def build_phantom():
    """Shepp-Logan head phantom + a small bright focal 'abnormality' (metal-like)."""
    ph = shepp_logan_phantom()                       # ~400x400, values in [0,1]
    ph = rescale(ph, 360 / ph.shape[0], anti_aliasing=True)
    n = ph.shape[0]
    yy, xx = np.mgrid[0:n, 0:n]
    # lesion inside the brain, off-center
    cy, cx, r = int(0.40 * n), int(0.60 * n), 7
    lesion = ((yy - cy) ** 2 + (xx - cx) ** 2) <= r ** 2
    ph = ph.copy()
    ph[lesion] = 1.8                                 # brighter than skull (=1)
    lesion_only = np.zeros_like(ph)
    lesion_only[lesion] = 1.8
    return ph, lesion_only, (cy, cx, r)


def jump_sinogram(sino, d, jfrac):
    """Sudden patient motion: the detector position is unchanged for the first
    `jfrac` of projection angles, then STEPS by `d` rows for the remainder and
    stays there. Models a single abrupt head jerk mid-scan (still -> jump -> still).
    Returns (jumped_sinogram, jump_column_index)."""
    j0 = int(round(jfrac * sino.shape[1]))
    out = sino.copy()
    out[:, j0:] = shift(sino[:, j0:], shift=[d, 0.0], order=1,
                        mode="constant", cval=0.0)
    return out, j0


def lesion_track(lesion_sino):
    """Detector row of the lesion at each angle = argmax ridge of the lesion-only
    sinogram (exact, in skimage's own coordinate convention)."""
    return np.argmax(lesion_sino, axis=0).astype(float)


def main():
    ph, lesion_only, (cy, cx, r) = build_phantom()
    theta = np.linspace(0.0, 180.0, N_ANGLES, endpoint=False)

    sino = radon(ph, theta=theta, circle=True)                 # (n_det, n_angles)
    lesion_sino = radon(lesion_only, theta=theta, circle=True)
    sino_jump, j0 = jump_sinogram(sino, JUMP_PIX, JUMP_FRAC)
    theta_jump = theta[j0]

    recon_clean = iradon(sino, theta=theta, filter_name="ramp", circle=True)
    recon_motion = iradon(sino_jump, theta=theta, filter_name="ramp", circle=True)

    # lesion sinusoid track (clean) and its jumped image (step at the jump angle)
    track = lesion_track(lesion_sino)
    track_jump = track.copy()
    track_jump[j0:] += JUMP_PIX

    fig, ax = plt.subplots(2, 2, figsize=(11.5, 10.6))
    im_kw = dict(cmap="gray")
    sino_extent = [0, 180, sino.shape[0], 0]

    # A  ideal image
    ax[0, 0].imshow(ph, **im_kw, vmin=0, vmax=1.8)
    ax[0, 0].add_patch(plt.Circle((cx, cy), r + 6, ec="#ff3b3b", fc="none", lw=1.6))
    ax[0, 0].set_title("A.  Ideal image  f\nShepp–Logan head phantom + focal hyperdensity",
                       fontsize=11)
    ax[0, 0].axis("off")

    # B  clean sinogram + lesion track
    ax[0, 1].imshow(sino, **im_kw, aspect="auto", extent=sino_extent)
    ax[0, 1].plot(theta, track, "--", color="#ff3b3b", lw=1.2, alpha=0.9)
    ax[0, 1].set_title("B.  Sinogram  R[f]  (clean)\nlesion = clean connected sinusoid",
                       fontsize=11)
    ax[0, 1].set_xlabel("projection angle  θ  (deg)")
    ax[0, 1].set_ylabel("detector position  s")

    # C  jumped sinogram + jumped track (step discontinuity marked)
    ax[1, 1].imshow(sino_jump, **im_kw, aspect="auto", extent=sino_extent)
    ax[1, 1].plot(theta[:j0], track_jump[:j0], "--", color="#ff3b3b", lw=1.2, alpha=0.9)
    ax[1, 1].plot(theta[j0:], track_jump[j0:], "--", color="#ff3b3b", lw=1.2, alpha=0.9)
    ax[1, 1].axvline(theta_jump, color="#ffd23b", lw=1.4, ls=":", alpha=0.9)
    ax[1, 1].set_title("C.  Sinogram with sudden motion jump\nstep offset at the jump angle; trace still coherent",
                       fontsize=11)
    ax[1, 1].set_xlabel("projection angle  θ  (deg)")
    ax[1, 1].set_ylabel("detector position  s")

    # D  reconstruction from jumped sinogram — LUNG WINDOW (wide, low-centered)
    ax[1, 0].imshow(recon_motion, **im_kw, vmin=LUNG_VMIN, vmax=LUNG_VMAX)
    ax[1, 0].add_patch(plt.Circle((cx, cy), r + 6, ec="#ff3b3b", fc="none", lw=1.6))
    ax[1, 0].set_title("D.  Inverse Radon of jumped sinogram (lung window)\nsharp motion streaks obscure the lesion",
                       fontsize=11)
    ax[1, 0].axis("off")

    # flow header + horizontal loop arrows (A->B across the top, C->D across the
    # bottom); the B->C step is self-evident (two stacked sinograms) so no arrow.
    akw = dict(xycoords="figure fraction", textcoords="figure fraction",
               arrowprops=dict(arrowstyle="-|>", color="#12305a", lw=2))
    fig.text(0.50, 0.985, "image  →  Radon  →  sudden patient motion (jump)  →  inverse Radon",
             ha="center", va="top", fontsize=12, color="#12305a", weight="bold")
    plt.annotate("", xy=(0.55, 0.70), xytext=(0.45, 0.70), **akw)            # A->B
    plt.annotate("", xy=(0.45, 0.30), xytext=(0.55, 0.30), **akw)            # C->D

    fig.text(0.5, 0.015,
             "A sudden mid-scan patient jump is a single step discontinuity in the sinogram "
             "(C, yellow line): the lesion track offsets abruptly but stays a clean, connected "
             "sinusoid. Back-projecting that discontinuity, however, throws sharp radiating "
             "streaks across the reconstructed image (D, lung window) that bury the same "
             "lesion. Detecting the lesion in sinogram space can thus be more robust to motion "
             "than detecting it in the reconstructed image.",
             ha="center", va="bottom", fontsize=9, wrap=True, color="#222222")

    fig.subplots_adjust(left=0.06, right=0.97, top=0.90, bottom=0.15,
                        hspace=0.30, wspace=0.16)
    fig.savefig("schematic_sinogram_motion.png", dpi=160)
    print("wrote schematic_sinogram_motion.png",
          "| recon_motion range %.2f..%.2f" % (recon_motion.min(), recon_motion.max()))


if __name__ == "__main__":
    main()
