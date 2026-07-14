# Key Equations — CT Projection-Domain Deep Learning

*A reference sheet for the projection-domain (sinogram / raw-CT) work: the parallel-beam
model, the exact fan-beam↔parallel rebinning, the helical trajectory, the
autocorrelation identity, the Gabor/curvelet feature basis, and the information ordering.*

## Notation

- **f(x, y)** — attenuation field (the image); **F** — its 2-D Fourier transform.
- **θ, s** — parallel-beam projection angle and detector offset.
- **β, γ** — fan-beam **source angle** and **fan angle** of a ray.
- **u** — flat-detector coordinate (projected to the isocenter plane).
- **D** — source-to-isocenter distance.
- **h** — helical pitch (table travel per full rotation).
- **(r₀, φ₀)** — polar position of a point in the object.

---

## 1. Parallel-beam Radon transform

@@ p_θ(s) = ∫ f(s cosθ − t sinθ,  s sinθ + t cosθ) dt

A point mass at (r₀, φ₀) maps to a **sinusoid** in the sinogram:

@@ s = r₀ cos(θ − φ₀)

## 2. Fourier-slice (projection-slice) theorem

@@ (1-D FT of p_θ)(ω) = F(ω cosθ,  ω sinθ)

## 3. Filtered back-projection (inversion)

@@ f(x, y) = ∫₀^π (p_θ ∗ H)(x cosθ + y sinθ) dθ,     H(ω) = |ω|

The ramp filter |ω| is the source of high-frequency noise amplification / ill-conditioning.

## 4. Fan-beam → parallel-beam rebinning (exact)

**Equiangular** (curved/third-generation detector), ray indexed by (β, γ):

@@ θ = β + γ,     s = D sin γ

**Equispaced** (flat detector), ray indexed by (β, u), with γ = arctan(u / D):

@@ θ = β + arctan(u / D),     s = u D / √(D² + u²)

Both reduce to the same relation `s = D sin γ`, `θ = β + γ` — the mapping is a
**nonlinear** resampling (note the `sin`), not an affine transform.

## 5. Helical (spiral) source trajectory

@@ z(β) = (h / 2π) β     (pitch h per rotation)

The source moves in z during rotation, so projections through a given axial plane are
incomplete; a per-slice sinogram is an *interpolated* (partially reconstructed) object.

## 6. The sinogram encodes the image autocorrelation (Wiener–Khinchin + Fourier-slice)

Image autocorrelation and its spectrum:

@@ R_f(τ) = ∫ f(x) f(x + τ) dx,     (2-D FT of R_f) = |F|²

The 1-D autocorrelation of one projection is a **radial slice of the 2-D power spectrum**:

@@ r_θ(u) = ∫ p_θ(s) p_θ(s + u) ds,     (1-D FT of r_θ)(ω) = |F(ω cosθ, ω sinθ)|²

Sweeping θ therefore fills |F|², whose inverse transform is R_f — equivalently:

@@ r_θ = Radon[R_f]

(Exact in parallel-beam; under fan-beam/helical geometry it holds only after the
nonlinear rebinning of §4 and is treated as a conceptual guide.)

## 7. Gabor atom — and its curvelet generalization

@@ ψ(x;  f₀, θ, σ, φ) = exp(−|x|² / 2σ²) · cos(2π f₀ (x · n_θ) + φ)

Jointly localized in space and frequency (uncertainty bound), selective in orientation θ
and scale ∝ 1/f₀. Under real divergent geometry the point-response is a *curved* ridge,
so the matched basis generalizes from straight Gabor atoms toward **curvelet-type**
elements (curvelet / curved-Gabor / shearlet) — still anisotropic, oriented, and
scale-selective, but adapted to a curved locus.

## 8. Information ordering (data-processing inequality)

For a real scan, Y (diagnosis) → f → p_raw (raw fan-beam helical data) → x (image) is a
Markov chain, so:

@@ I(Y; x) ≤ I(Y; p_raw)

For a **synthetic** sinogram p_syn = R[x] (forward-projected from the image):

@@ I(Y; p_syn) ≤ I(Y; x) ≤ I(Y; p_raw)

The synthetic sinogram cannot exceed the image it came from — the honest bound behind the
"representation and architecture, not fidelity" scope of the synthetic-data experiments.
