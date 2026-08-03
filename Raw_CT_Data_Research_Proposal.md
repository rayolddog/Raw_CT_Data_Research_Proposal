# Deep Learning on Raw CT Projection Data: A Research Proposal

**Author:** John Bramble, MD

**Research collaborator:** Claude (Anthropic Fable 5)

*A technical research proposal — not a funding application. It makes the case for training deep neural networks on **raw CT projection (sinogram) data** rather than on reconstructed images; sets out the mathematical basis in the Radon transform and the image autocorrelation; presents preliminary results indicating the approach is feasible; and specifies what an academically-resourced investigator would need to extend it to real fan-beam / helical acquisitions and to clinically-acquired data at scale.*

**A note on this collaboration.** This proposal is the product of a collaboration between a physician and an artificial-intelligence system. John Bramble, a practicing radiologist, originated its central idea — that diagnosis is better posed on raw CT projection data than on the reconstructed image — together with the clinical insights that motivate it (including the posterior-fossa argument of §4); he directs the investigation, supplies the domain judgment, and is responsible for its claims. Claude, Anthropic's Fable 5 model, has served as an active research collaborator throughout: helping to formalize the mathematics, survey the literature and locate relevant public datasets, design and interpret the computational experiments, and draft and check this document. The division of labor is characteristic of a productive human–AI partnership — the human contributes the originating insight, the clinical grounding, and the accountability; the AI contributes breadth, formalization, and tireless drafting and verification. The work is offered, in part, as a modest demonstration that such a partnership can carry out serious scientific groundwork.

---

## 1. Introduction and thesis

A modern CT scanner does not measure an image. It measures **projections** — line integrals of X-ray attenuation across a divergent (fan/cone) beam, sampled as the source rotates helically around the patient. The familiar axial image is a *computed* object, produced from those projections by a reconstruction algorithm that is regularized, vendor-specific, and — in practice — not invertible without loss.

The thesis of this proposal is that, for **automated diagnosis**, the projection data are the better substrate. The argument is precise and information-theoretic: because the reconstructed image is computed from the projections alone, it can carry **no more** diagnostic information than the projections already contain (the data-processing inequality, §3.5). Reconstruction is therefore, at best, information-preserving and, in reality, information-discarding: finite kernels, discretization, bit-depth quantization, apodization, and display windowing all break the equality. A network that reads the projections directly is reading the data before that loss — and before the reconstruction-*introduced* artifacts (beam-hardening streak, cone-beam and iterative-reconstruction textures) that never existed in the measurements.

Deep networks have already shown they can read diagnostic structure from projections: several groups have learned the sinogram→image reconstruction map directly (AUTOMAP; Li, Chen et al.; DEAR — §2). Those networks learn to make a *better image*. This proposal reuses the same projection-domain modeling capability for a **different endpoint — the diagnosis itself** — and never forms the image. The exemplar task throughout is detection of acute **intracranial hemorrhage (ICH)**, a time-critical finding for which the earliest possible flag has clinical value.

The proposal is deliberately honest about what present evidence can and cannot show. Real CT is fan-beam and helical; the public data used for the feasibility experiments provide reconstructed images only, so the experimental sinograms are **synthetic parallel-beam re-projections**. That regime rigorously tests **representation and architecture** — *can* a network detect ICH from a projection encoding, and how efficiently — but it cannot demonstrate the raw-data fidelity and artifact advantages, which are properties of true measured projections. Separating these two regimes is the central discipline of the argument, and defines the work that remains (§6).

---

## 2. Historical background

The mathematics predates the machine by half a century. **Johann Radon** (1917) proved that a function on the plane is determined by its integrals over all lines, and gave the inversion — the theoretical core of tomography. **Ronald Bracewell** (1956) independently used projection–reconstruction ("strip integration") to map the radio sky, and articulated the **Fourier-slice relationship**. **Dennis Gabor** (1946) introduced the jointly space/frequency-localized "elementary signals" now called Gabor atoms in his *Theory of Communication*; Gabor also invented **holography** (Nobel Prize, 1971).

The clinical machine followed from **Allan Cormack** (1963–64), who worked out the radiological line-integral inversion, and **Godfrey Hounsfield** (1971–73), who built the first scanner at EMI; the two shared the 1979 Nobel Prize in Physiology or Medicine. The canonical mathematical treatment — parallel- and fan-beam filtered back-projection, algebraic reconstruction — is Kak & Slaney (1988). **Helical (spiral) CT** (Kalender et al., 1990) made the source trace a helix relative to the patient; exact inversion for the helical cone beam came only later (**Katsevich**, 2002). The autocorrelation–power-spectrum identity underlying the feature argument of §3.3 is the **Wiener–Khinchin** theorem (Wiener 1930; Khinchin 1934).

The deep-learning era reopened the inverse problem: **AUTOMAP** (Zhu et al., *Nature* 2018) learned image formation as a manifold transform; **Li, Chen et al.** (*IEEE TMI* 2019) learned the sinogram→image map directly on CT; **Xie, Shan & Wang** (DEAR, 2019) pursued few-view reconstruction. Each learns to make a *better image*. The present proposal reuses the same projection-domain modeling capability for a *different endpoint* — the diagnosis — and never forms the image.

---

## 3. Mathematical framework

### 3.1 The real acquisition geometry: fan-beam and helical

A clinical scanner is **not** a parallel-beam Radon device. A point source illuminates a detector arc, so the measured rays are **divergent (fan-beam)**: a projection is g(β, γ), with β the source angle and γ the fan angle of a ray. Fan data relate to parallel data by a rebinning,

@@ θ = β + γ,     s = D sin γ,

with D the source-to-isocenter distance; fan-beam FBP additionally carries a distance weighting. Acquisition is **helical**: the table translates while the gantry rotates, so the source describes a helix relative to the patient,

@@ z(β) = (h / 2π) β     (pitch h per rotation).

Reconstructing an axial plane then requires longitudinal interpolation (single-slice helical), cone-beam methods (FDK), or exact helical inversion (Katsevich 2002), and in practice increasingly **model-based iterative** or **deep-learning** reconstruction. The operational point: **raw data are divergent-beam, helically sampled, and reconstruction is a regularized, vendor-specific, non-injective-in-practice inverse.** That non-injectivity is exactly where kernel choice, apodization, and iterative priors inject the image-domain variability that projection-domain detection avoids.

### 3.2 The parallel-beam Radon transform (an idealization we use to reason)

The parallel-beam transform is the tractable model that exhibits projection structure. For attenuation field f,

@@ p_θ(s) = ∫ f(s cosθ − t sinθ,  s sinθ + t cosθ) dt.

A point mass at polar position (r₀, φ₀) maps to s = r₀ cos(θ − φ₀) — a sinusoid, hence *sino*gram. The **Fourier-slice theorem** connects it to the image spectrum F:

@@ (1-D FT of p_θ)(ω) = F(ω cosθ,  ω sinθ),

so projections tile the 2-D spectrum radially and f is recoverable with full angular coverage. Inversion is filtered back-projection,

@@ f(x,y) = ∫₀^π (p_θ ∗ H)(x cosθ + y sinθ) dθ,     H(ω) = |ω|,

whose ramp filter |ω| amplifies high-frequency noise — the ill-conditioning that forces the regularization choices of §3.1. This parallel-beam model is an **illustrative, computable stand-in** for the projection structure; it is not the acquisition physics, and the synthetic data of §5 inherit its idealization.

### 3.3 The sinogram encodes the image autocorrelation

The feature the detector must read is second-order. Define the image autocorrelation

@@ R_f(τ) = ∫ f(x) f(x + τ) dx,     (2-D FT of R_f) = |F|²     (Wiener–Khinchin).

The 1-D autocorrelation of one projection,

@@ r_θ(u) = ∫ p_θ(s) p_θ(s + u) ds,     (1-D FT of r_θ)(ω) = |F(ω cosθ, ω sinθ)|²,

is, by Fourier-slice applied to |·|², a **radial slice of the 2-D power spectrum**. Sweeping θ fills |F|², whose inverse transform is R_f. In words: **the projection autocorrelations, across angle, reconstruct the image autocorrelation** — each projection is an angular sample of the object's correlation/texture structure. A hemorrhage — a focal, hyperattenuating, texturally distinct region — perturbs R_f in an orientation-dependent way that is present, angularly encoded, in the sinogram.

![**Figure 1.** The autocorrelation identity, computed on a **real RSNA positive-ICH slice** (the raw image and its invertible sinogram are not shown — only non-identifiable, phase-free statistics, per competition data-use terms). **(A)** 2-D autocorrelation R_f (log). **(B)** Power spectrum |F|² with a radial slice at θ₀ = 55° (Fourier-slice). **(C)** The 1-D autocorrelation of each projection, r_θ(u), stacked over angle; the marked column corresponds to the radial slice in B. **(D)** The Radon transform of the 2-D autocorrelation R_f. Panels C and D are equal (Pearson r = 0.995), demonstrating r_θ = Radon[R_f]: the sinogram's second-order structure is the object's autocorrelation, sampled by angle.](schematic_autocorrelation.png){0.82}

### 3.4 Gabor wavelets: the natural feature basis

A Gabor atom is a Gaussian-windowed oriented sinusoid,

@@ ψ(x;  f₀, θ, σ, φ) = exp(−|x|² / 2σ²) · cos(2π f₀ (x · n_θ) + φ),

jointly localized in space and frequency (attaining the uncertainty bound; Gabor 1946) and selective in orientation θ and scale ∝ 1/f₀. Two facts make it the right basis here. First, an oriented image structure maps to a localized (θ, s) signature in the sinogram, so a Gabor bank over the sinogram reads projected orientation and scale directly — Gabor energy is a **localized power spectrum**, complementing the global autocorrelation of §3.3. Second, the first convolutional layers of trained CNNs are empirically Gabor-like; a CNN on the sinogram is, in effect, learning a **data-adaptive Gabor/autocorrelation front end**. A projection-domain detector therefore formalizes what its early layers would learn regardless — usable both as initialization (a fixed Gabor front end) and as an interpretability handle. Under real divergent geometry the basis generalizes rather than breaks: a point traces a *curved* locus rather than a straight ridge — indeed even the parallel-beam trace is a sinusoid, already curved — so the matched atoms shift from straight Gabor wavelets toward **curvelet-type** elements (as in curvelet, curved-Gabor, and shearlet systems): still anisotropic, oriented, and scale-selective, but adapted to a curved locus. Because a single small convolution kernel can represent only a locally-linear segment, this curvature is either composed across layers or built in explicitly (e.g. deformable convolutions); either way the Gabor rationale survives real geometry as a **design principle** rather than a closed-form identity. The exact fan-to-parallel relations and the other key identities are collected in `EQUATIONS.md`.

### 3.5 Information ordering — and an honest caveat

For a real scan the causal chain is

@@ Y (diagnosis)  →  f (attenuation field)  →  p_raw (fan-beam helical data)  →  x (reconstructed image),

with x computed from p_raw alone, so (Y, p_raw, x) form a Markov chain. The **data-processing inequality** gives

@@ I(Y; x) ≤ I(Y; p_raw).

Reconstruction cannot increase the diagnostic information the data carry; equality requires reconstruction to be a *sufficient statistic* for Y, which finite kernels, discretization, quantization, apodization, and display windowing all break. **This is the rigorous form of the fidelity claim.**

But public archives provide x only. Synthetic sinograms are **produced by forward projection**, p_syn = R[x], so the chain is Y → x → p_syn and

@@ I(Y; p_syn) ≤ I(Y; x) ≤ I(Y; p_raw).

The synthetic sinogram is a re-encoding of the already-reconstructed image — it cannot exceed it — and it is parallel-beam, not fan-beam helical. **Consequently synthetic-sinogram experiments can rigorously establish representation and architecture claims and simulate geometry/dose robustness, but they cannot by themselves demonstrate the raw-domain fidelity or artifact advantages.** Those require true p_raw (§6).

![**Figure 2.** The image↔projection loop and the motion argument. **(A)** An ideal image f (Shepp–Logan head phantom with a focal hyperdensity, red circle). **(B)** Its sinogram R[f]; the lesion is a clean, connected sinusoid (red track). **(C)** A sudden mid-scan patient motion, modeled as a step offset of the sinogram past the jump angle (yellow line) — the lesion trace offsets abruptly but each segment remains a coherent sinusoid. **(D)** Inverse Radon of the motion-corrupted sinogram, shown in a wide, low-centered lung window: the step discontinuity back-projects into sharp radiating streaks that bury the lesion. The abnormality's signature is better preserved in projection space (B, C) than in the reconstructed image (D), which motivates detection before reconstruction. All panels are real Radon/inverse-Radon transforms; the lesion track is the argmax ridge of the lesion-only sinogram under the same step offset.](schematic_sinogram_motion.png){0.75}

---

## 4. Advantages of the projection-domain approach

Each advantage below is, at root, a property of true raw projections. The synthetic experiments of §5 can probe some of them by simulation but cannot fully demonstrate them.

1. **Reconstruction artifacts are avoided.** Beam-hardening, streak, cone-beam and iterative-reconstruction textures are introduced *by* reconstruction; a sinogram reader never sees them. *(Approachable by simulation; confirmable on raw data.)*
2. **Higher intrinsic fidelity.** Raw projections carry greater bit depth and finer detector sampling than the windowed, resampled image retains — formalized by the data-processing inequality of §3.5. *(A raw-data property; not demonstrable on synthetic sinograms.)*
3. **Freedom from cross-algorithm inaccuracy.** An image-domain model inherits kernel/slice/vendor dependence — a leading cause of generalization failure across reconstructions. A projection-domain detector is instead calibrated to one specific scanner's raw output; the reconstruction-algorithm variable is removed, not averaged over. In practice these kernels are set to individual radiologist *preference* at protocol setup — radiologists in one group routinely disagree over reconstruction algorithms, or simply reject studies as "too noisy" — and vary across a multi-vendor fleet, resisting standardization even under a coordinated institutional effort; so the image-domain training distribution is shaped by a subjective, site-specific choice that raw data sits upstream of. This removes the **kernel/preference axis** of variation, not the scanner-geometry axis, which remains scanner-specific. *(Partly testable by simulated-geometry robustness; confirmable on raw data.)*
4. **Speed of notification.** Detection precedes or runs concurrently with reconstruction, enabling the earliest physically possible flag for a finding in which minutes change neurological outcome. *(Latency measurable now; end-to-end only on-scanner.)*
5. **Two semi-independent readers.** Because the projection- and image-domain models read *different representations* of the same acquisition, they can be deployed together: agreement raises confidence, disagreement flags a case for human review, and a single adversarial perturbation is far harder to craft so that it survives into both a raw sinogram and its reconstruction. *(On synthetic data the two are not truly independent — the sinogram is projected from the image — so this benefit cashes out fully only on real raw data.)*

**A concrete clinical case — the posterior fossa.** Reconstruction-introduced artifacts are not uniform across the brain: beam-hardening streaks (trans-petrous / interpetrous) and dental artifact fall hardest on the posterior fossa and brainstem — precisely where small hemorrhages are most easily missed on CT, and where MRI is well established as superior. Because a projection-domain reader never sees a reconstruction-introduced artifact, the approach's clinical value may be *largest exactly where CT is weakest*. This yields a falsifiable prediction, testable on a suitably labeled clinical cohort: a projection-domain detector should show a disproportionate sensitivity advantage for **infratentorial** (brainstem / posterior-fossa) hemorrhage relative to supratentorial, versus an image-domain detector.

---

## 5. Feasibility: preliminary results

All results below are on **synthetic parallel-beam sinograms** forward-projected from the RSNA 2019 Intracranial Hemorrhage dataset (>870,000 expertly labeled head-CT slices; six labels per slice — epidural, intraparenchymal, intraventricular, subarachnoid, subdural, and *any*). Each slice is encoded on a calibrated linear-Hounsfield scale (value = HU + 1024) and forked into a reconstructed branch (256×256, 12-bit) and a synthetic 256-angle sinogram (256×256, float32) from one shared source. Stratified cohorts of 50,000 and 200,000 slices were built with zero processing failures, preserving the full-dataset hemorrhage prevalence (14.5% positive) and all 32 observed label-combinations proportionally. A **twin protocol** underlies every comparison: an identical architecture and initialization is trained on each domain, differing *only* by input, so any difference is attributable to the domain. Work ran on a single-GPU workstation.

**Detection is feasible and matches the image domain.** A projection-domain network detects ICH at best validation **macro-AUC 0.946** across the six classes (SE-ResNeXt50, 200k), statistically on par with the matched image-domain twin. This answers the primary representation question affirmatively: a network *can* read ICH from a projection encoding, at image-domain-competitive accuracy.

**Both a slice-level and a patient-disjoint AUC are reported here, deliberately.** The 0.946 above was measured on a **slice-level** train/validation split. Because each RSNA study contributes ~30 axial slices of the same skull, a slice-level split places correlated slices from one patient on both sides — patient-level leakage that inflates the absolute AUC. This is easy to overlook (patient-level leakage from slice-level splits is a well-documented yet still-common pitfall in medical-imaging machine learning), so it is stated explicitly and paired with a **patient-disjoint** measurement: a patient-grouped subset — at most a few well-separated slices per patient — trained with a smaller projection-native CNN reached macro-AUC **0.871**. The 0.946 → 0.871 gap is *consistent with* removing leakage but does **not** cleanly isolate it, because the patient-disjoint run also differed in architecture (a smaller model), in scale (≈37,000 vs 200,000 slices, and the sweep below is still rising at 180k), and in class prevalence. The honest reading is a **bracket**: 0.871 is a conservative, leakage-free floor and 0.946 a leakage-susceptible ceiling, with the patient-clean value for a matched model at full data lying between them. Fixing that value — the *same* architecture and data volume, varying *only* slice-level vs patient-grouped splitting — is the natural first experiment for a group taking this work forward. The **domain-parity finding is unaffected either way**: the projection-vs-image twins share one identical split as a paired design, so any leakage acts symmetrically and the *relative* result (projection ≈ image) stands regardless of the absolute level.

**Performance still climbs with data.** A whitened data-size sweep shows both domains improving monotonically with N, the projection-domain model matching or marginally edging the image-domain model throughout (image-domain vs projection-domain macro-AUC):

- **N = 50,000:** 0.807 vs 0.824.
- **N = 100,000:** 0.832 vs 0.835.
- **N ≈ 180,000:** 0.847 vs 0.851.

The curve has not plateaued at ~180k — directly relevant to the data-scale argument of §6.

**The learned representation carries a domain signature.** With small local kernels on raw input, first-layer filters are indistinguishable between domains (both become generic oriented high-pass filters, AUC ≈ 0.81). But with large (31×31) stems and the input's average power spectrum **whitened** — removing the low-order statistics a network fits first — a clear domain signature emerges: sinogram-native kernels are **anisotropic, detector-axis-oriented, and lower-frequency**; image-domain kernels are isotropic and higher-frequency. Notably, the literal filtered-back-projection ramp (§3.2) does **not** appear — the network learns Radon-geometry-aligned structure directly, not textbook back-projection. This motivated a **sinogram-native anisotropic architecture** whose stem geometry is matched to the projection trace; an architecture search over such designs found a configuration with a small but **statistically robust** gain over the baseline (+0.005 macro-AUC; paired *t* = 3.4 over six seeds, Cohen's *d* = 1.4).

**Angular robustness is a real but conditional advantage.** Under simulated angular under-sampling (a removed wedge of projection angles, applied natively in each domain), a controlled factorial over augmentation × spectral-whitening × domain × model-capacity establishes that the projection domain's graceful degradation is **not** an unconditional intrinsic property. Much of it is a *trained* property — reproducible in the image domain once that branch is shown the matched physical corruption — and the residual projection-domain advantage is conditional: clearly present at moderate angle loss and high model capacity (retention 0.81 vs 0.57 for the image-domain model at a 30% wedge), but absent at severe angle loss, where both domains collapse without augmentation. This result was obtained as a self-imposed negative control — the apparent advantage was first traced to, and separated from, a training-augmentation confound before the conditional advantage was affirmed.

**Summary.** The representation-and-architecture question is answered: ICH is detectable from a projection-domain encoding at image-domain-competitive accuracy, the network's learned features are interpretable and geometry-aligned, and a projection-native architecture measurably helps. What synthetic parallel-beam data *cannot* show — the fidelity and artifact advantages of §4 — is the object of §6.

**Code availability.** The Python programs behind these preliminary studies — the forward-projection and cohort-construction pipeline, the twin-protocol training code, the kernel-visualization and whitening analyses, and the sinogram-native architecture search — are available at github.com/rayolddog/projection-domain-ich. They document the neural-network design considerations that arise specifically when the input is a raw CT sinogram rather than a reconstructed image, and are intended as background for a group developing a raw-data model for ICH or other pathologies.

---

## 6. Directions for further investigation

The feasibility evidence is a floor, not a ceiling. Realizing the argument requires three extensions, each needing resources beyond an individual investigator's reach — which is the honest reason this is offered as an open research proposal that an academically-credentialed group could take up.

**6.1 From synthetic parallel-beam to real fan-beam / helical geometry.** The experiments of §5 use a parallel-beam idealization. The immediate technical work is a **fan-beam helical acquisition-and-reconstruction pipeline** — helical→fan rebinning (§3.1) and a differentiable reconstruction layer — feeding the same detector head, so that the representation results can be reproduced under real geometry.

**6.2 Reconstruction-fidelity validation against measured projections.** The public **Mayo Clinic / AAPM** *LDCT-and-Projection-data* collection (TCIA) provides real measured **DICOM-CT-PD** helical projections *with paired reconstructions* for ~99 non-contrast head, 100 chest, and 100 abdomen exams, on Siemens and GE scanners. This is the natural instrument to **check reconstruction and forward-projection fidelity**: forward-project the collection's reconstructed images through the documented scanner geometry and compare against the actual measured projections, quantifying how faithfully a simulator reproduces real raw data. It requires no manufacturer agreement — only a data-use registration. It is, however, a *fidelity-validation* cohort, not a training cohort: ~99 head exams contain far too few hemorrhages to train a detector.

**6.3 The data-scale problem, and why academic/institutional access is essential.** Training a projection-domain ICH detector to clinical performance is a large-data problem, and the §5 sweep — still rising at ~180k — indicates the requirement is not small. A production-grade detector across six subtypes (several individually rare) plausibly needs on the order of **200,000 studies**. At the ~14% ICH prevalence of the reference cohort, 200,000 studies yield roughly 28,000 positive studies — and far fewer for the rarest subtypes; in unselected consecutive clinical practice the positive yield is typically *lower* than this curated figure, which only raises the number of studies required.

The binding constraint is subtler than volume: **raw projection data are not retained.** Scanners discard the projections after reconstruction, and PACS stores only images — so a large raw-data cohort cannot be mined retrospectively; it must be **prospectively retained at acquisition**, which requires both institutional data-governance and manufacturer cooperation. Under prospective accrual, a single large academic medical center performing on the order of ~50,000 head CTs per year would need roughly **four years** to reach 200,000 studies; a consortium of several large academic facilities could assemble a comparable cohort in about **one year**, which is the realistic route to an initial feasibility dataset. An investigator with academic credentials and institutional standing — able to secure IRB approval, multi-site data-governance, and a manufacturer raw-data agreement — is therefore a prerequisite, not a convenience.

**6.4 Task-driven angular weighting as implicit artifact reduction.** The angular-robustness experiments of §5 removed a *generic* span of projection angles. A more clinically-pointed version targets the specific angles that *cause* the dominant posterior-fossa artifact. The rays that traverse both petrous pyramids carry the highest line integrals; they are where a polychromatic beam hardens most, and they are the origin of the interpetrous streak and dark-band artifact that degrade the brainstem and posterior fossa (§4). This corruption is **not stochastic noise** (quantum mottle) but a **systematic, structured** artifact — in task-based / detection-theory terms, *structured noise* or *clutter* that lowers lesion detectability (the index d′) exactly where CT is already weakest.

Selectively **down-weighting** the highest-ray-sum projections — not deleting them, since naive deletion introduces its own limited-angle streaks — is a recognized artifact-reduction strategy, the same logic as metal-artifact reduction (MAR): identify the corrupted projections and discount them rather than reconstruct through them. The hypothesis this suggests is stronger than generic sparse-view robustness: **a projection-domain detector can learn a task-driven, angularly-adaptive weighting of the raw data — attending to informative rays and discounting the systematically-corrupted ones — performing implicit artifact reduction and detection in a single step, without reconstructing or inpainting.** Where reconstruction must either propagate the streak or estimate through it (MAR inpainting, with its own errors), a detector can learn to rely on the uncorrupted angles.

*Proposed experiment.* On posterior-fossa cases, selectively down-weight the high-ray-sum (petrous-traversing) projections and compare (a) a projection-domain detector's **infratentorial** sensitivity against (b) an image-domain model reading the correspondingly reconstructed images — streak-laden, or MAR-inpainted. If the projection detector preserves sensitivity where the image model loses it in the streak, this would demonstrate the posterior-fossa advantage (§4) on a concrete, physically-motivated intervention rather than a generic angular wedge. It also anticipates the natural objection — *why not simply apply a better beam-hardening correction?* — with a testable answer: posterior-fossa bone correction is imperfect, and a projection-domain detector can learn the residual without committing to any particular correction. This is presently a **hypothesis and experimental design**, not a demonstrated result: the §5 experiments removed generic, not artifact-selective, angles, and the robustness observed there tracked model capacity rather than architecture.

**6.5 The priority near-term experiment: dose / noise robustness.** One experiment is testable *now*, on synthetic data, and it unifies three threads — angular robustness (§5), dose, and the structured-noise argument (§6.4). Simulate dose reduction by **Poisson-thinning the projections**, then compare projection-domain detection against image-domain detection on the correspondingly noisier reconstructions, across a range of simulated dose levels. The hypothesis: at low dose, quantum mottle is Poisson and roughly independent per detector reading in the raw data, whereas reconstruction *correlates* and redistributes it into streak and blotch — so a projection-domain reader should retain detection sensitivity to a lower dose than the image-domain model. A confirmed result supports an ALARA argument in which the binding constraint on dose is diagnostic *confidence*: a reader robust to projection-domain noise could make lower-dose acquisitions clinically acceptable. This is the highest-value near-term test because, unlike the fidelity and artifact claims, it is at least partially demonstrable without real raw data.

**6.6 Further technical questions worth pursuing.**
- **Where the global trace assembles.** A first-layer kernel cannot see a full sinusoidal lesion trace; an effective-receptive-field / deeper-layer analysis should locate where the network integrates the trace across angle.
- **A reconstruction / self-supervised objective.** Classification never produced the literal FBP ramp; a reconstruction objective should, and would test whether the ramp is latent but unused.
- **Fixed-Gabor vs learned front end**, and an explicit autocorrelation feature (§3.3–3.4), as both initialization and interpretability probes.
- **Fair image-domain comparison under realistic artifacts.** The image-domain twin here was trained on clean synthetic reconstructions; a fair contest should expose it to realistic streak/metal/motion artifacts, which real clinical images contain and which may narrow — or, per §4, widen — the projection-domain advantage, especially infratentorially.
- **Back-projected saliency maps** to render projection-domain decisions in anatomic space for clinical interpretability.
- **The infratentorial-sensitivity test** (§4), requiring anatomic localization of bleeds beyond the subtype labels of the reference dataset.

**6.7 The eventual clinical form: a dual-domain ensemble, augmenting not replacing.** The projection-domain detector is best positioned to *augment* existing image-domain systems rather than displace them, and this is grounded in how state-of-the-art ICH detection is actually achieved. In the RSNA 2019 Brain CT Hemorrhage challenge (1,345 teams), **every** top solution reached its accuracy through an **ensemble of many models** — 7 to 31 per solution, 2-D CNNs paired with bidirectional GRU/LSTM sequence models over the slice axis, their per-model outputs blended — and ensembling improved performance for all of them (Wang et al. [15], the published first-place solution). On this exact task, combining complementary models is the established route to top performance. The natural clinical instantiation of the present work is therefore a **two-domain ensemble**: from the *same* acquisition, run one detector on the raw projections and one on the reconstructed image, then combine — or cross-check — their outputs (the "two semi-independent readers" of §4, point 5). The domains are complementary in *error mode*: the image-domain model can be misled by reconstruction artifacts the raw-domain model never sees, each carries different inductive biases and failure cases, and an adversarial perturbation would have to survive into *both* a raw sinogram and its reconstruction to fool the pair — so agreement raises confidence while disagreement flags a case for human review. One caveat bounds the claim honestly: by the data-processing inequality (§2.5), the raw data carry at least as much diagnostic information as the reconstruction, so a sufficiently powerful raw-domain model would in principle *subsume* the image-domain one. The ensemble's value is thus a **finite-data, finite-capacity complementarity** — real, and per the RSNA evidence reliably exploitable — and, pragmatically, a low-risk adoption path that adds a raw-data reader alongside the image-domain systems already in clinical use. It also defines a clean prospective endpoint once real raw data exist: test whether the raw + image ensemble exceeds the image-domain detector alone — the operational measure of whether the projection domain contributes clinically usable information.

---

## 7. Conclusion

Reconstruction is a lossy, vendor-specific, artifact-introducing inverse standing between the measurement and the diagnosis. The information-theoretic argument, the autocorrelation/Gabor feature basis, and — now — preliminary results showing image-domain-competitive ICH detection from synthetic sinograms together make a coherent case that **diagnosis is better posed on the raw projection data than on the reconstructed image**. The representation question is answered; the remaining advantages are properties of real measured data, reachable only with fan-beam/helical raw acquisitions and clinically-acquired cohorts at academic scale. The purpose of this document is to put that case, and that remaining program, on the record clearly enough for a suitably resourced investigator to take it up.

---

## References

*Classical citations are standard references from the canonical literature; page-level details should be verified against the originals. Deep-learning and dataset citations were confirmed against publisher / PubMed / archive records.*

1. Radon J. "Über die Bestimmung von Funktionen durch ihre Integralwerte längs gewisser Mannigfaltigkeiten." *Ber. Sächs. Akad. Wiss. Leipzig, Math.-Phys. Kl.* 1917;69:262–277.
2. Bracewell RN. "Strip integration in radio astronomy." *Aust. J. Phys.* 1956;9:198–217.
3. Gabor D. "Theory of communication." *J. Inst. Electr. Eng.* 1946;93(26):429–457.
4. Cormack AM. "Representation of a function by its line integrals, with some radiological applications." *J. Appl. Phys.* 1963;34:2722–2727.
5. Hounsfield GN. "Computerized transverse axial scanning (tomography): Part 1. Description of system." *Br. J. Radiol.* 1973;46:1016–1022.
6. Kak AC, Slaney M. *Principles of Computerized Tomographic Imaging.* IEEE Press, 1988.
7. Kalender WA, Seissler W, Klotz E, Vock P. "Spiral volumetric CT with single-breath-hold technique, continuous transport, and continuous scanner rotation." *Radiology* 1990;176(1):181–183.
8. Katsevich A. "Theoretically exact filtered backprojection-type inversion algorithm for spiral CT." *SIAM J. Appl. Math.* 2002;62(6):2012–2026.
9. Zhu B, Liu JZ, Cauley SF, Rosen BR, Rosen MS. "Image reconstruction by domain-transform manifold learning (AUTOMAP)." *Nature* 2018;555:487–492. doi:10.1038/nature25988
10. Li Y, Li K, Zhang C, Montoya J, Chen G-H. "Learning to Reconstruct CT Images Directly from Sinogram Data under a Variety of Data Acquisition Conditions." *IEEE Trans. Med. Imaging* 2019;38(10):2469–2481. doi:10.1109/TMI.2019.2910760
11. Zhang C, Li Y, Chen G-H. "Accurate and robust sparse-view angle CT image reconstruction using deep learning and prior image constrained compressed sensing (DL-PICCS)." *Med. Phys.* 2021;48(10):5765–5781. doi:10.1002/mp.15183
12. Xie H, Shan H, Wang G. "Deep Encoder-Decoder Adversarial Reconstruction (DEAR) Network for 3D CT from Few-View Data." *Bioengineering* 2019;6(4):111. doi:10.3390/bioengineering6040111
13. Flanders AE, Prevedello LM, Shih G, et al. "Construction of a Machine Learning Dataset through Collaboration: The RSNA 2019 Brain CT Hemorrhage Challenge." *Radiol. Artif. Intell.* 2020;2(3):e190211. doi:10.1148/ryai.2020190211
14. Moen TR, Chen B, Holmes DR III, Duan X, Yu Z, Yu L, Leng S, Fletcher JG, McCollough CH. "Low-dose CT image and projection dataset." *Med. Phys.* 2021;48(2):902–911. doi:10.1002/mp.14594 *(Mayo Clinic / AAPM LDCT-and-Projection-data; real DICOM-CT-PD helical projection data + paired reconstructions, The Cancer Imaging Archive.)*
15. Wang X, Shen T, Yang S, Lan J, Xu Y, Wang M, Zhang J, Han X. "A deep learning algorithm for automatic detection and classification of acute intracranial hemorrhages in head CT scans." *NeuroImage: Clinical* 2021;32:102785. doi:10.1016/j.nicl.2021.102785 *(Published first-place solution to the RSNA 2019 Kaggle challenge; every top team used ensembles of 7–31 CNN + bidirectional-GRU/LSTM sequence models.)*
