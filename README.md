# Raw CT Data Research Proposal

**Deep Learning on Raw CT Projection Data — rationale, feasibility evidence, and directions for projection-domain diagnosis.**

Author: John Bramble, MD — in collaboration with Claude (Anthropic Fable 5)

This repository holds a technical research proposal (not a funding application) arguing
that automated diagnosis from CT is better posed on the **raw projection (sinogram)
data** than on the reconstructed image — because reconstruction is a lossy,
vendor-specific, artifact-introducing inverse, and by the data-processing inequality the
projections carry at least as much diagnostic information as any image computed from
them. The exemplar task is detection of acute intracranial hemorrhage (ICH).

## Read it

- **[Raw_CT_Data_Research_Proposal.pdf](Raw_CT_Data_Research_Proposal.pdf)** — the
  formatted document (equations and figures render correctly here).
- `Raw_CT_Data_Research_Proposal.md` — the source (custom `@@`-equation and
  `![...]{scale}` figure syntax; the PDF is the readable version).

## What it contains

1. **Thesis** — why the projection domain, stated information-theoretically.
2. **Mathematical basis** — fan-beam/helical geometry, the parallel-beam Radon
   idealization, the Fourier-slice / Wiener–Khinchin identity by which *the sinogram
   encodes the image autocorrelation* (Figure 1), the Gabor feature basis, and the
   data-processing-inequality argument (Figure 2).
3. **Feasibility results** (on synthetic parallel-beam sinograms from the RSNA 2019 ICH
   dataset): projection-domain ICH detection at **macro-AUC 0.946**, on par with a
   matched image-domain twin; an interpretable learned-kernel domain signature; a
   sinogram-native architecture with a statistically robust gain; and a controlled,
   *conditional* angular-robustness result.
4. **Directions** — extension to real fan-beam/helical geometry; reconstruction-fidelity
   validation against the Mayo Clinic / AAPM measured-projection dataset; the data-scale
   problem (≈200k studies, prospective raw-data retention, single-center vs multi-center
   accrual); and open technical questions.

## Honest scope

The feasibility experiments use **synthetic parallel-beam** re-projections of
reconstructed images, which rigorously test *representation and architecture* but cannot
demonstrate the raw-data fidelity/artifact advantages. Those require true measured
fan-beam helical projections and clinically-acquired data at academic scale — the work
this proposal sets out for a suitably resourced investigator.

## Public measured-projection data: the Mayo Clinic / AAPM dataset

The real measured projection data needed to extend this work (the Directions above) is
publicly available as the **Low Dose CT Image and Projection Data
(LDCT-and-Projection-data)** collection from Mayo Clinic on The Cancer Imaging Archive
(TCIA), sponsored by the AAPM and NIH/NIBIB:

- **<https://www.cancerimagingarchive.net/collection/ldct-and-projection-data/>**
  — DOI [10.7937/9npb-2637](https://doi.org/10.7937/9npb-2637); described in Moen et al.,
  "Low-dose CT image and projection dataset," *Medical Physics* 48(2), 2021
  ([10.1002/mp.14594](https://doi.org/10.1002/mp.14594)).

**What the archive contains — 299 patients:** 99 head / neuro (case IDs `N###`), 100 chest
(`C###`), and 100 abdomen (`L###`). For each patient it provides three components:

1. **Measured raw projection data** in the open **DICOM-CT-PD** format — an extended DICOM
   that stores the *acquired* attenuation/projection (sinogram) values together with the
   full acquisition geometry in the header. This is the projection-domain data this
   proposal argues for, as measured rather than re-projected.
2. **Reconstructed DICOM images** (all via filtered back-projection).
3. **Excel clinical reports** — demographics, diagnostic/pathology annotations, measurements.

Each patient has **two projection sets**: a full routine-dose acquisition and a
**simulated reduced-dose** version produced by inserting noise into the full-dose
projections (head and abdomen at 25 % of routine dose, chest at 10 %). Acquisitions are
**helical**, on **two vendors' scanners** — roughly 150 cases on a Siemens SOMATOM
Definition Flash and 149 on a GE Lightspeed VCT — which makes the collection directly
useful for probing the *vendor-specific* nature of reconstruction. It grew out of the 2016
AAPM / Mayo Low Dose CT Grand Challenge.

**Fit and limits for this program.** The 99 head cases supply real measured fan-beam /
helical neuro projections — the concrete way to test whether the representation and
architecture survive real geometry, which the synthetic parallel-beam experiments here
cannot. The two-vendor split directly exercises the vendor-specific-reconstruction
argument. Two limits worth stating: at 299 patients (99 head) this is a **fidelity and
geometry validation resource, not a training set at the ≈200k-study scale** the full
program needs; and it was assembled for low-dose reconstruction research, so it carries
general pathology annotations rather than a curated ICH ground truth — hemorrhage labels
on the head subset would need to be established separately.

**Use this collection to *verify reconstruction fidelity* — not to train an ICH detector.**
Its 99 head studies hold far too few hemorrhages, and no ICH labels, to source a model
dedicated to hemorrhage detection. For scale: the feasibility study here trained on a
200,000-slice RSNA-derived cohort held at the dataset's base prevalence of **14.5 %
positive** (≈ 28,000 positive slices). (A separate *patient-decorrelated* subset — built to
break the ~30-correlated-slices-per-patient structure of the full ~750k-image set by keeping
only a few independent slices per patient, hemorrhage slices first — runs at a higher ~34 %
positive fraction by construction; the two figures describe different cohorts.)

## Figures

`schematic_autocorrelation.png` (Figure 1) and `schematic_sinogram_motion.png` (Figure 2)
are generated by `make_autocorr_schematic.py` and `make_schematic.py`.

## Build the PDF

```
python md_to_pdf.py Raw_CT_Data_Research_Proposal.md Raw_CT_Data_Research_Proposal.pdf
```
Requires `reportlab` and DejaVu fonts (for Greek/math glyphs).
