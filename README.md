# Mangrove Stress and Strain Analysis

Measuring how mangroves deform under environmental stress, using low-cost video and digital image correlation (DIC).

## Abstract

Mangroves are an essential part of South Florida ecology, yet their structural properties, like strain, are largely unmeasured. We developed a cost effective materials science approach to measuring mangrove strain from environmental stressors, based on a previous study using walnut trees. We analysed footage of real mangroves in nature painted with random speckle patterns using digital image correlation (DIC), a non-contact, full-field optical technique that measures surface displacement and strain by tracking the speckle pattern between a reference frame and the deformed frame(s). We developed a DIC algorithm that works by dividing the region of interest into subsets, which are then located in subsequent frames by optimising a correlation criterion over a model of local deformation. The algorithm forms an initial integer-pixel match by normalising frame cross-correlation, and then refines to sub-pixel accuracy using the inverse compositional Gauss-Newton algorithm, which minimises the zero-normalised sum of squared differences under a first order (affine) shape function. For continuous video sequences, we track subsets across frames as Lagrangian material points. Strain is derived from the resulting displacement field by local least-squares plane fitting and is reported as the Green-Lagrange finite strain tensor. Applying this technique to mangroves, while unconventional, could give ecologists a new way to measure how mangroves respond to physical stress.

## How it works

1. **Prepare the tree.** Paint a random, high-contrast speckle pattern on the bark or stem.
2. **Film it.** Record video with a fixed camera while the tree is under load, for example from wind, tides or applied weights.
3. **Track the pattern.** The DIC algorithm divides the painted region into small subsets and finds each one again in every frame to sub-pixel accuracy.
4. **Compute strain.** Displacement fields are turned into strain maps and time histories: Green-Lagrange strain, principal strains and von Mises equivalent strain.

| Stage | Method |
|---|---|
| Whole-pixel matching | Normalised cross-correlation |
| Sub-pixel refinement | Inverse-compositional Gauss–Newton (IC-GN), ZNSSD criterion, first-order shape function |
| Interpolation | Bicubic B-spline |
| Video tracking | Lagrangian material points with automatic reference-frame updating |
| Strain | Local least-squares plane fit → Green-Lagrange strain tensor |

On synthetic images with a known exact answer, the algorithm measures displacement to about **0.002 px** and strain to about **40 µε**.

## Quick start

```bash
cd "DIC Algorithm"
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python run_dic.py path/to/video.mp4 --select-roi --pixel-size 0.02 --units mm
```

The results (strain maps, a strain-history CSV and an overlay video) are written to `<video>_dic/`.

## Documentation

| Document | Contents |
|---|---|
| [DIC Algorithm/README.md](DIC%20Algorithm/README.md) | Practical guide: filming, calibration, ROI and masks, all command options, outputs, quality checks, troubleshooting, comparing with reference data |
| [DIC Algorithm/THEORY.md](DIC%20Algorithm/THEORY.md) | Mathematical principles: correlation criterion, IC-GN, interpolation, tracking, strain measures, error sources, references |
| [DIC_Capture_Checklist.docx](DIC_Capture_Checklist.docx) | Checklist for capturing DIC footage |

## Repository layout

```
DIC Algorithm/
  run_dic.py                  command-line tool
  dic/                        the DIC package (correlation, tracking, strain, video, plotting)
  tests/                      automated accuracy tests
  examples/                   synthetic tensile-test validation demo and its output
  DIC Videos (RAW)/           homemade pilot test: photos and analysis results
DIC_Capture_Checklist.docx
```

## Status

- **Algorithm:** complete, and validated on synthetic speckle footage with known strain.
- **Pilot test:** footage of a speckled wooden stick under step loading, filmed primitively. Tracking was excellent (ZNCC ≈ 0.998), and every loading step was picked up. The stick bent *away from* the camera, though, so the measured values are dominated by false strain from out-of-plane motion. This is a known limit of 2D DIC, and it is why camera placement matters (see the filming guidance in the practical guide).
- **Next:** analysis of field footage of painted mangroves, with comparison against reference measurements.

## Data

The raw video files are not stored in this repository because they exceed GitHub's 100 MB per-file limit. The processed results for the pilot test are included.

## Limitations

The method measures 2D, in-plane strain, so the camera should face the measured surface square-on. Movement towards or away from the camera appears as false strain of roughly −Δz/Z (movement ÷ camera distance). This matters for curved, swaying trunks and branches in the field. A longer camera distance with zoom reduces the effect. Stereo (3D) DIC removes it.
