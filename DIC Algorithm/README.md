# 2D DIC strain analysis for video

Subset-based 2D digital image correlation (DIC) in Python. It tracks a speckle pattern through a video or an image sequence and measures full-field displacement and strain.

This README covers the practical steps. For the underlying mathematics, see [THEORY.md](THEORY.md).

---

## 1. Installation

You need Python 3.10 or newer. From the project folder:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

To check the installation, run the tests. All of them should pass.

```bash
.venv/bin/python -m pytest -q tests
```

Every command below assumes your terminal is in the project folder:

```bash
cd "/Users/zacharycorbin-cheah/Documents/DIC Algorithm"
```

---

## 2. Workflow overview

1. **Prepare and film the specimen** (section 3).
2. **Calibrate the pixel size**, i.e. how many mm one pixel represents (section 4).
3. **Choose the region of interest (ROI)**, and optionally a mask (section 5).
4. **Run the analysis** (section 6).
5. **Check data quality** before trusting any numbers (section 8).
6. **Compare with reference measurements** (section 9).

---

## 3. Preparing and filming the specimen

The quality of the footage limits the accuracy more than anything in the software.

**Speckle pattern**
- Use random, high-contrast dots on a matte background, such as black dots on white paint.
- In the *video*, each dot should be about **3–5 pixels** across, and dots should cover roughly half the surface.
- Avoid regular patterns like grids or stripes. They let a subset match in more than one place.
- The pattern must stretch with the surface. Paint works well. Stickers and loose tape do not.

**Camera**
- Mount the camera on a tripod and point it **square-on (perpendicular) to the surface** being measured.
- **The deformation must happen in the plane of the image.** Movement towards or away from the camera makes the speckles look bigger or smaller, which shows up as false strain of roughly −Δz/Z. Here Δz is how far the surface moves towards or away from the camera, and Z is the distance from camera to surface. For example, 1 mm of movement at 500 mm distance gives about 0.2% of false strain. A longer distance with more zoom reduces this.
- Use manual focus, fixed exposure and fixed white balance. Turn off image stabilisation.
- Use a shutter speed fast enough that nothing blurs.
- Use even, diffuse lighting with no glare or moving shadows.
- Use the highest-quality recording mode. Better still, record an image sequence (PNG/TIFF) rather than compressed video.

**Test procedure**
- Record a few seconds **before loading starts**. The first processed frame is the undeformed reference.
- Keep hands, strings, clamps and loads out of the region you want to measure.
- Note anything that will help line up the timing with reference data later, such as when loading starts or when weights are added.

---

## 4. Pixel-size calibration

Strain has no units, so it doesn't need calibration. Displacements are measured in pixels unless you supply a scale.

1. Put something of known length in the plane of the specimen, such as a ruler, or use a known specimen dimension.
2. Measure its length in pixels in the first frame. Any image viewer that shows cursor coordinates will do.
3. Divide: `pixel_size = length_mm / length_px`.
4. Pass the result as `--pixel-size <value> --units mm`.

---

## 5. Region of interest and masks

**Rectangular ROI**
- `--select-roi` opens the first frame. Drag a box and press **Enter**. The terminal prints the ROI it used, e.g. `ROI=(112, 48, 390, 260)`.
- `--roi X Y W H` reuses that box, so repeat runs on the same setup stay consistent.

**Mask** (for specimens that are narrow, tilted or irregular)

A rectangle can include background, which stays still while the specimen moves and spoils the match. A mask fixes this.

- A mask is a black-and-white image the same size as the video frame. Grid points are kept only on **white** pixels.
- Points are tested at their *centres*. For every subset to sit fully on the specimen, shrink (erode) the white region by half the subset size.
- Black out anything that covers the surface, such as strings, clamps or markings.

Here is an example of building a mask from colour, as was done for the sample stick video. It assumes a light specimen against a darker background. Adjust the thresholds for your own footage.

```python
import cv2, numpy as np
frame = cv2.VideoCapture("video.mp4").read()[1]
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
m = ((hsv[..., 2] > 130) & (hsv[..., 1] > 40)).astype(np.uint8) * 255   # bright, slightly coloured
m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((41, 41), np.uint8))   # fill in the dark speckles
m[465:540] = 0                                                         # remove an obstruction
subset = 41
m = cv2.erode(m, np.ones((subset + 4, subset + 4), np.uint8), borderValue=255)
cv2.imwrite("mask.png", m)
```

You can use a mask and a ROI together. The ROI also keeps the run fast because it limits how many grid points are created.

---

## 6. Running the analysis

```bash
.venv/bin/python run_dic.py <video-or-folder> [options]
```

The input can be a video file (`.mp4`, `.mov`, …) or a folder of images. Images are sorted by filename, so number them with leading zeros.

**Typical commands**

```bash
# First run: draw the ROI and calibrate displacements to mm
.venv/bin/python run_dic.py test.mp4 --select-roi --pixel-size 0.02 --units mm

# Repeat run with a fixed ROI, comparing against an extensometer (engineering strain)
.venv/bin/python run_dic.py test.mp4 --roi 100 50 400 300 --pixel-size 0.02 --units mm --strain-kind engineering

# Quick first look at a long video: every 5th frame, no overlay video
.venv/bin/python run_dic.py test.mp4 --roi 100 50 400 300 --every 5 --no-video

# Narrow specimen with a mask and large speckles
.venv/bin/python run_dic.py test.mp4 --roi 880 0 140 820 --mask mask.png --subset 41 --step 8 --search 40
```

### Option reference

| Option | Default | Meaning |
|---|---|---|
| `-o, --out` | `<input>_dic/` | Output folder |
| `--roi X Y W H` | whole frame | Rectangular region of interest, in pixels |
| `--select-roi` | – | Draw the ROI on the first frame |
| `--mask FILE` | – | Black-and-white image; grid points on black pixels are ignored |
| `--start`, `--stop` | 0, end | Frame range to process. `--start` is the reference (undeformed) frame. |
| `--every N` | 1 | Process every N-th frame |
| `--subset` | 31 | Subset size in pixels (odd). Should contain at least 3 speckles in each direction. |
| `--step` | 10 | Spacing between grid points in pixels |
| `--search` | 30 | Search radius in pixels for the initial whole-pixel match. Increase it if the specimen moves a long way between processed frames. |
| `--reference` | `auto` | `auto`: update the reference frame only when matching degrades. `first`: always compare with the first frame. `incremental`: compare with the previous frame. |
| `--update-threshold` | 0.9 | Mean ZNCC below which `auto` updates the reference |
| `--zncc-threshold` | 0.7 | Points with a worse match than this are rejected (set to NaN) |
| `--prefilter σ` | 0 | Gaussian smoothing of each frame, in pixels. 0.5–1 helps with noisy or compressed video. |
| `--strain-window` | 5 | Size of the strain fitting window, in grid points per side (odd) |
| `--strain-kind` | `green-lagrange` | Or `engineering` (small-strain). Match this to your reference measurement. |
| `--pixel-size`, `--units` | 1, px | Scale for displacements |
| `--component` | `exx` | Field shown in the overlay video: `exx eyy exy e1 e2 von_mises u v zncc` |
| `--clim MIN MAX` | auto | Fixed colour limits for the overlay video |
| `--no-video` | – | Skip writing the overlay video (faster) |

**Run time.** Processing takes about 0.3–0.6 s per frame on a laptop for 1080p footage and a few hundred to a few thousand points. Decoding the video and interpolating the images take most of that time.

---

## 7. Outputs

Everything is written to the output folder:

| File | Contents |
|---|---|
| `strain_summary.csv` | One row per frame: frame number, time, mean displacement, ROI-average `mean_exx/eyy/exy/e1/e2/von_mises`, `max_e1`, `max_von_mises`, `mean_zncc` and `valid_fraction`. It opens directly in Excel. |
| `strain_history.png` | Left panel: mean strain over time. Right panel: correlation quality and the share of valid points. |
| `fields_frameN.png` | Maps of u, v, εxx, εyy, εxy and εvM on the last frame, drawn over the deformed image |
| `overlay_<component>.mp4` | The chosen field painted onto every frame at the tracked positions |
| `results.npz` | Full data for every frame: `u`, `v`, `zncc` and every strain component, each shaped `(frames, ny, nx)`, plus `grid_x`, `grid_y`, `frames`, `times` and the settings |

**Sign and axis conventions**
- x is to the right and y is **down** (image convention).
- `exx` is horizontal strain in the image and `eyy` is vertical. If the specimen is loaded vertically in the frame, the loading-direction strain is `eyy`.
- Displacements are measured from the first processed frame, at the material points of the original grid (a Lagrangian description).
- `exy` is the tensor shear strain. Engineering shear strain γ is `2·exy`.

**Working with results in Python**

```python
from dic import DICResult, DICSettings, FrameSource, run_dic
import numpy as np

res = DICResult.load("test_dic/results.npz")
s = res.strain(-1)                      # last frame: s.exx, s.eyy, s.exy, s.e1, s.e2, s.von_mises
# strain at the grid point nearest to pixel (x=420, y=310), over time
i = np.abs(res.grid_y - 310).argmin(); j = np.abs(res.grid_x - 420).argmin()
eyy_t = res.strain_history()["eyy"][:, i, j]

# re-run from Python with custom settings
res = run_dic(FrameSource("test.mp4", start=0, step=2),
              DICSettings(subset_size=31, step=8, pixel_size=0.02, units="mm"),
              roi=(100, 50, 400, 300))
```

To recompute strain with a different window or strain type, call `res.strain(k, window=7, kind="engineering")`. The correlation doesn't need to run again.

---

## 8. Checking data quality

Check these before using any numbers:

1. **`strain_history.png`, right panel.** Mean ZNCC should stay above about 0.95, and valid points should stay near 100%. Short dips usually mean something blocked the view or there was motion blur.
2. **`overlay_*.mp4`.** The coloured region should stay attached to the specimen for the whole video.
3. **Strain before loading.** Frames before any load should read near zero. The scatter in those frames is your *noise floor*. Strains smaller than about 3× that scatter can't be told apart from noise.
4. **Physical sense.** Equal negative strain in every direction (εxx ≈ εyy < 0), together with points drifting towards the image centre, means the surface moved **away from the camera**, not that the material shrank. The positive equivalent means it moved towards the camera.

**Troubleshooting**

| Symptom | Likely cause | Fix |
|---|---|---|
| Many invalid points from the first frame | Subset too small for the speckle size, or a featureless area | Increase `--subset`; check that the ROI or mask covers only speckled surface |
| Points lost during fast motion | Moves more than `--search` px between processed frames | Use `--every 1`, or increase `--search` |
| Noisy strain maps | Video compression or sensor noise | Use `--prefilter 0.7`, a larger `--subset` or `--strain-window`, and better footage |
| Strain peaks look too low | Too much smoothing | Use a smaller `--subset`, `--step` or `--strain-window` |
| Uniform false strain that follows motion | Movement towards or away from the camera, or camera movement | Change the setup (section 3). 2D DIC can't correct for this. |
| Matching degrades then suddenly recovers | `auto` reference update happened | Normal behaviour. The frames are listed under "Reference frames used". |

---

## 9. Comparing with reference measurements

- **Extensometer.** Set the ROI over the extensometer's gauge length and compare `mean_e**` from the CSV in the loading direction. Use `--strain-kind engineering`.
- **Strain gauge.** Read the strain at the gauge location from `results.npz` (section 7). The DIC value is an average over the *virtual strain gauge*, about `(strain_window − 1)·step + subset` pixels. Make that comparable to the physical gauge length.
- **Timing.** `time_s` is `frame / fps`, counted from the start of the video. Shift it to line up with one clear event shared by both data sets, such as the start of loading or failure.
- **Expected agreement.** With good footage, DIC typically matches extensometers to within about 100–500 µε, plus a few percent of the reading. Hand-made patterns and compressed video do worse.

---

## 10. Project layout

```
run_dic.py                 command-line tool
dic/
  correlation.py           subset matching: whole-pixel search and sub-pixel IC-GN refinement
  pipeline.py              frame-by-frame tracking, reference updating, results container
  strain.py                displacement-gradient fitting and strain tensors
  video.py                 reading videos and image folders, interactive ROI selection
  visualize.py             figures, overlay video, CSV summary
  synthetic.py             synthetic speckle images with known deformation
examples/
  synthetic_tensile_demo.py   end-to-end validation against an analytic necking field
tests/test_dic.py          automated accuracy tests
THEORY.md                  mathematical background
```

---

## 11. Validation

`examples/synthetic_tensile_demo.py` creates a speckle video with an exactly known necking field (up to about 28% strain), runs the full pipeline and reports the error against the true values. `pytest tests` checks rigid translation, uniform strain, strain-fit accuracy, large-strain kinematics and the full pipeline with saving and reloading.

On synthetic 8-bit images, the displacement error is about 0.002 px and the uniform-strain error is about 40 µε. Expect real footage to be worse, mainly because of compression, lighting and out-of-plane movement.

**Sample footage (`DIC Videos (RAW)/0Z8A8243.MP4`).** Tracking was excellent (ZNCC ≈ 0.998), but the stick bent away from the camera. That produced up to about −10% of false strain in every direction, so the result is not material strain. It's a useful example of the out-of-plane limitation described in sections 3 and 8.
