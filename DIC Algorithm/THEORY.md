# Mathematical principles of the 2D DIC algorithm

## Abstract

Digital image correlation (DIC) is a non-contact optical method that measures full-field surface displacement and strain from images of a randomly speckled specimen. This implementation uses the *local, subset-based* approach. The reference image is divided into small square subsets, and each subset is found again in later images by optimising a correlation criterion over a parametric model of local deformation.

The pixel-level match comes from normalised cross-correlation. Sub-pixel accuracy comes from the inverse-compositional Gauss–Newton (IC-GN) algorithm, which minimises the zero-normalised sum of squared differences (ZNSSD) under a first-order (affine) shape function. Grey levels between pixels come from bicubic B-spline interpolation.

For video, subsets are tracked through time as Lagrangian material points. The reference image is updated automatically when the deformation becomes too large to match against the original. Strain is computed from the resulting displacement field by local least-squares plane fitting and reported as the Green–Lagrange (finite) strain tensor, its principal values and an equivalent von Mises strain.

On synthetic images with a known exact answer, the method reaches displacement errors of about 0.002 px and strain errors of about 4×10⁻⁵. Its main physical limitation is that it is a 2D method, which assumes the surface stays in a plane parallel to the camera sensor.

---

## 1. The measurement principle

Suppose an image $f(\mathbf{x})$ is taken of a speckled surface before deformation, and an image $g(\mathbf{x})$ after it. DIC rests on the assumption that the grey-level pattern is **carried along with the material**. If a material point at $\mathbf{X}$ moves to $\mathbf{x} = \mathbf{X} + \mathbf{u}(\mathbf{X})$, then

$$
g\big(\mathbf{X} + \mathbf{u}(\mathbf{X})\big) \;\approx\; a\, f(\mathbf{X}) + b ,
$$

where $a$ and $b$ allow for changes in lighting (contrast and brightness). The goal is to recover the displacement field $\mathbf{u}(\mathbf{X}) = (u, v)$ from these two images.

A single pixel contains far too little information to identify its own motion; this is known as the *aperture problem*. DIC therefore assumes the displacement is smooth over a small neighbourhood, the **subset**, of $(2M+1)\times(2M+1)$ pixels. It then solves for the motion of the subset as a whole. A random, isotropic speckle pattern makes each subset's grey-level signature unique, so it can only match in one place.

---

## 2. Subset kinematics: the shape function

Inside a subset centred at $\mathbf{x}_0$, a point at local offset $(\Delta x, \Delta y)$ is mapped by a **first-order (affine) shape function** with parameter vector $\mathbf{p} = (u, u_x, u_y, v, v_x, v_y)^\mathsf{T}$:

$$
\mathbf{W}(\Delta\mathbf{x};\mathbf{p}) =
\begin{bmatrix} 1+u_x & u_y & u \\ v_x & 1+v_y & v \\ 0 & 0 & 1 \end{bmatrix}
\begin{bmatrix} \Delta x \\ \Delta y \\ 1 \end{bmatrix}.
$$

- $u, v$ is the translation of the subset centre.
- $u_x = \partial u/\partial x$, $u_y$, $v_x$, $v_y$ are the local displacement gradients, i.e. stretch, shear and rotation.

This is a first-order Taylor expansion of the displacement field about the subset centre. Writing it as a $3\times3$ homogeneous matrix means warps form a group: they can be composed by matrix multiplication and inverted by matrix inversion. The IC-GN update in §4 relies on this.

A first-order shape function is exact for homogeneous deformation. When the strain varies sharply within a subset, it causes a small *undermatched* bias, because subset-scale curvature is ignored. This is why the subset should be small compared with any strain gradients of interest.

---

## 3. The correlation criterion

How well a subset matches is measured by the **zero-normalised sum of squared differences**:

$$
C_\text{ZNSSD}(\mathbf{p}) = \sum_{\Delta\mathbf{x}\in S}
\left[ \frac{f(\mathbf{x}_0+\Delta\mathbf{x}) - \bar f}{\Delta f}
     - \frac{g\big(\mathbf{x}_0+\mathbf{W}(\Delta\mathbf{x};\mathbf{p})\big) - \bar g}{\Delta g} \right]^2 ,
$$

where $\bar f$ and $\bar g$ are the subset means, and $\Delta f = \sqrt{\sum (f-\bar f)^2}$ (and likewise $\Delta g$) are the subset grey-level norms. Subtracting the mean removes any brightness offset $b$, and dividing by the norm removes any contrast scale $a$. The criterion is therefore unaffected by uniform lighting changes, which is important for video.

It is directly related to the **zero-normalised cross-correlation (ZNCC)**, which is the quality score the software reports:

$$
C_\text{ZNCC} = 1 - \tfrac12\, C_\text{ZNSSD} \in [-1, 1],
$$

where 1 is a perfect match. Points with a ZNCC below a threshold (default 0.7) are rejected.

---

## 4. Finding the match: from pixel to sub-pixel

### 4.1 Initial guess

Iterative refinement only converges when it starts within a few pixels of the true answer. For the first frame, a whole-pixel estimate of $(u, v)$ is found by exhaustively evaluating normalised cross-correlation over a search window. OpenCV does this efficiently using Fourier transforms. For later frames, the solution from the previous frame is used as the starting guess. This is valid because motion between frames is small. Any point that fails is searched for again from scratch.

### 4.2 Inverse-compositional Gauss–Newton (IC-GN)

The sub-pixel solution minimises $C_\text{ZNSSD}$ over all six parameters. A conventional *forward-additive* Gauss–Newton solver linearises the *current* image $g$ around the current estimate. That means recomputing image gradients and a $6\times6$ matrix (the Hessian) on every iteration.

The **inverse-compositional** form (Baker & Matthews, 2004) swaps the roles of the two images. Each iteration finds a small update $\Delta\mathbf{p}$ applied to the *reference* subset, then composes its inverse with the current warp:

$$
\mathbf{W}(\cdot\,;\mathbf{p}) \;\leftarrow\; \mathbf{W}(\cdot\,;\mathbf{p}) \circ \mathbf{W}(\cdot\,;\Delta\mathbf{p})^{-1}.
$$

Linearising about $\Delta\mathbf{p}=\mathbf{0}$ gives a Jacobian that depends only on the reference image. For each subset pixel, the *steepest-descent image* is

$$
\mathbf{J} = \nabla f \,\frac{\partial \mathbf{W}}{\partial \mathbf{p}}
= \big[\, f_x,\; f_x\Delta x,\; f_x\Delta y,\; f_y,\; f_y\Delta x,\; f_y\Delta y \,\big],
$$

and the Gauss–Newton step is

$$
\Delta\mathbf{p} = -\mathbf{H}^{-1} \sum_{S} \mathbf{J}^\mathsf{T}
\left[ (f - \bar f) - \frac{\Delta f}{\Delta g}\,(g(\mathbf{W}) - \bar g) \right],
\qquad
\mathbf{H} = \sum_{S} \mathbf{J}^\mathsf{T}\mathbf{J}.
$$

$\mathbf{J}$, $\mathbf{H}$ and $\mathbf{H}^{-1}$ are computed **once per subset**, not once per iteration. The only per-iteration cost is sampling the deformed subset in $g$. The solution is mathematically equivalent to the forward-additive one but much faster (Pan et al., 2013). Iteration stops when the update is negligible:

$$
\sqrt{\Delta u^2 + \Delta v^2 + M^2\big(\Delta u_x^2 + \Delta u_y^2 + \Delta v_x^2 + \Delta v_y^2\big)} < 10^{-3}\ \text{px}.
$$

The factor $M$ (half the subset width) converts each gradient change into the displacement it causes at the subset edge. Convergence typically takes 3–5 iterations.

A subset is rejected as texture-less if its Hessian is ill-conditioned or its grey-level spread $\Delta f$ is near zero. Without enough texture, there is no unique match.

### 4.3 Sub-pixel interpolation

The warped coordinates $\mathbf{W}(\Delta\mathbf{x};\mathbf{p})$ fall between pixels, so $g$ must be interpolated. The choice of interpolant directly limits accuracy. Low-order schemes such as bilinear interpolation cause a periodic, *systematic* displacement error that depends on the sub-pixel position (Schreier et al., 2000). This implementation uses **bicubic B-spline interpolation**: the image is pre-filtered into spline coefficients once, after which values can be read anywhere. The reference gradients $f_x, f_y$ use a fourth-order central-difference stencil, $\tfrac{1}{12}[1, -8, 0, 8, -1]$.

All subsets are solved simultaneously as batched array operations. Each iteration therefore costs one interpolation call over all active subsets.

---

## 5. Tracking through a video

Each tracked point is a **material point**. Its position is fixed in the first (reference) frame at $\mathbf{X}$, and its displacement $\mathbf{U}(\mathbf{X}, t)$ is always reported relative to that frame. This is a *Lagrangian* description, so strain fields from different times describe the same material and can be compared directly.

How each frame is matched depends on the reference strategy:

| Strategy | Matched against | Advantage | Drawback |
|---|---|---|---|
| `first` | Frame 0 | Independent errors; no drift | Fails at large deformation, when the speckles no longer look alike |
| `incremental` | Previous frame | Handles any deformation | Errors add up frame after frame (random-walk drift) |
| `auto` | Frame 0, updated only when the mean ZNCC drops below a threshold | Large-deformation capable with only a few updates | – |

When the reference is updated at frame $k$, the current point positions $\mathbf{x}_k = \mathbf{X} + \mathbf{U}_k$ become the new subset centres. Later displacements are measured relative to frame $k$ and added to $\mathbf{U}_k$. Error accumulates only at the few update steps rather than at every frame. A first-order shape function absorbs moderate homogeneous strain on its own: about 28% strain was tracked against the original reference in the synthetic tests.

---

## 6. From displacement to strain

### 6.1 Smoothed displacement gradients

Strain depends on the *gradients* of displacement. Differentiating noisy data directly makes the noise much worse. Instead, the gradients at each grid point are estimated by fitting a plane by weighted least squares over an $N\times N$ window of neighbouring grid points (*pointwise least squares*; Pan et al., 2009):

$$
u(x, y) \approx a_0 + a_1\,(x - x_0) + a_2\,(y - y_0), \qquad
\frac{\partial u}{\partial x} = a_1,\quad \frac{\partial u}{\partial y} = a_2 ,
$$

and the same for $v$. The weights are 1 for valid points and 0 for rejected ones. The fit is evaluated for every point at once by *normalised convolution*: moment sums are computed by correlation and then each $3\times3$ normal-equation system is solved. Missing data therefore reduces the number of points in a fit without biasing it.

Every strain value averages over a **virtual strain gauge** of side length

$$
L_\text{VSG} \approx (N-1)\,s + 2M + 1,
$$

where $s$ is the grid step. This sets the trade-off between **spatial resolution** and **strain noise**. A larger window lowers noise roughly in proportion to $1/L_\text{VSG}$, but it flattens peaks and gradients narrower than $L_\text{VSG}$.

### 6.2 Strain measures

The displacement gradient $\nabla\mathbf{u}$ gives the deformation gradient $\mathbf{F} = \mathbf{I} + \nabla\mathbf{u}$. Two strain measures are available.

- **Green–Lagrange strain** (default; exact for large deformation and rotation):

$$
\mathbf{E} = \tfrac12\left(\mathbf{F}^\mathsf{T}\mathbf{F} - \mathbf{I}\right), \qquad
\begin{aligned}
E_{xx} &= u_x + \tfrac12(u_x^2 + v_x^2),\\
E_{yy} &= v_y + \tfrac12(u_y^2 + v_y^2),\\
E_{xy} &= \tfrac12(u_y + v_x + u_x u_y + v_x v_y).
\end{aligned}
$$

- **Infinitesimal (engineering) strain**, valid for small strains and small rotations: $\varepsilon_{xx} = u_x$, $\varepsilon_{yy} = v_y$, $\varepsilon_{xy} = \tfrac12(u_y + v_x)$. The two measures differ by about $\tfrac12 \varepsilon^2$. At 2% strain the difference is about 1% of the value, and it grows quickly beyond that.

Derived quantities:

- **Principal strains**, which are the largest and smallest stretches, in mutually perpendicular directions:

$$
\varepsilon_{1,2} = \frac{\varepsilon_{xx}+\varepsilon_{yy}}{2} \pm \sqrt{\left(\frac{\varepsilon_{xx}-\varepsilon_{yy}}{2}\right)^2 + \varepsilon_{xy}^2}
$$

- **Von Mises equivalent strain**, assuming incompressibility ($\varepsilon_3 = -(\varepsilon_1+\varepsilon_2)$):

$$
\varepsilon_\text{vM} = \frac{2}{\sqrt3}\sqrt{\varepsilon_1^2 + \varepsilon_2^2 + \varepsilon_1\varepsilon_2}
$$

---

## 7. Sources of error and limits of the 2D method

**Random error (noise).** Image noise of standard deviation $\sigma$ spreads into the displacement roughly as

$$
\sigma_u \approx \frac{\sqrt{2}\,\sigma}{\sqrt{\sum_S f_x^2}} ,
$$

so the error falls as subset size and speckle contrast (intensity gradient) increase. Strain noise is about the displacement noise divided by the virtual strain gauge length. Video compression adds noise that is not uniform and can be correlated between frames.

**Systematic error.** The main causes are:
- *Interpolation bias*, reduced by using B-splines (§4.3).
- *Shape-function undermatching* in strong strain gradients (§2).
- *Smoothing* of narrow strain peaks by the virtual strain gauge (§6.1).

**Out-of-plane motion.** This is the fundamental limitation of 2D DIC. A pinhole camera at distance $Z$ images a surface that moves by $\Delta z$ *towards or away from* the lens with magnification $\propto 1/Z$. That produces a false, equal-in-all-directions strain (Sutton et al., 2008):

$$
\varepsilon_\text{false} \approx -\frac{\Delta z}{Z},
$$

together with a false displacement directed towards or away from the image centre. It cannot be told apart from real strain using one camera. It is reduced by a longer working distance (a telephoto or telecentric lens) and avoided entirely by 3D stereo-DIC. The sample stick video showed this clearly: εxx ≈ εyy ≈ −5% to −12% as the stick bent away from the camera, with points converging towards the image centre.

**Other limits.** The pattern must stay continuous, so DIC cannot measure across cracks or where surface coating flakes off. Lens distortion causes false strain towards the image edges unless the camera is calibrated. Rolling-shutter cameras distort fast-moving objects.

---

## 8. Validation approach

Testing DIC on images produced by interpolating a reference bitmap is circular: the test images then contain the same interpolation error as the algorithm, which hides real bias. Here the synthetic speckle patterns are instead sums of **analytic Gaussian blobs**,
$I(\mathbf{X}) = \sum_k a_k\, e^{-|\mathbf{X}-\mathbf{X}_k|^2/r_k^2}$.
A deformed image is rendered exactly by evaluating $I$ at $\mathbf{X} = \boldsymbol{\varphi}^{-1}(\mathbf{x})$, where the inverse map is found by fixed-point iteration. Noise and 8-bit quantisation are then added. Results are compared against the exact displacement and strain fields.

---

## References

1. Sutton, M. A., Orteu, J.-J., & Schreier, H. W. (2009). *Image Correlation for Shape, Motion and Deformation Measurements*. Springer.
2. Pan, B., Qian, K., Xie, H., & Asundi, A. (2009). Two-dimensional digital image correlation for in-plane displacement and strain measurement: a review. *Measurement Science and Technology*, 20, 062001.
3. Pan, B., Li, K., & Tong, W. (2013). Fast, robust and accurate digital image correlation calculation without redundant computations. *Experimental Mechanics*, 53, 1277–1289.
4. Baker, S., & Matthews, I. (2004). Lucas-Kanade 20 years on: a unifying framework. *International Journal of Computer Vision*, 56(3), 221–255.
5. Pan, B., Asundi, A., Xie, H., & Gao, J. (2009). Digital image correlation using iterative least squares and pointwise least squares for displacement field and strain field measurements. *Optics and Lasers in Engineering*, 47, 865–874.
6. Schreier, H. W., Braasch, J. R., & Sutton, M. A. (2000). Systematic errors in digital image correlation caused by intensity interpolation. *Optical Engineering*, 39(11), 2915–2921.
7. Sutton, M. A., Yan, J. H., Tiwari, V., Schreier, H. W., & Orteu, J.-J. (2008). The effect of out-of-plane motion on 2D and 3D digital image correlation measurements. *Optics and Lasers in Engineering*, 46, 746–757.
8. International Digital Image Correlation Society (2018). *A Good Practices Guide for Digital Image Correlation*. Jones, E. M. C., & Iadicola, M. A. (Eds.).
