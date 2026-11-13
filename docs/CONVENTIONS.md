# Lumina conventions (contract version 1)

This file is the contract between `training/` (which makes the labels and the model) and `web/`
(which renders with the model's output). A wrong convention does not crash: it renders a plausible
object lit from the wrong side. So every rule here has a test, and both implementations are checked
against the same vectors in `shared/`.

Change a rule only together with: both implementations, `shared/` (regenerate with
`uv run python -m lumina_train.vectors`), this file, and `CONTRACT_VERSION` in
`training/lumina_train/model_contract.py` and `web/src/lighting/modelContract.ts` if the model's
input or output meaning changes.

## 1. Frame

All lighting is expressed in **camera space** for the current video frame: right-handed, **+X
right, +Y up, -Z forward** (the three.js camera default). In version 1 the AR scene's world frame
_is_ the camera frame (camera at the origin with identity rotation), so an estimate goes to three.js
unchanged.

```
                 +Y (up)
                  |
                  |      -Z (forward: what the camera sees)
                  |     /
                  |    /
                  |   /
                  |  /
                  | /
     camera  ---- o ------------> +X (right, as seen in the un-mirrored image)
                 /
                /
               /
             +Z (behind the camera, toward the user of a rear camera)
```

A light "behind the camera" (direction +Z) lights the side of an object that the camera sees. A light
in front of the camera (-Z, for example a window in view) back-lights the object.

## 2. Image

The model sees the **un-mirrored** image, exactly as the sensor delivers it: the pixel on the image's
right is in the +X half of the view.

If the UI mirrors a front ("selfie") camera for display, the estimate must be mirrored to match the
displayed picture, **once**, with `mirrorX` (`web/src/lighting/sh.ts`). Under x -> -x the
coefficients odd in x change sign: indices 3 (x), 4 (xy) and 7 (xz).

```
sh_mirrored[i] = s[i] * sh[i],   s = [+1, +1, +1, -1, -1, +1, +1, -1, +1]
```

Tested: `mirrorX(sh)` evaluated at (x, y, z) equals `sh` evaluated at (-x, y, z) (pytest and vitest).

## 3. Spherical harmonics

Real SH up to band 2: **nine coefficients per colour channel**, basis and ordering exactly three.js
`SphericalHarmonics3.getBasisAt` (r187) for a unit vector (x, y, z):

| i | band | Y_i(x, y, z) |
|---|---|---|
| 0 | 0 | 0.282095 |
| 1 | 1 | 0.488603 * y |
| 2 | 1 | 0.488603 * z |
| 3 | 1 | 0.488603 * x |
| 4 | 2 | 1.092548 * x * y |
| 5 | 2 | 1.092548 * y * z |
| 6 | 2 | 0.315392 * (3 z^2 - 1) |
| 7 | 2 | 1.092548 * x * z |
| 8 | 2 | 0.546274 * (x^2 - y^2) |

Layout everywhere: `sh[i] = [r, g, b]`, shape [9, 3]. The model output [1, 9, 3] is the same
layout with a batch axis; flattened, value `3 * i + c` is coefficient i of channel c.

Coefficients describe incident **radiance** L(w) in linear RGB, not pre-convolved irradiance:

```
L_i = integral over the sphere of L(w) Y_i(w) dw
```

Irradiance at a unit normal n (cosine-weighted radiance over the hemisphere around n):

```
E(n) = sum_i A_band(i) * L_i * Y_i(n),   A = [pi, 2 pi / 3, pi / 4]
```

A white (albedo 1) Lambertian surface then has outgoing radiance `E(n) / pi`, and a pixel value
`srgb(E(n) / pi)` with the IEC 61966-2-1 transfer function
(`12.92 x` for x <= 0.0031308, else `1.055 x^(1/2.4) - 0.055`).

### three.js: settled by measurement

**three.js r187 `LightProbe` takes radiance SH in exactly this basis. `toThreeSH` is the identity.**

Source reading: `SphericalHarmonics3.getIrradianceAt` and the GLSL `shGetIrradianceAt` multiply the
coefficients by `0.886227 = pi * 0.282095`, `2 * 0.511664 = (2 pi / 3) * 0.488603`,
`2 * 0.429043 = (pi / 4) * 1.092548`, and so on: they apply the cosine convolution themselves.
`MeshLambertMaterial` then multiplies irradiance by `BRDF_Lambert = diffuseColor / pi`.

Measurement (`cd web && npx playwright test e2e/parity.spec.ts`, 2026-11-13, three 0.187.0, headless
Chromium 156 with SwiftShader): a white `MeshLambertMaterial` unit sphere lit only by
`new LightProbe(toThreeSH(sh), 1)`, `NoToneMapping`, `SRGBColorSpace` output, orthographic camera
down -Z, compared with `shared/sphere_reference.png` over the 11,620 pixels inside radius 0.95:

| coefficients given to the probe | mean abs error (8-bit levels) | max |
|---|---|---|
| radiance SH as is (the convention) | 0.013 | 1 |
| pre-convolved, A_band / pi times radiance | 14.795 | 34 |
| pi times radiance | 75.776 | 102 |

The test asserts the first row is within mean 0.5 and max 2 levels and that both controls are off by
more than 2.5 levels on average, so it fails if three.js changes its convention. If it ever does, the
conversion goes in `toThreeSH` and nowhere else.

Conditions of that result, which the app must keep or re-measure: probe `intensity` 1; only
`MeshLambertMaterial` was measured. `MeshStandardMaterial` takes the same probe irradiance but, per
the r187 source (`RE_IndirectDiffuse_Physical`), scales its diffuse by
`1 - singleScattering - multiScattering` (energy given to the specular lobe) and adds view-dependent
specular; the size of that difference is UNVERIFIED (not measured). Re-run the parity harness with it
before claiming a match. Tone mapping (ACES, AgX) changes the pixels by design and is not part of this
contract.

## 4. Panoramas and labels (training side)

Equirectangular panoramas use the three.js equirect mapping in a right-handed, +Y-up world frame:
row 0 is the zenith (+Y), and column coordinate `u = atan2(z, x) / (2 pi) + 0.5` from the left
edge, so the centre column looks along +X and `u = 0.25` looks along -Z. Seen from inside, the
picture is not mirrored: turning right increases u.

`camera_rotation(yaw, pitch, roll)` returns the camera-to-world matrix
`R = Ry(yaw) Rx(pitch) Rz(roll)`; a camera-frame direction d_c is the world direction `R d_c`.
Positive yaw turns left, positive pitch looks up, positive roll turns the camera body
counter-clockwise as seen from behind it.

The label for a crop is the SH of the **whole** panorama expressed in that camera's frame:
`project_equirect(env, cam_to_world=R)`, or equivalently and much faster `rotate_sh(sh_world, R)`
(exact: rotation never mixes bands). Pixel solid angles are exact:
`(2 pi / W) (cos theta_top - cos theta_bottom)`, summing to 4 pi.

## 5. Units and exposure

The training label is the radiance SH of the environment **after the same exposure scale and
white-balance gains that produced the training image**, in linear RGB. The network therefore
predicts "the lighting as this camera exposure sees it": a white diffuse sphere lit by the
prediction and passed through the sRGB transfer function has the brightness and colour cast it
would have had in that photo. Absolute scene brightness is not recoverable and is never claimed.

## 6. Dominant light

From the luminance SH `l_i = 0.2126 r_i + 0.7152 g_i + 0.0722 b_i`:

```
v = (l_3, l_1, l_2)                 # (x, y, z): Y_3 ~ x, Y_1 ~ y, Y_2 ~ z
direction = v / |v|                 # unit vector TOWARD the light, camera frame
directionality = |v| / (sqrt(3) * l_0), clamped to [0, 1]
```

For any non-negative radiance `|v| <= sqrt(3) l_0` (0.488603 / 0.282095 = 1.732051), so
directionality is 0 for uniform light and 1 for a single point light. The direction is `null` when
`l_0 <= 0` or directionality < 1e-6. Use directionality to fade a derived shadow out in flat light.
For a single zonal lobe the band-1 vector points exactly along the lobe axis (tested).

## 7. Model file

| | |
|---|---|
| Path | `web/public/models/lumina-sh.onnx`, served at `/lumina/models/lumina-sh.onnx` |
| Format | ONNX, opset 17 or newer (placeholder: 18), static shapes |
| Input | `image`: float32 [1, 3, 96, 128], NCHW, RGB, sRGB-encoded values in 0..1. Made from the camera frame by centre-cropping to 4:3 and resizing to 128 x 96. Any normalisation is inside the graph. |
| Output | `sh`: float32 [1, 9, 3] (coefficient, RGB), linear radiance SH in the camera frame |
| Metadata | `lumina.contract` = "1"; `lumina.placeholder` = "true" or "false" |

The committed file is a **placeholder** (random weights, seed 20261113, biased toward a uniform grey
radiance of 0.5 plus a band-1 term toward +Y). Regenerate it with
`uv run python -m lumina_train.placeholder`. The training engineer replaces it; the tests in
`training/tests/test_model_contract.py` and `web/e2e/model.spec.ts` check only the contract, so
they keep working on the real model.

## 8. Shared test vectors

`shared/sh_vectors.json` (schema 1), written by `uv run python -m lumina_train.vectors`:

- `normals`: 20 unit normals (6 axes, 8 cube diagonals, 6 random with seed 20261113) and
  `basis_at_normals`.
- `environments`, each with `sh`, `irradiance` at every normal, `dominant_direction`,
  `directionality` and `sh_mirrored_x`. Coefficients are exact (Funk-Hecke with Gauss-Legendre
  quadrature, `training/lumina_train/environments.py`) and pytest checks they equal a numerical
  projection of the rendered panorama to 1e-4:
  - `constant_white`: radiance 1 everywhere.
  - `lobe_pos_x`, `lobe_pos_y`, `lobe_pos_z`, `lobe_neg_z`: white soft lobe `exp(8 (w . d - 1))`.
  - `lobe_oblique_warm`: lobe colour (3.0, 2.4, 1.65) toward (2, 2, 1)/3 over ambient
    (0.05, 0.06, 0.08).
  - `sky_ground`: (0.6, 0.75, 1.0) for y > 0, (0.35, 0.25, 0.15) for y < 0.
- `shared/sphere_reference.png`: 128 x 128, 8-bit sRGB RGB, white Lambertian sphere under
  `lobe_oblique_warm`, orthographic camera down -Z, row 0 = +Y, black background.

pytest fails if the committed files differ from what the code generates (`test_vectors.py`).

## 9. Worked examples

All numbers below are printed by `uv run python -m lumina_train.worked_examples`, red channel
unless noted.

**Constant white.** L_0 = 4 pi * 0.282095 = 3.54491, all other coefficients 0. E(n) = pi * L_0 *
0.282095 = 3.14160 for every n (pi up to the 6-digit basis constants), so a white sphere has
outgoing radiance 1.000 and pixel value 255.

**White lobe toward +Y** (`lobe_pos_y`):
`sh[:, r] = [0.221557, 0.335779, 0, 0, 0, 0, -0.166429, 0, -0.288263]`.
Only Y_1 (~y) is non-zero in band 1, so the light is up. Irradiance: E(+Y) = 0.704865,
E(+X) = 0.113899, E(-Y) = 0.017640; as 8-bit sRGB of E / pi: 130, 54, 17.
Dominant direction (0, 1, 0), directionality 0.875.

**Mirroring.** `lobe_pos_x` has `sh[3] = +0.335779` (x term). `mirrorX` makes it -0.335779 and the
dominant direction becomes (-1, 0, 0): a light on the right of the sensor image is on the left of
the mirrored display.

**Camera rotation.** The same +X lobe seen by a camera yawed right by 90 degrees
(`camera_rotation(-pi/2, 0, 0)`, now facing world +X) has, in camera space,
`sh[2] = -0.335779` (z term) and dominant direction (0, 0, -1): straight ahead.

**Reference sphere** (`lobe_oblique_warm`): dominant direction (0.667, 0.667, 0.333),
directionality 0.632. At the normal facing the light, E = (2.2717, 1.8802, 1.4144), outgoing
radiance E / pi = (0.7231, 0.5985, 0.4502), 8-bit sRGB (221, 203, 179), the brightest pixel in
`sphere_reference.png`.

Reproduce every number in this section: `uv run python -m lumina_train.worked_examples`
(deterministic, no seed involved).

## 10. Traps

| You wrote | What happens | What you see |
|---|---|---|
| Pre-convolved (irradiance) coefficients into `LightProbe` | three.js convolves again | flatter, lower-contrast shading (mean error 14.8 levels above) |
| `sh * Math.PI` "to convert to irradiance" | 3.14 times too bright | blown-out object |
| `mirrorX` applied in the estimator and again in the renderer | double flip | light on the wrong side, only on the front camera |
| Labels from the panorama without rotating into the camera frame | network learns world-frame light | direction ignores where the camera points |
| Video pixels fed as linear values | input contract says sRGB-encoded 0..1 | systematic brightness and contrast error |
| `cv2.imwrite` on an RGB array | file stored as BGR | colour cast swapped (pytest guards the reference PNG) |
| `MeshStandardMaterial` compared against the Lambert reference | diffuse scaled by `1 - single - multi` scattering, plus specular | a mismatch of UNVERIFIED size |
