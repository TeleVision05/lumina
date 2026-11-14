# Lumina web app

How the browser app is put together, what was measured, and what it cannot do. The contract it
implements is `docs/CONVENTIONS.md`; read that first. Every number below comes from a command in
this file, run on 2026-11-13 on an Apple M2 Max while other jobs kept the load average near 40
(12 cores), so the software-rendering timings are pessimistic.

## Data flow

```
 source element (camera <video>, sample/dropped <video>, dropped <img>), never mirrored in pixels
   │  every ~83 ms (12 Hz, one frame in flight, polled by a 10 ms timer, not by rendering)
   ▼
 FrameGrabber: centre crop 4:3 -> createImageBitmap resize to 256x192 -> exact 2x2 box -> 128x96 RGBA
   │
   ├─ Neural network (Web Worker, onnxruntime-web, WebGPU else single-thread WASM) ┐
   ├─ Image statistics (main thread, < 1 ms)                                        ├─ SH9 + direction + confidence
   └─ Fixed studio rig (constant, already in display frame)                          ┘   (sensor camera frame)
   ▼
 Vector One Euro filter (27 coefficients, one shared adaptive cutoff), every rendered frame
   ▼
 forDisplay(): mirrorX once if the video is shown mirrored  (lighting/estimate.ts)
   ▼
 RigNode x2 ("fixed" and "Lumina"): LightProbe (residual SH) + DirectionalLight (from band 1,
 energy subtracted) + shadow opacity + SH environment map (glossy materials only)
   ▼
 Stage: one scene, rendered once per rig with a scissor for the comparison split, then a
 grain/softness pass, onto a transparent canvas over the CSS video
```

| Path                             | What                                                                                                          |
| -------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| `web/src/capture/`               | Sources (camera, files, sample clips), plain-language capture errors, preprocessing                           |
| `web/src/estimators/`            | The `Estimator` interface, fixed studio rig, image statistics, neural worker and client, ONNX metadata reader |
| `web/src/lighting/`              | SH maths (foundation), filter, light rig, env synthesis, colour temperature, CPU probe render                 |
| `web/src/render/`                | three.js stage, rigs, objects, ground, post pass, PBR shader patch, quality tier                              |
| `web/src/ui/`, `web/src/styles/` | Interface                                                                                                     |
| `web/src/app/`                   | Wiring, settings, scheduler, the `?debug` test handle                                                         |

## How each piece works

**Mirroring.** The `<video>` is mirrored with CSS only for a front or unknown-facing camera
(`shouldMirror`). Estimators always read the un-mirrored pixels; `forDisplay` mirrors the
estimate exactly once. Proven end to end by `e2e/robustness.spec.ts`: with the mirrored fake
camera, the light direction is on the display side opposite the sensor's bright side in 20 of 22
samples, and in 1 of 26 with the mirror removed (mutation check, then restored).

**Preprocessing.** `cropRect43` (largest centred 4:3), browser resize to twice the model size, an
exact 2x2 box average, then NCHW float32 with values `byte / 255` (sRGB-encoded, as the contract
requires). The foundation ships no Python preprocessing vector, so `capture.test.ts` uses
hand-computed cases (crop rectangles, a 2x2 box, plane order, a left/right/top orientation chain).
Swapping R and B makes 2 of them fail (mutation check). The browser's resampling filter for the
first step is not identical to any Python resize; see "Dropping in the trained model".

**Estimators.** All implement `estimate(frame) -> {sh, direction, directionality, confidence}`.

- Fixed studio: neutral key from the upper left over a cool fill; a white matte ball peaks at
  about 0.85 linear.
- Image statistics: ambient = mean linear colour (a white ball in uniform light shows exactly
  that); direction = centroid of the brightest tenth of a 16x12 grid, mapped to
  `(0.8 x, 0.35 + 0.6 y, 0.6)` (a stated heuristic); directional share up to 60% from contrast;
  confidence at most 0.4.
- Neural network: `estimators/neural.worker.ts`. WebGPU when `navigator.gpu` gives an adapter,
  otherwise WASM with `numThreads = 1` (Pages cannot send COOP/COEP). The ORT wasm files are
  Vite-emitted, content-hashed, same-origin assets. Confidence = exposure quality, halved while
  the model's metadata says `lumina.placeholder = true`; the UI then says the weights are
  untrained. If the model cannot load, the app switches to image statistics and says so.
  `?wasm` in the URL forces WASM.

**Filter.** `lighting/filter.ts`. One cutoff for the whole vector keeps the colour of a moving light
stable; speed is divided by the luminance of L0 so dim and bright scenes behave the same. The
Responsiveness slider interpolates geometrically between `{0.05 Hz, beta 0.02, dCutoff 0.3}` and
`{2 Hz, 0.4, 1}`, end points chosen by a grid search over both measures below
(`npx vitest run src/lighting/filter.test.ts`, seed 20261113):

| Responsiveness | Full step settles within 10% (60 Hz) | Jitter kept, 5% white noise at 60 Hz                  |
| -------------- | ------------------------------------ | ----------------------------------------------------- |
| 0 (steady)     | 917 ms                               | 5.4%                                                  |
| 0.5 (default)  | 267 ms (367 ms at 30 Hz)             | 14.6% (37.2% for a 12 Hz estimate held across frames) |
| 1 (quick)      | 67 ms                                | 33.4%                                                 |

**Light rig.** `lighting/lightRig.ts`. The share of band 1 given to the DirectionalLight rises
with directionality (smoothstep 0.06 to 0.45, at most 85%). The light's irradiance per channel is
the band-1 vector projected on the direction divided by 0.488603; its exact SH projection
`e * Y_i(d)` is subtracted from all nine probe coefficients. Shadow opacity on the ground = the
fraction of the ground's irradiance the light supplies, times confidence and the Shadow slider,
so flat light casts no shadow. Over the shared analytic environments, probe plus exact
clamped-cosine light reproduces the estimate's irradiance within 7.89% of peak (worst,
`lobe_oblique_warm`; 0% for constant white). Removing the subtraction fails 7 unit tests and
brightens the three.js render by 24 levels (mutation check, restored).

**Environment map (heuristic).** 128x64 equirect of the SH radiance (clamped at 0), with the camera
picture pasted on the forward hemisphere inside an assumed 55 degree vertical field of view, brought
to the estimate's brightness. Only the forward view is real; nothing behind the camera is
invented, so a chrome ball shows mostly soft gradients with the room at its rim. Converted by
three.js PMREM (never below a 256 px cube) every 150 ms on GPUs, every 2 s on software renderers,
and not at all while the matte ball is shown.

**Measured in three.js** (`cd web && npx playwright test e2e/shading.spec.ts`, white unit sphere,
128 px, against `shared/sphere_reference.png`, 8-bit levels):

| Render                                                         | Mean abs | Max | Mean signed |
| -------------------------------------------------------------- | -------- | --- | ----------- |
| `MeshLambertMaterial`, rig (residual probe + DirectionalLight) | 4.32     | 19  | -0.10       |
| `MeshStandardMaterial` roughness 1, probe only                 | 1.16     | 4   | -1.16       |
| same + SH env map, with `render/pbrPatch.ts`                   | 0.13     | 1   | +0.13       |
| same + SH env map, unpatched (control)                         | 50.67    | 69  | +50.67      |

The patch exists because three.js r187 adds an env map's irradiance to the probe's; it routes the
probe irradiance through three.js's own IBL path and throws if the shader chunk changes shape.

**Tone mapping.** None: linear to sRGB, the pipeline the foundation's parity test measured. The
estimate is "the lighting as this camera exposure sees it", so no curve is what matches the
footage. A grain and softness pass (after the sRGB conversion, on covered pixels only) helps the
object sit in the video; grain is frozen under reduced motion.

**Quality.** Software renderers (SwiftShader, llvmpipe: what Chrome uses on a blocklisted GPU) get
no MSAA, a 512 px shadow map and a 0.5 render scale; a governor moves the scale between 0.4 and 1
from the mean frame cost. Measured under SwiftShader at 1280x720: frame grab 80 to 100 ms with
`drawImage` + `getImageData` vs 2.7 ms with `createImageBitmap`; render 157 ms with 4x MSAA vs 18 ms
without at half scale; `backdrop-filter` over the video took the frame interval from 157 ms to
18 ms, so the interface uses none. Shadow maps re-render only when the light, object or ground move.

## Performance

`cd web && npx playwright test e2e/perf.spec.ts` (software) and
`LUMINA_GPU=1 npx playwright test e2e/perf.spec.ts` (GPU-backed headless, no window). Inference is
the worker's `session.run` time. The table below was measured with the 43,935-byte placeholder
model, before the trained model (6.4 MB) shipped; the lines after it were measured with the trained
model. Frame values are medians with p95 over about the last 2 s, after 8 s per view.

| Run (load average at start)      | View        | Inference median / p95 | Inferences per second | Frame interval median / p95 | Main-thread work median / p95 |
| -------------------------------- | ----------- | ---------------------- | --------------------- | --------------------------- | ----------------------------- |
| Metal GPU, WebGPU inference (56) | Lumina      | 2.9 / 5.0 ms           | 12.4                  | 16.7 / 17.2 ms              | 0.3 / 0.9 ms                  |
|                                  | Compare     | 2.9 / 5.0 ms           | 12.4                  | 16.7 / 17.1 ms              | 0.4 / 1.1 ms                  |
|                                  | Chrome ball | 3.1 / 5.9 ms           | 12.5                  | 16.7 / 17.6 ms              | 0.4 / 3.0 ms                  |
| Metal GPU, WASM inference (50)   | Lumina      | 0.8 / 1.9 ms           | 12.4                  | 16.6 / 17.6 ms              | 0.4 / 1.1 ms                  |
|                                  | Compare     | 0.8 / 1.3 ms           | 12.5                  | 16.6 / 17.8 ms              | 0.4 / 1.0 ms                  |
|                                  | Chrome ball | 0.8 / 1.8 ms           | 12.3                  | 16.7 / 17.2 ms              | 0.4 / 2.3 ms                  |
| SwiftShader, WASM inference (62) | Lumina      | 0.9 / 2.0 ms           | 8.4                   | 14.0 / 137.8 ms             | 0.3 / 1.4 ms                  |
|                                  | Compare     | 0.9 / 2.7 ms           | 4.0                   | 14.0 / 323.0 ms             | 0.3 / 1.5 ms                  |
|                                  | Chrome ball | 0.8 / 2.7 ms           | 3.4                   | 7.9 / 799.4 ms              | 0.4 / 2.8 ms                  |

GPU renderer string: `ANGLE (Apple, ANGLE Metal Renderer: Apple M2 Max)`; software:
`ANGLE (Google, Vulkan 1.3.0 (SwiftShader Device (LLVM 10.0.0)))`. Earlier SwiftShader runs the
same day at lower load reached 11.0 to 11.8 inferences per second in the Lumina view, which shows
how much the software numbers depend on what else the machine is doing.

Trained model (6,439,046 bytes), 2026-11-13, M2 Max, load average about 30, `npx playwright test
e2e/perf.spec.ts` (software) and `LUMINA_GPU=1 npx playwright test e2e/perf.spec.ts` (Metal GPU,
headless, no window):

| Run                                | View        | Inference median / p95 | Inferences per second | Frame interval median / p95 |
| ---------------------------------- | ----------- | ---------------------- | --------------------- | --------------------------- |
| SwiftShader WebGL, WASM inference  | Lumina      | 4.30 / 5.00 ms         | 12.5                  | 16.6 / 50.3 ms              |
|                                    | Compare     | 4.20 / 4.40 ms         | 12.4                  | 15.7 / 69.2 ms              |
|                                    | Chrome ball | 4.20 / 4.40 ms         | 11.8                  | 16.7 / 17.6 ms              |
| Metal GPU, WebGPU inference        | Lumina      | 5.80 / 6.70 ms         | 12.5                  | 16.6 / 17.4 ms              |
|                                    | Compare     | 5.80 / 6.20 ms         | 12.5                  | 16.7 / 17.2 ms              |
|                                    | Chrome ball | 5.20 / 7.90 ms         | 12.5                  | 16.7 / 17.6 ms              |

For the trained model WASM is not slower than WebGPU (4.2 vs 5.8 ms median): at this size
dispatch overhead dominates, so the WebGPU default is kept only because it frees the main
thread's worker on slow CPUs (UNVERIFIED: not measured on a slow CPU). In the compare view under
software rendering, GPU-process stalls delay timers; on the GPU the rate holds at 12.5 per second.

## Tests

| Command                                                            | Covers                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| ------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `npm test`                                                         | vitest: SH maths and shared vectors (foundation), filter step response and jitter, light rig energy, shadow opacity, mirroring, env map layout, CPU probe vs the Python PNG (0 levels difference), preprocessing, capture errors, estimators, ONNX metadata, scheduler, governor, shader patch, UI copy, settings                                                                                                                                                                                                                                                              |
| `npm run e2e`                                                      | Playwright, headless Chromium, fake camera fed by `e2e/fixtures/scene.ts` (generated Y4M, no ffmpeg): foundation parity and ONNX smoke, shading measurements, app start and inference, estimator and view switching change the shading (mean colour moves about 113 levels), lamp tracking from the camera and from dropped photos, mirroring, drop event, dropped video, sample clip absence, camera denied / missing / busy / insecure page, non-media file, model 404 fallback, failed camera switch, keyboard-only flow, reduced motion, phone portrait layout, perf probe |
| `npm run e2e:dist`                                                 | Builds, serves `dist/` under `/lumina/` with `scripts/serve-pages.mjs` (Pages-like: no SPA fallback, no custom headers) and runs every app spec against it, failing on any console error or 4xx/5xx request                                                                                                                                                                                                                                                                                                                                                                    |
| `LUMINA_SCREENSHOTS=1 npx playwright test e2e/screenshots.spec.ts` | Rewrites `docs/media/web-*.png`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |

The fake camera reports no `facingMode`, like a laptop webcam, so it is shown mirrored.

`e2e/truth.spec.ts` (runs only under `npm run e2e:dist`; needs `ffmpeg` and `uv` on PATH) converts
`samples/rotate-environment.mp4` to a Y4M fake camera, records every network result in the page for
16 s, and checks the frames against an ffmpeg resize, the output against Python onnxruntime
(`scripts/ort_reference.py`) and the estimated light azimuth, its lag and the probe ball against the
clip's per-frame truth. See the README for the measured numbers. `e2e/record.spec.ts` records the
demo video (`LUMINA_RECORD=out.webm LUMINA_CLIP=<name> LUMINA_GPU=1 npx playwright test
e2e/record.spec.ts`).

## Dropping in the trained model

The trained model (`placeholder = false`) is what ships as of 2026-11-13; these steps describe
how to replace it with another.

Replace `web/public/models/lumina-sh.onnx` with a model that meets the contract and has
`lumina.placeholder = "false"`. No code changes: the placeholder note and the halved confidence
disappear on their own. Then:

1. Run `npm run e2e` and `npm run perf` (and `LUMINA_GPU=1 npm run perf`); compare WebGPU and WASM
   inference times. If WASM wins for the real model too, consider making it the default.
2. Check the training resize against the browser's: the app resamples the 4:3 crop with
   `createImageBitmap(..., { resizeQuality: 'medium' })` to 256x192 then box-averages 2x2.
   Train with an area or bilinear downscale and blur augmentation so the difference does not
   matter. UNVERIFIED: how close the two are numerically.

## Sample clips

Put clips in `web/public/samples/` (owned by the training engineer). They are listed at build time
(`vite/plugins.ts`); the first in alphabetical order is played. With none, the intro hides the
button and says so. Use H.264 MP4 for Safari; Playwright's Chromium cannot decode H.264, so the
e2e fixture is VP9 (`web/e2e/fixtures/README.md`).

## Deploying

`.github/workflows/ci.yml` runs the e2e job and, on pushes to `main`, builds `web/dist` and deploys
it with the GitHub Pages actions. In the repository settings, set Pages > Source to "GitHub
Actions" once. Target: https://television05.github.io/lumina/

## Limits and what is not verified

- UNVERIFIED: real devices. Everything was run in headless Chromium (SwiftShader and Metal). Not
  run on iOS Safari, Android Chrome, Firefox or desktop Safari, so the front/rear switch with real
  `facingMode`, the iOS motion-permission prompt and WebGPU on Safari are untested.
- UNVERIFIED: whether GitHub Pages gzips `.wasm`; the WASM runtime is 14.8 MB raw (3.9 MB gzip)
  and the WebGPU runtime 28.7 MB (7.3 MB gzip). Only one is downloaded per visit.
- UNVERIFIED: the CI workflow has not run on GitHub (no remote exists).
- The camera's field of view (55 degrees vertical) and the floor height are assumptions; there is
  no tracking, so the object does not stay put when the camera moves.
- Image statistics' direction mapping is a heuristic, and its confidence is capped at 0.4 to say so.
- The chrome ball's reflection is SH plus the forward camera view; it is not a measured
  environment.
