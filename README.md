# Lumina

Live demo: https://television05.github.io/lumina/

Lumina estimates the lighting of a room from one webcam frame and uses it to light a 3D object. A
small convolutional network (1.59 M parameters, 6.4 MB) regresses nine spherical-harmonic RGB
coefficients from a 96 x 128 frame. The network runs in your browser with onnxruntime-web, and
three.js turns the coefficients into a LightProbe, a derived DirectionalLight and a shadow catcher,
smoothed over time with a One Euro filter. Move a lamp and the object's shading, colour cast and
shadow follow it. How well that works is measured below, on synthetic data only.

[![Demo: fixed studio light on the left, Lumina on the right, over a panning clip with changing light (click for the 13 s video)](docs/media/lumina-demo.jpg)](docs/media/lumina-demo.mp4)

The demo clip (`docs/media/lumina-demo.mp4`, 13 s) was recorded headless from the bundled synthetic
sample clips played as a fake camera: the left half of the ball uses a fixed studio light, the right
half uses Lumina's estimate. The picture is a rendered panorama, not a real room.

## Try it

Open the demo, press "Use my camera" (or drop a photo or video on the page, or use the bundled
sample clip). Tested only in headless Chromium (no real camera was used). WebGPU is used when available,
otherwise single-thread WASM; no special server headers are required.

## Run it locally

```sh
uv sync                                    # Python 3.12 venv in .venv
uv run pytest                              # Python tests (102)

cd web
npm ci
npm run lint && npm run format:check && npm run typecheck
npm test                                   # vitest (132 tests)
npx playwright install chromium            # once
npm run e2e                                # headless Chromium with a fake camera, never the real one
npm run e2e:dist                           # production build under /lumina/ like Pages, plus the truth test
                                           # (needs ffmpeg and uv on PATH)
npm run dev                                # http://localhost:5179/lumina/
```

Retrain and re-evaluate: `uv run python -m lumina_train.train`, then `uv run python -m
lumina_train.evaluate` (needs the HDRIs, see `docs/DATA.md`). The training/web contract is
`docs/CONVENTIONS.md`; the app is described in `docs/WEB.md`.

## How well does it work

**Honest summary: on held-out synthetic data the shipped model beats every simple baseline on
average error and finds the light direction to about 30 degrees, but the size of its predicted
lighting is still far smaller than the true spread, and nothing here has been measured on a real
webcam.** The weights are the step-29,000 checkpoint of a run launched with `--steps 50000` and
stopped at about step 29,000 (1.84 h on one laptop GPU). It was chosen by the lowest validation
sRGB RMSE among the three saved checkpoints (steps 2,000 / 10,000 / 29,000: 67.3 / 41.2 / 39.7), not
by test results (the test set had only been scored with the earlier step-2,000 model). The launch command is recorded in
`web/public/models/lumina-sh.json`; `--steps 29000` would not reproduce the weights because the
learning-rate schedule depends on it. All numbers are from `eval/results.json`, generated
2026-11-14, on 103 held-out Poly Haven HDRIs (1,545 synthetic crops), mean over HDRIs with 95 %
bootstrap CIs over HDRIs. Lower is better. Full tables and failure cases: `docs/EVAL.md`. Test
error (sRGB RMSE 37.1) is in line with validation (39.7), so the test set did not turn out worse
than validation suggested.

**Output spread.** On the test crops the variance of the predicted coefficients is 1.0 % of the
label variance (9.6 % on the out-of-distribution set), the per-coefficient correlation with the
truth is 0.06 to 0.43, and R2 against always answering the average is 0.014. The labels span a
huge range (exposure-relative radiance), so the demo's lighting stays much closer to an average
than the true lighting does, while still moving in the right direction and colour (the truth test
below). The previous release, the step-2,000 snapshot, had 0.006 %, 0.01 to 0.21 and -0.007.

| Method | RMSE (linear) | si-RMSE (linear) | sRGB RMSE (8-bit levels) | Colour error (deg) | Direction error, directional scenes (deg) |
|---|---|---|---|---|---|
| Lumina CNN | 0.884 [0.411, 1.658] | 0.517 [0.222, 1.025] | 37.1 [33.6, 40.6] | 5.5 [4.9, 6.0] | 30.1 [26.3, 33.9] |
| Training-set mean lighting | 1.405 [0.962, 2.148] | 0.673 [0.384, 1.141] | 99.7 [95.2, 104.2] | 18.9 [18.2, 19.5] | 53.1 [47.8, 58.3] |
| Image-mean colour, ambient only | 1.099 [0.611, 1.885] | 0.613 [0.364, 1.012] | 70.5 [65.2, 75.5] | 8.7 [7.9, 9.6] | 90.0 (no direction) |
| Image-mean colour + brightest-region direction | 1.157 [0.668, 1.943] | 0.939 [0.516, 1.638] | 109.5 [103.4, 115.3] | 8.7 [7.9, 9.6] | 79.9 [75.7, 84.2] |

For comparison, the previous release (step-2,000 snapshot) scored 1.061 / 0.583 / 61.9 / 13.0 /
53.2 on the same five columns.

Paired comparison (per-HDRI difference CNN minus baseline, 95 % bootstrap CI, `docs/EVAL.md`):

- On the test split the CNN is better than all three baselines on linear RMSE, sRGB RMSE, colour
  error and direction error (the whole CI is below zero). On si-RMSE it is better than the training
  mean and the direction heuristic, with no clear difference from the plain image-mean colour.
- On 48 out-of-distribution ambientCG HDRIs it is better than every baseline on every metric
  (CNN RMSE 0.711 [0.488, 0.989], sRGB RMSE 42.9 against 86.8 for the training mean, direction
  31.7 against 39.5 degrees).
- Train/test leakage check: `snowy_forest` (train) vs `snowy_forest_path_01`/`_02` (test) are by
  different photographers and show different scenes, so they are not the same place; no result was
  excluded (`docs/EVAL.md`, limits).

## Limits

- Every number above is on synthetic camera crops of HDRIs. **Accuracy on real webcams has not been
  measured**; there is no real-photo test set with ground-truth lighting.
- Estimates are relative to the frame's exposure: absolute brightness is not recoverable.
- Nine spherical-harmonic coefficients are low frequency: hard shadows and sharp highlights are
  not represented.
- Dominant-light direction is about 30 degrees off on average on directional scenes (synthetic
  test); diffuse scenes have no meaningful direction (about 56 degrees).
- Browser speed, measured with the shipped model (`npm run perf`, 2026-11-14, Apple M2 Max, load
  average about 8, headless Chromium, synthetic camera at 12 inferences per second): network
  inference 4.1 to 4.4 ms median (p95 4.3 to 5.0) on single-thread WASM with software WebGL, and
  5.5 to 7.4 ms median (p95 8.4 to 9.4) on WebGPU with the Metal GPU. Frame interval 16.7 ms median
  on both. Other machines and real GPUs will differ. Details: `docs/WEB.md`.
- On the four bundled sample clips (`web/public/samples/*.json`, scored by the same model) the light
  direction error is median 5.6 degrees on the sunny pan, 8.7 on the rotating-environment clip, 49.6
  on the indoor pan and 125 on the diffuse pan (where a direction hardly exists); their mean sRGB
  RMSE is 60 to 99 levels, higher than the 37 on the test crops, so a few clips are not
  representative of the test average.

## Tests that prove it

`web/e2e/truth.spec.ts` plays a sample clip with known lighting per frame as the fake camera in the
production build and checks the following (ten runs on 2026-11-14 with the shipped model; run
`npm run e2e:dist` to reproduce):

- the 128 x 96 frames the network sees match an independent ffmpeg resize (median 3.45 to 3.48 levels of
  255 apart), and the network's output in the browser equals Python onnxruntime on those very
  frames (largest difference 1.2e-6 to 1.4e-6 on coefficients up to 2.85);
- as the environment turns 360 degrees, the estimated light azimuth sweeps the same way as the true
  one: agreement R = 0.99 in all ten runs against 0.03 to 0.07 for the reversed sweep, median
  direction error 8.7 to 9.4 degrees; the displayed estimate trails the network by 0 clip frames
  (search resolution 2 frames); the lit probe ball's bright side matches the estimated side in 97 to
  100 % of samples and the true side in 99 to 100 %.

Thresholds: same-sweep agreement above 0.9 and above the reversed sweep by 0.5, median direction
error under 20 degrees, probe agreement above 0.9 with both, lag under 1.5 s. They were set after
observing runs, and tightened (never loosened) when the model changed: for the earlier, much weaker
step-2,000 model they were 0.3, +0.1 and probe-vs-truth 0.6, with observed agreement of only 0.46 to
0.58. Mutation checks, done by hand (not as committed tests), each fail the test: pairing frames with
the reversed truth (R 0.05 vs 0.99), a constant estimate (R 0.20), and an x-mirrored estimate
(R 0.04 vs 0.99). The clip is a render of a CC0 panorama from the test split, so this is
still synthetic data. On the slow CI runner the number of network results per second is lower; the
test therefore records until it has 150.

## Privacy

Camera frames stay on your device. The frame is read into a canvas, resized to 128 x 96 and fed to
the network inside a Web Worker in your browser. The only network request the app makes is the
one-off download of its own files (page, scripts, WASM and `lumina-sh.onnx`) from the same origin.
No frame, estimate or analytics is sent anywhere. This is checked by `web/e2e/privacy.spec.ts`,
which fails if a camera session makes any request to another origin, any non-GET request or any
request with a body, and the only network call in the source is the model fetch in
`web/src/estimators/neural.worker.ts`.

## Licence and attribution

MIT, see `LICENSE`. The weights are trained only on CC0 data. Training and test HDRIs come from
[Poly Haven](https://polyhaven.com/hdris) (CC0) and, for the out-of-distribution test only,
[ambientCG](https://ambientcg.com) (CC0); the sample clips are rendered from four Poly Haven
test HDRIs (ticknock_02, storeroom, snowy_forest_path_02, hochsal_field). Thanks to their artists.
Full list: `CREDITS.md`, `docs/DATA.md`.
