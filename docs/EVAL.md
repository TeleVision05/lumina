# Evaluation

No results yet: the committed model is a random placeholder. Every number added here must name the
committed command, seed and split file that reproduce it, and the date it was measured.

## Planned protocol (from the research brief)

- **Split by HDRI, never by crop** (about 80/10/10, fixed seed, split file committed). A fixed list
  of test crops: poses x exposure/white-balance settings x all test HDRIs.
- **Dominant-light angular error** (degrees; mean, median, 90th percentile) against the direction
  from ground-truth SH (`dominant_direction`, docs/CONVENTIONS.md section 6), reported separately
  for directional scenes (high directionality) and diffuse scenes.
- **Diffuse-sphere error**: RMSE and scale-invariant RMSE between spheres rendered from predicted
  and ground-truth SH (`render_diffuse_sphere`, albedo 0.5), plus RGB angular error of the mean
  sphere colour.
- **Baselines** with the same metric: constant training-set mean SH; grey-world image mean as L0
  only; image mean plus brightest-region direction; the CNN. Mean with 95% bootstrap CI over
  HDRIs (not crops), and the CNN's win rate against each baseline.
- **Robustness**: exposure +/-2 EV, white balance 3000-9000 K, JPEG quality, noise, field of view,
  indoor vs outdoor.
- **Browser performance** (M2 Max, headless and headed Chromium, wasm and WebGPU separately):
  median and p95 inference time after warm-up, full frame time, model size; record browser
  version, backend, `crossOriginIsolated` and thread count.

Numbers from papers (DeepLight, Garon et al.) use different test sets and definitions; quote them
for context only, never as a like-for-like comparison.
