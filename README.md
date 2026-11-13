# Lumina

Real-time lighting estimation for browser AR. Lumina reads the live camera feed, infers the
lighting of the real room with a small convolutional network running in the browser, and lights a
virtual 3D object with it, so the object's shading, colour cast and shadow follow the room.

Status: the shared spherical-harmonics maths, the conventions contract, the test vectors and the
web app exist. The app ships a **placeholder** model (random weights, correct input/output
contract) and says so on screen; the trained model drops into `web/public/models/` with no code
change. How the app works and what was measured: `docs/WEB.md`.

![Fixed studio light and Lumina side by side over the camera picture](docs/media/web-compare.png)

## Layout

| Path | What |
|---|---|
| `training/lumina_train` | Python package: SH maths, test-vector writer, placeholder model export |
| `web/` | Vite + TypeScript + three.js + onnxruntime-web app (base path `/lumina/`) |
| `shared/` | Test vectors written by Python, read by both test suites |
| `docs/CONVENTIONS.md` | The contract between training and the web app. Read it first. |
| `docs/WEB.md` | The web app: data flow, light rig, filter, measurements, tests, limits |

## Commands

```sh
uv sync                                    # Python 3.12 venv in .venv
uv run pytest                              # Python tests
uv run python -m lumina_train.vectors      # regenerate shared/sh_vectors.json + sphere_reference.png
uv run python -m lumina_train.placeholder  # regenerate web/public/models/lumina-sh.onnx

cd web
npm ci
npm run lint && npm run format:check && npm run typecheck
npm test                                   # vitest
npx playwright install chromium            # once
npm run e2e                                # headless Chromium, fake camera: parity, ONNX, the app
npm run build                              # static site in web/dist, base /lumina/
npm run e2e:dist                           # build, serve under /lumina/ like Pages, run app specs
npm run perf                               # inference and frame timings (LUMINA_GPU=1 for Metal)
npm run dev                                # http://localhost:5179/lumina/
```

Pre-commit hooks: `uv run pre-commit install` (ruff, eslint, prettier, 500-line limit).

## Licence

MIT, see `LICENSE`. Data and asset credits are in `CREDITS.md`.
