# Lumina

Real-time lighting estimation for browser AR. Lumina reads the live camera feed, infers the
lighting of the real room with a small convolutional network running in the browser, and lights a
virtual 3D object with it, so the object's shading, colour cast and shadow follow the room.

Status: foundation only. The shared spherical-harmonics maths, the conventions contract, the test
vectors and a **placeholder** model (random weights, correct input/output contract) exist. The
dataset pipeline, the trained model and the app UI do not exist yet.

## Layout

| Path | What |
|---|---|
| `training/lumina_train` | Python package: SH maths, test-vector writer, placeholder model export |
| `web/` | Vite + TypeScript + three.js + onnxruntime-web app (base path `/lumina/`) |
| `shared/` | Test vectors written by Python, read by both test suites |
| `docs/CONVENTIONS.md` | The contract between training and the web app. Read it first. |

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
npm run e2e                                # headless Chromium: three.js parity + ONNX smoke
npm run build                              # static site in web/dist, base /lumina/
```

Pre-commit hooks: `uv run pre-commit install` (ruff, eslint, prettier, 500-line limit).

## Licence

MIT, see `LICENSE`. Data and asset credits are in `CREDITS.md`.
