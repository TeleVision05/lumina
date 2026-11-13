# Credits and data

Every dataset, model weight, texture, video or other asset used to build or shipped with Lumina is
listed here with its source URL and licence. Only CC0 or similarly unrestricted material is allowed;
research-only datasets (for example Laval Indoor HDR, SUN360) are not.

| Asset | Source | Licence | Used for |
|---|---|---|---|
| three.js 0.187.0 (shipped in the web bundle) | https://github.com/mrdoob/three.js | MIT | Rendering |
| onnxruntime-web 1.31.1 (JS and wasm binaries shipped in the web build) | https://github.com/microsoft/onnxruntime | MIT | In-browser inference |
| Web app 3D objects (matte ball, chrome ball, glazed vase, brass knot) | Procedural, `web/src/render/objects.ts` | Part of this repo (MIT) | Virtual objects; no third-party models or textures |
| e2e test footage (room clip, photos, `web/e2e/fixtures/placeholder-clip.mp4`) | Synthetic, `web/e2e/fixtures/scene.ts` | Part of this repo (MIT) | Fake camera and file tests |

The web app uses system fonts only and loads nothing from third-party servers at runtime.

Planned: Poly Haven HDRIs (https://polyhaven.com/hdris, licence https://polyhaven.com/license, CC0),
downloaded through the public API with a generic client User-Agent, as allowed by
https://github.com/Poly-Haven/Public-API/blob/master/ToS.md.

The current `web/public/models/lumina-sh.onnx` is a randomly initialised placeholder and contains no
data-derived weights.
