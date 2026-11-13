# Credits and data

Every dataset, model weight, texture, video or other asset used to build or shipped with Lumina is
listed here with its source URL and licence. Only CC0 or similarly unrestricted material is allowed;
research-only datasets (for example Laval Indoor HDR, SUN360) are not.

| Asset                                                                         | Source                                   | Licence                 | Used for                                           |
| ----------------------------------------------------------------------------- | ---------------------------------------- | ----------------------- | -------------------------------------------------- |
| three.js 0.187.0 (shipped in the web bundle)                                  | https://github.com/mrdoob/three.js       | MIT                     | Rendering                                          |
| onnxruntime-web 1.31.1 (JS and wasm binaries shipped in the web build)        | https://github.com/microsoft/onnxruntime | MIT                     | In-browser inference                               |
| Web app 3D objects (matte ball, chrome ball, glazed vase, brass knot)         | Procedural, `web/src/render/objects.ts`  | Part of this repo (MIT) | Virtual objects; no third-party models or textures |
| e2e test footage (room clip, photos, `web/e2e/fixtures/placeholder-clip.mp4`) | Synthetic, `web/e2e/fixtures/scene.ts`   | Part of this repo (MIT) | Fake camera and file tests                         |

The web app uses system fonts only and loads nothing from third-party servers at runtime.

## Training data and weights

| Asset                                                                             | Source                                                                               | Licence                                      | Used for                                                                                                                                                                                                    |
| --------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ | -------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 997 Poly Haven HDRIs (792 train, 102 validation, 103 test), downloaded 2026-11-13 | https://polyhaven.com/hdris, ids listed in `docs/DATA.md`                            | CC0 1.0, https://polyhaven.com/license       | Training, validation and held-out test of the network; the four sample clips in `web/public/samples/` are rendered from four test-split HDRIs (ticknock_02, storeroom, snowy_forest_path_02, hochsal_field) |
| 48 ambientCG HDRIs, downloaded 2026-11-13                                         | https://ambientcg.com, ids listed in `docs/DATA.md`                                  | CC0 1.0, https://docs.ambientcg.com/license/ | Out-of-distribution test only, never trained on                                                                                                                                                             |
| `web/public/models/lumina-sh.onnx` (6,439,046 bytes)                              | Trained in this repo (`training/lumina_train`) on the Poly Haven training split only | MIT (weights trained only on CC0 data)       | The shipped network                                                                                                                                                                                         |

HDRIs were fetched through the public APIs with the generic User-Agent `lumina-hdri-fetch/0.1`, as
allowed by https://github.com/Poly-Haven/Public-API/blob/master/ToS.md. The HDRI files themselves
are not redistributed in this repository; CC0 requires no attribution, and Poly Haven and ambientCG
are credited here regardless. Thank you to their artists.
