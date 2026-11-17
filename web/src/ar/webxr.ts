/**
 * WebXR `immersive-ar` session for Android Chrome (and other WebXR AR browsers):
 * - Real-world floor/surface tracking via `XRHitTestSource`
 * - Tap-to-place (`select`) onto the tracked floor plane with contact & directional shadows
 * - Live camera frame readback (`camera-access`) so Lumina's neural lighting estimator runs in AR,
 *   with WebXR `light-estimation` as fallback when raw camera access is unavailable.
 */
import { Quaternion, Vector3 } from 'three';
import type { FrameSample } from '../capture/frame';
import { makeEstimate, type LightingEstimate } from '../lighting/estimate';
import type { Vec3 } from '../lighting/sh';
import type { Stage } from '../render/stage';
import { XrCameraGrabber } from './cameraGrab';
import {
  createReticle,
  MIN_FLOOR_UP,
  poseInfo,
  rayFromPose,
  rayHitsDisc,
  rayHitsSphere,
  RETICLE_TAP_RADIUS,
  type Reticle,
} from './placement';

export interface ArPlacementState {
  tracking: boolean;
  placed: boolean;
}

export interface WebXrOptions {
  stage: Stage;
  overlayRoot: HTMLElement;
  onStateChange: (state: ArPlacementState) => void;
  onEnd: () => void;
}

function estimateFromWebXrProbe(estimate: XRLightEstimate): LightingEstimate {
  const c = estimate.sphericalHarmonicsCoefficients;
  const sh: Vec3[] = [];
  for (let i = 0; i < 9; i++) {
    const o = i * 3;
    sh.push([c[o] ?? 0, c[o + 1] ?? 0, c[o + 2] ?? 0]);
  }
  return makeEstimate(sh, 0.7);
}

export class WebXrArSession {
  private readonly reticle: Reticle;
  private hitTestSource: XRHitTestSource | null = null;
  private binding: XRWebGLBinding | null = null;
  private grabber: XrCameraGrabber | null = null;
  private lightProbe: XRLightProbe | null = null;
  private placed = false;
  private lastTracking = false;
  private ended = false;
  private readonly tmpPos = new Vector3();

  private constructor(
    private readonly session: XRSession,
    private readonly options: WebXrOptions,
  ) {
    this.reticle = createReticle();
    options.stage.scene.add(this.reticle.group);
    options.stage.setArMode(true);

    this.initFeatures();
    session.addEventListener('select', this.onSelect);
    session.addEventListener('end', this.onSessionEnd);
  }

  static async start(options: WebXrOptions): Promise<WebXrArSession> {
    const xr = navigator.xr;
    if (xr === undefined) throw new Error('WebXR is not available in this browser');
    const session = await xr.requestSession('immersive-ar', {
      requiredFeatures: ['hit-test'],
      optionalFeatures: ['local-floor', 'dom-overlay', 'camera-access', 'light-estimation'],
      domOverlay: { root: options.overlayRoot },
    });
    const renderer = options.stage.renderer;
    renderer.xr.enabled = true;
    renderer.xr.setReferenceSpaceType('local');
    await renderer.xr.setSession(session);
    return new WebXrArSession(session, options);
  }

  private initFeatures(): void {
    const session = this.session;
    void session
      .requestReferenceSpace('viewer')
      .then((viewerSpace) => session.requestHitTestSource?.({ space: viewerSpace }))
      .then((source) => {
        if (!this.ended && source !== undefined) this.hitTestSource = source;
        else source?.cancel();
      })
      .catch((err: unknown) => {
        console.warn('WebXR hit-test source unavailable', err);
      });

    const gl = this.options.stage.renderer.getContext();
    if (typeof XRWebGLBinding !== 'undefined' && gl instanceof WebGL2RenderingContext) {
      try {
        this.binding = new XRWebGLBinding(session, gl);
        this.grabber = new XrCameraGrabber(this.options.stage.renderer);
      } catch (err) {
        console.warn('WebXR camera-access unavailable, will use fallback lighting', err);
      }
    }

    if (typeof session.requestLightProbe === 'function') {
      void session
        .requestLightProbe()
        .then((probe) => {
          if (!this.ended) this.lightProbe = probe;
        })
        .catch(() => undefined);
    }
  }

  private onSelect = (e: XRInputSourceEvent): void => {
    const refSpace = this.options.stage.renderer.xr.getReferenceSpace();
    if (refSpace === null) return;
    const rayPose = e.frame.getPose(e.inputSource.targetRaySpace, refSpace);
    if (rayPose === undefined) return;
    const ray = rayFromPose(rayPose.transform);

    if (this.reticle.group.visible) {
      const reticlePos = poseInfo(this.reticle.group.matrix.elements).position;
      if (rayHitsDisc(ray, reticlePos, RETICLE_TAP_RADIUS)) {
        this.placeAtReticle();
        return;
      }
    }
    if (this.placed) {
      const { center, radius } = this.options.stage.objectBounds();
      if (rayHitsSphere(ray, center, Math.max(radius * 1.4, 0.2))) {
        this.unplace();
      }
    }
  };

  placeAtReticle(): boolean {
    if (!this.reticle.group.visible) return false;
    this.tmpPos.copy(poseInfo(this.reticle.group.matrix.elements).position);
    const { stage } = this.options;
    stage.ground.setPose(this.tmpPos);
    stage.setObjectVisible(true);
    if (!this.placed) {
      this.placed = true;
      this.options.onStateChange({ tracking: this.lastTracking, placed: true });
    }
    return true;
  }

  private unplace(): void {
    if (!this.placed) return;
    this.placed = false;
    this.options.stage.setObjectVisible(false);
    this.options.onStateChange({ tracking: this.lastTracking, placed: false });
  }

  step(frame: XRFrame): void {
    const refSpace = this.options.stage.renderer.xr.getReferenceSpace();
    if (this.hitTestSource === null || refSpace === null) return;
    const hits = frame.getHitTestResults(this.hitTestSource);
    let floorPose: XRPose | null = null;
    for (const hit of hits) {
      const pose = hit.getPose(refSpace);
      if (pose !== undefined && poseInfo(pose.transform.matrix).up >= MIN_FLOOR_UP) {
        floorPose = pose;
        break;
      }
    }
    const tracking = floorPose !== null;
    if (floorPose !== null) {
      this.reticle.group.matrix.fromArray(floorPose.transform.matrix);
    }
    this.reticle.group.visible = tracking;
    if (tracking !== this.lastTracking) {
      this.lastTracking = tracking;
      this.options.onStateChange({ tracking, placed: this.placed });
    }
  }

  cameraOrientation(frame: XRFrame): Quaternion | null {
    const refSpace = this.options.stage.renderer.xr.getReferenceSpace();
    if (refSpace === null) return null;
    const o = frame.getViewerPose(refSpace)?.transform.orientation;
    return o === undefined ? null : new Quaternion(o.x, o.y, o.z, o.w);
  }

  grabCameraFrame(frame: XRFrame): FrameSample | null {
    if (this.binding === null || this.grabber === null) return null;
    const refSpace = this.options.stage.renderer.xr.getReferenceSpace();
    if (refSpace === null) return null;
    const pose = frame.getViewerPose(refSpace);
    const cam = pose?.views[0]?.camera;
    if (cam === undefined || cam === null) return null;
    const tex = this.binding.getCameraImage(cam);
    if (tex === null) return null;
    return this.grabber.grab(tex, cam.width, cam.height);
  }

  getLightEstimate(frame: XRFrame): LightingEstimate | null {
    if (this.lightProbe === null) return null;
    const est = frame.getLightEstimate?.(this.lightProbe);
    return est !== undefined && est !== null ? estimateFromWebXrProbe(est) : null;
  }

  get isPlaced(): boolean {
    return this.placed;
  }

  async end(): Promise<void> {
    if (this.ended) return;
    try {
      await this.session.end();
    } catch {
      this.onSessionEnd();
    }
  }

  private onSessionEnd = (): void => {
    if (this.ended) return;
    this.ended = true;
    this.session.removeEventListener('select', this.onSelect);
    this.session.removeEventListener('end', this.onSessionEnd);
    this.hitTestSource?.cancel();
    this.hitTestSource = null;
    this.grabber?.dispose();
    this.grabber = null;
    this.binding = null;
    this.options.stage.scene.remove(this.reticle.group);
    this.reticle.dispose();
    const { stage } = this.options;
    stage.renderer.xr.enabled = false;
    stage.ground.group.visible = true;
    stage.object.group.visible = true;
    this.options.onEnd();
  };
}
