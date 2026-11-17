/**
 * WebXR `immersive-ar` session for Android Chrome (and other WebXR AR browsers):
 * - Real-world floor/surface tracking via `XRHitTestSource`
 * - Tap-to-place (`select`) onto the tracked floor plane with contact & directional shadows
 * - Live camera frame readback (`camera-access`) so Lumina's neural lighting estimator runs in AR,
 *   with WebXR `light-estimation` as fallback when raw camera access is unavailable.
 */
import {
  CircleGeometry,
  Group,
  Matrix4,
  Mesh,
  MeshBasicMaterial,
  Quaternion,
  RingGeometry,
  Vector3,
} from 'three';
import type { FrameSample } from '../capture/frame';
import { makeEstimate, type LightingEstimate } from '../lighting/estimate';
import type { Vec3 } from '../lighting/sh';
import type { Stage } from '../render/stage';
import { XrCameraGrabber } from './cameraGrab';

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

const AR_DEFAULT_SCALE = 0.45;

function createReticle(): { group: Group; dispose: () => void } {
  const group = new Group();
  const ringGeo = new RingGeometry(0.14, 0.175, 48).rotateX(-Math.PI / 2);
  const dotGeo = new CircleGeometry(0.022, 24).rotateX(-Math.PI / 2);
  const ringMat = new MeshBasicMaterial({
    color: 0xffffff,
    transparent: true,
    opacity: 0.9,
    depthWrite: false,
  });
  const dotMat = new MeshBasicMaterial({
    color: 0xffffff,
    transparent: true,
    opacity: 0.95,
    depthWrite: false,
  });
  group.add(new Mesh(ringGeo, ringMat), new Mesh(dotGeo, dotMat));
  group.matrixAutoUpdate = false;
  group.visible = false;
  return {
    group,
    dispose: () => {
      ringGeo.dispose();
      dotGeo.dispose();
      ringMat.dispose();
      dotMat.dispose();
    },
  };
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
  private readonly reticle: { group: Group; dispose: () => void };
  private hitTestSource: XRHitTestSource | null = null;
  private binding: XRWebGLBinding | null = null;
  private grabber: XrCameraGrabber | null = null;
  private lightProbe: XRLightProbe | null = null;
  private placed = false;
  private lastTracking = false;
  private ended = false;
  private readonly tmpPos = new Vector3();
  private readonly tmpQuat = new Quaternion();
  private readonly tmpScale = new Vector3();

  private constructor(
    private readonly session: XRSession,
    private readonly options: WebXrOptions,
  ) {
    this.reticle = createReticle();
    options.stage.scene.add(this.reticle.group);
    options.stage.ground.group.visible = false;
    options.stage.object.group.visible = false;

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

  private onSelect = (): void => {
    this.placeAtReticle();
  };

  placeAtReticle(): boolean {
    if (!this.reticle.group.visible) return false;
    const mat = new Matrix4().copy(this.reticle.group.matrix);
    mat.decompose(this.tmpPos, this.tmpQuat, this.tmpScale);
    const { stage } = this.options;
    stage.ground.setPose(this.tmpPos);
    stage.object.group.position.set(0, 0, 0);
    stage.object.group.scale.setScalar(AR_DEFAULT_SCALE);
    stage.ground.placeContact(0, 0, stage.object.radius * 2.8 * AR_DEFAULT_SCALE, 0.32);
    stage.ground.group.visible = true;
    stage.object.group.visible = true;
    if (!this.placed) {
      this.placed = true;
      this.options.onStateChange({ tracking: this.lastTracking, placed: true });
    }
    return true;
  }

  step(frame: XRFrame): void {
    const refSpace = this.options.stage.renderer.xr.getReferenceSpace();
    if (this.hitTestSource === null || refSpace === null) return;
    const hits = frame.getHitTestResults(this.hitTestSource);
    const pose = hits[0]?.getPose(refSpace) ?? null;
    const tracking = pose !== null;
    if (pose !== null) {
      this.reticle.group.matrix.fromArray(pose.transform.matrix);
    }
    this.reticle.group.visible = tracking;
    if (tracking !== this.lastTracking) {
      this.lastTracking = tracking;
      this.options.onStateChange({ tracking, placed: this.placed });
    }
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
