/**
 * Lumina app: wires a frame source, the lighting pipeline, the three.js stage and the interface,
 * and runs the render loop. Each frame: maybe start an estimate, filter, update both light rigs,
 * render the chosen view, refresh the reading card.
 */
import type { Quaternion } from 'three';
import { detectArSupport, type ArSupport } from '../ar/capabilities';
import { launchQuickLook } from '../ar/quicklook';
import { WebXrArSession, type ArPlacementState } from '../ar/webxr';
import { makeEstimate } from '../lighting/estimate';
import { rotateSH9 } from '../lighting/shRotate';
import { CaptureError, PROBLEM_COPY } from '../capture/captureErrors';
import { MediaLoadError, openFile, openVideo, sampleClips, sampleUrl } from '../capture/media';
import {
  countCameras,
  nextCameraRequest,
  openCamera,
  type CameraRequest,
  type CameraSource,
  type FrameSource,
} from '../capture/source';
import { NEURAL_COPY, type NeuralStatus } from '../estimators/neural';
import { Stage } from '../render/stage';
import { byId, prefersReducedMotion, showToast } from '../ui/dom';
import { bindControls } from '../ui/controls';
import { bindHowItWorks } from '../ui/howItWorks';
import { bindPlacement, bindSplit } from '../ui/interaction';
import { bindIntro, type IntroView } from '../ui/intro';
import { bindLayout } from '../ui/layout';
import { ReadingCard } from '../ui/reading';
import { startDeviceTilt, stopDeviceTilt } from '../ui/tilt';
import { LightingPipeline } from './pipeline';
import { Rolling } from './scheduler';
import { loadSettings, saveSettings, type Settings } from './settings';

const MAX_GRAIN = 0.05;
const MAX_BLUR = 1.2;

export class App {
  settings: Settings = loadSettings();
  readonly stage: Stage;
  readonly pipeline: LightingPipeline;
  source: FrameSource | null = null;
  readonly frameIntervals = new Rolling(120);
  readonly frameWork = new Rolling(120);
  private readonly reading = new ReadingCard();
  private readonly intro: IntroView;
  private readonly update: (patch: Partial<Settings>) => void;
  private arSupport: ArSupport = 'none';
  private xrSession: WebXrArSession | null = null;
  private arReturnToCamera = false;
  private lastFrame = 0;
  private seed = 0;

  constructor() {
    this.stage = new Stage(byId('scene', HTMLCanvasElement));
    this.pipeline = new LightingPipeline((s) => {
      this.onNeuralStatus(s);
    });
    this.intro = bindIntro({
      camera: () => void this.useCamera({ facing: 'user' }),
      sample: () => void this.useSample(),
      file: (f) => void this.useFile(f),
    });
    this.update = bindControls(this.settings, {
      onChange: (s) => {
        this.applySettings(s);
      },
      onSwitchCamera: () => void this.switchCamera(),
      onChangeSource: () => {
        this.stopSource();
      },
      onDeviceTilt: (on) => void this.setDeviceTilt(on),
    });
    bindSplit(this.settings.split, (x) => {
      this.settings.split = x;
      saveSettings(this.settings);
    });
    bindPlacement(this.stage, () => undefined);
    bindHowItWorks();
    bindLayout();
    this.applySettings(this.settings);
    window.addEventListener('resize', () => {
      this.resize();
    });
    this.resize();
    void this.initAr();
    this.stage.renderer.setAnimationLoop(this.loop);
    // The estimator is polled on a timer, not per rendered frame, so a slow frame never delays
    // inference. The rate gate inside the pipeline keeps it at TARGET_HZ, one frame in flight.
    window.setInterval(() => {
      if (this.xrSession === null && this.source !== null) {
        this.pipeline.tick(performance.now(), this.source);
      }
    }, 10);
  }

  private applySettings(s: Settings): void {
    this.settings = { ...s, split: this.settings.split };
    this.stage.setObject(s.object);
    if (this.xrSession === null) this.stage.ground.setTilt((s.tiltDeg * Math.PI) / 180);
    this.pipeline.selected = s.estimator;
    this.pipeline.setResponsiveness(s.responsiveness);
    byId('split', HTMLDivElement).hidden =
      s.view !== 'split' || this.source === null || this.xrSession !== null;
    saveSettings(this.settings);
  }

  private resize(): void {
    const w = window.innerWidth;
    const h = window.innerHeight;
    this.stage.resize(w, h, Math.min(window.devicePixelRatio || 1, 2));
  }

  private onNeuralStatus(s: NeuralStatus): void {
    const note = byId('reading-note', HTMLParagraphElement);
    if (s.state === 'failed') {
      showToast(NEURAL_COPY.failed);
      this.update({ estimator: 'stats' });
      note.textContent = NEURAL_COPY.failed;
      note.hidden = false;
    } else if (s.state === 'ready' && s.placeholder === true) {
      note.textContent = NEURAL_COPY.placeholder;
      note.hidden = false;
    }
  }

  /** Show a problem where the visitor can see it: the start card, or a toast while running. */
  private report(title: string, body: string): void {
    if (this.source === null) this.intro.showProblem(title, body);
    else showToast(`${title}. ${body}`);
  }

  /** Open and show a new source. Returns false (and reports the problem) when it fails. */
  private async setSource(open: () => Promise<FrameSource>, quiet = false): Promise<boolean> {
    this.intro.clearProblem();
    this.intro.setBusy(true);
    try {
      const next = await open();
      this.stopSource();
      this.mount(next);
      return true;
    } catch (err) {
      console.error('could not start the source', err);
      if (quiet) return false;
      if (err instanceof CaptureError) {
        const copy = PROBLEM_COPY[err.problem];
        this.report(copy.title, copy.body);
      } else if (err instanceof MediaLoadError) {
        this.report('That file did not open', err.message);
      } else {
        this.report(PROBLEM_COPY.unknown.title, PROBLEM_COPY.unknown.body);
      }
      return false;
    } finally {
      this.intro.setBusy(false);
    }
  }

  private mount(source: FrameSource): void {
    this.source = source;
    const media = byId('media', HTMLDivElement);
    media.replaceChildren(source.element);
    media.classList.toggle('is-mirrored', source.mirrored);
    this.pipeline.reset();
    byId('app', HTMLDivElement).dataset.phase = 'running';
    for (const id of ['reading', 'dock']) byId(id, HTMLElement).hidden = false;
    byId('switch-camera', HTMLButtonElement).hidden = true;
    if (source.kind === 'camera') {
      void countCameras().then((n) => {
        byId('switch-camera', HTMLButtonElement).hidden = n < 2;
      });
    }
    this.applySettings(this.settings);
    byId('scene', HTMLCanvasElement).focus({ preventScroll: true });
  }

  stopSource(): void {
    this.source?.stop();
    this.source = null;
    byId('media', HTMLDivElement).replaceChildren();
    byId('app', HTMLDivElement).dataset.phase = 'intro';
    for (const id of ['reading', 'dock', 'settings', 'split']) byId(id, HTMLElement).hidden = true;
    byId('start-camera', HTMLButtonElement).focus();
  }

  useCamera(req: CameraRequest): Promise<boolean> {
    return this.setSource(() => openCamera(req));
  }

  useSample(): Promise<boolean> {
    const name = sampleClips()[0];
    if (name === undefined) return Promise.resolve(false);
    return this.setSource(() => openVideo(sampleUrl(name), 'sample', 'Sample clip'));
  }

  useFile(f: File): Promise<boolean> {
    return this.setSource(() => openFile(f));
  }

  private async switchCamera(): Promise<void> {
    const current = this.source;
    if (current?.kind !== 'camera') return;
    const cam = current as CameraSource;
    const req = await nextCameraRequest(cam);
    const back: CameraRequest = {
      facing: cam.facing ?? 'user',
      deviceId: cam.deviceId ?? undefined,
    };
    cam.stop(); // phones cannot open two cameras at once
    if (await this.setSource(() => openCamera(req), true)) return;
    if (await this.setSource(() => openCamera(back), true)) {
      showToast('The other camera did not start, so Lumina went back to the one you were using.');
      return;
    }
    this.stopSource();
    this.intro.showProblem(PROBLEM_COPY.unknown.title, PROBLEM_COPY.unknown.body);
  }

  private async setDeviceTilt(on: boolean): Promise<void> {
    if (!on) {
      stopDeviceTilt();
      this.update({ deviceTilt: false });
      return;
    }
    const result = await startDeviceTilt((deg) => {
      this.stage.ground.setTilt((deg * Math.PI) / 180);
    });
    if (result !== 'on') {
      showToast(
        result === 'denied'
          ? 'Motion access was turned down, so the ground tilt stays manual.'
          : 'This device does not report its tilt, so the ground tilt stays manual.',
      );
    }
    this.update({ deviceTilt: result === 'on' });
  }

  private async initAr(): Promise<void> {
    this.arSupport = await detectArSupport();
    if (this.arSupport === 'none') return;
    const label = this.arSupport === 'webxr' ? 'Place in AR' : 'View in AR';
    for (const id of ['start-ar', 'ar-action']) {
      const btn = byId(id, HTMLButtonElement);
      btn.textContent = label;
      btn.hidden = false;
      btn.addEventListener('click', () => {
        void this.startAr();
      });
    }
    byId('ar-exit', HTMLButtonElement).addEventListener('click', () => {
      void this.xrSession?.end();
    });
  }

  private async startAr(): Promise<void> {
    if (this.arSupport === 'quicklook') {
      try {
        await launchQuickLook(this.stage.object.group, `lumina-${this.settings.object}.usdz`);
      } catch (err) {
        console.error('AR Quick Look failed', err);
        showToast('Could not open AR Quick Look for this object.');
      }
      return;
    }
    if (this.arSupport !== 'webxr' || this.xrSession !== null) return;
    const wasCamera = this.source?.kind === 'camera';
    this.arReturnToCamera = wasCamera;
    if (this.source !== null) {
      this.source.stop();
      this.source = null;
      byId('media', HTMLDivElement).replaceChildren();
    }
    this.pipeline.reset();
    try {
      const appEl = byId('app', HTMLDivElement);
      this.xrSession = await WebXrArSession.start({
        stage: this.stage,
        overlayRoot: appEl,
        onStateChange: (st) => {
          this.onXrStateChange(st);
        },
        onEnd: () => {
          this.onXrEnd();
        },
      });
      appEl.dataset.phase = 'running';
      appEl.dataset.ar = 'webxr';
      for (const id of ['reading', 'dock', 'ar-hud']) byId(id, HTMLElement).hidden = false;
      byId('split', HTMLDivElement).hidden = true;
      this.onXrStateChange({ tracking: false, placed: false });
    } catch (err) {
      console.error('WebXR AR failed', err);
      this.report('AR floor tracking did not start', 'Check camera permission and ARCore support.');
      if (wasCamera) {
        this.arReturnToCamera = false;
        void this.useCamera({ facing: 'environment' });
      }
    }
  }

  private onXrStateChange(st: ArPlacementState): void {
    const status = byId('ar-status', HTMLSpanElement);
    if (!st.tracking) status.textContent = 'Move your phone to find the floor';
    else if (!st.placed) status.textContent = 'Tap the circle to place';
    else status.textContent = 'Tap the circle to move, or the object to pick up';
  }

  private onXrEnd(): void {
    this.xrSession = null;
    const appEl = byId('app', HTMLDivElement);
    delete appEl.dataset.ar;
    byId('ar-hud', HTMLDivElement).hidden = true;
    this.stage.restoreFromXR(this.settings.tiltDeg);
    if (this.arReturnToCamera) {
      this.arReturnToCamera = false;
      void this.useCamera({ facing: 'environment' });
    } else if (this.source === null) {
      this.stopSource();
    }
  }

  private loop = (_time: number, frame?: XRFrame): void => {
    // performance.now(), not the rAF timestamp: headless and throttled browsers can hand out
    // rAF timestamps that do not track wall time.
    const now = performance.now();
    const interval = this.lastFrame > 0 ? now - this.lastFrame : 0;
    this.lastFrame = now;
    if (this.xrSession !== null && frame !== undefined) {
      this.renderXrFrame(now, frame);
    } else if (this.source !== null) {
      this.renderFrame(now, this.source);
    } else {
      return;
    }
    const work = performance.now() - now;
    if (interval > 0) this.frameIntervals.push(interval, now);
    this.frameWork.push(work, now);
    this.stage.recordFrame(now, interval, work);
  };

  private renderXrFrame(now: number, frame: XRFrame): void {
    const xr = this.xrSession;
    if (xr === null) return;
    xr.step(frame);
    this.pipeline.tickFrame(now, () => xr.grabCameraFrame(frame));
    if (this.pipeline.lastFrame === null) {
      const probeEst = xr.getLightEstimate(frame);
      if (probeEst !== null) this.pipeline.setExternalEstimate(probeEst);
    }
    const worldRot = this.pipeline.lastFrame !== null ? xr.cameraOrientation(frame) : null;
    const est = this.updateRigs(now, false, worldRot);
    const view = est === null ? 'fixed' : this.settings.view;
    this.stage.renderXR(view);
    this.updateReading(now, est);
  }

  private updateRigs(
    now: number,
    mirrored: boolean,
    worldRot: Quaternion | null = null,
  ): ReturnType<LightingPipeline['displayEstimate']> {
    const s = this.settings;
    const est = this.pipeline.displayEstimate(now, mirrored);
    const { center, radius } = this.stage.objectBounds();
    const base = {
      groundNormal: this.stage.ground.normal,
      center,
      radius,
      shadowScale: s.shadow,
      needsEnv: this.stage.object.envMaterials.length > 0,
      sceneKey: this.stage.sceneKey(),
      nowMs: now,
    };
    this.stage.fixedRig.update(this.pipeline.studioEstimate, { ...base, camera: null });
    const frame = this.pipeline.lastFrame;
    if (est !== null) {
      const rigEst =
        worldRot !== null ? makeEstimate(rotateSH9(est.sh, worldRot), est.confidence) : est;
      const camera =
        worldRot === null && s.envFromCamera && frame !== null
          ? {
              rgba: frame.rgba,
              width: frame.width,
              height: frame.height,
              mirrored,
              vfovDeg: 55,
              weight: 0.85,
            }
          : null;
      this.stage.luminaRig.update(rigEst, { ...base, camera });
    }
    return est;
  }

  private updateReading(now: number, est: ReturnType<LightingPipeline['displayEstimate']>): void {
    this.reading.update(now, est, {
      inferenceMs: this.pipeline.inferenceMs.median,
      inferencePerSecond: this.pipeline.inferenceMs.rate(),
      backend: this.backendLabel(),
      fps: (() => {
        const m = this.frameIntervals.median;
        return m === null || m <= 0 ? null : 1000 / m;
      })(),
    });
  }

  /** Update both rigs and render. Exposed for the debug handle (tests read pixels after it). */
  renderFrame(now: number, source: FrameSource): void {
    const s = this.settings;
    const est = this.updateRigs(now, source.mirrored);
    if (!prefersReducedMotion()) this.seed = (this.seed + 0.6180339) % 1;
    const view = est === null ? 'fixed' : s.view;
    this.stage.render(view, this.settings.split, {
      grain: s.grain * MAX_GRAIN,
      blur: s.blur * MAX_BLUR,
      seed: this.seed,
    });
    this.updateReading(now, est);
  }

  backendLabel(): string {
    const st = this.pipeline.neural.status;
    if (this.pipeline.activeId !== 'neural') return 'main thread';
    return st.state === 'ready' ? st.backend : st.state;
  }
}
