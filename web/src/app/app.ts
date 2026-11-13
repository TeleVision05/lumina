/**
 * Lumina app: wires a frame source, the lighting pipeline, the three.js stage and the interface,
 * and runs the render loop. Each frame: maybe start an estimate, filter, update both light rigs,
 * render the chosen view, refresh the reading card.
 */
import { CaptureError, PROBLEM_COPY } from '../capture/captureErrors';
import { MediaLoadError, openFile, openVideo, sampleClips, sampleUrl } from '../capture/media';
import {
  countCameras,
  nextCameraRequest,
  openCamera,
  type CameraRequest,
  type FrameSource,
} from '../capture/source';
import { NEURAL_COPY, type NeuralStatus } from '../estimators/neural';
import { Stage } from '../render/stage';
import { byId, prefersReducedMotion, showToast } from '../ui/dom';
import { bindControls } from '../ui/controls';
import { bindHowItWorks } from '../ui/howItWorks';
import { bindPlacement, bindSplit } from '../ui/interaction';
import { bindIntro, type IntroView } from '../ui/intro';
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
    this.applySettings(this.settings);
    window.addEventListener('resize', () => {
      this.resize();
    });
    this.resize();
    requestAnimationFrame(this.loop);
    // The estimator is polled on a timer, not per rendered frame, so a slow frame never delays
    // inference. The rate gate inside the pipeline keeps it at TARGET_HZ, one frame in flight.
    window.setInterval(() => {
      if (this.source !== null) this.pipeline.tick(performance.now(), this.source);
    }, 10);
  }

  private applySettings(s: Settings): void {
    this.settings = { ...s, split: this.settings.split };
    this.stage.setObject(s.object);
    this.stage.ground.setTilt((s.tiltDeg * Math.PI) / 180);
    this.pipeline.selected = s.estimator;
    this.pipeline.setResponsiveness(s.responsiveness);
    byId('split', HTMLDivElement).hidden = s.view !== 'split' || this.source === null;
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

  private async setSource(open: () => Promise<FrameSource>): Promise<void> {
    this.intro.clearProblem();
    this.intro.setBusy(true);
    try {
      const next = await open();
      this.stopSource();
      this.mount(next);
    } catch (err) {
      console.error('could not start the source', err);
      if (err instanceof CaptureError) {
        const copy = PROBLEM_COPY[err.problem];
        this.intro.showProblem(copy.title, copy.body);
      } else if (err instanceof MediaLoadError) {
        this.intro.showProblem('That file did not open', err.message);
      } else {
        this.intro.showProblem(PROBLEM_COPY.unknown.title, PROBLEM_COPY.unknown.body);
      }
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

  useCamera(req: CameraRequest): Promise<void> {
    return this.setSource(() => openCamera(req));
  }

  useSample(): Promise<void> {
    const name = sampleClips()[0];
    if (name === undefined) return Promise.resolve();
    return this.setSource(() => openVideo(sampleUrl(name), 'sample', 'Sample clip'));
  }

  useFile(f: File): Promise<void> {
    return this.setSource(() => openFile(f));
  }

  private async switchCamera(): Promise<void> {
    const current = this.source;
    if (current?.kind !== 'camera') return;
    const req = await nextCameraRequest(current as Parameters<typeof nextCameraRequest>[0]);
    current.stop(); // phones cannot open two cameras at once
    await this.useCamera(req);
    if (this.source === null)
      showToast('The other camera did not start. Choose Use my camera to try again.');
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

  private loop = (): void => {
    requestAnimationFrame(this.loop);
    // performance.now(), not the rAF timestamp: headless and throttled browsers can hand out
    // rAF timestamps that do not track wall time.
    const now = performance.now();
    const interval = this.lastFrame > 0 ? now - this.lastFrame : 0;
    this.lastFrame = now;
    const source = this.source;
    if (source === null) return;
    this.renderFrame(now, source);
    const work = performance.now() - now;
    if (interval > 0) this.frameIntervals.push(interval, now);
    this.frameWork.push(work, now);
    this.stage.recordFrame(now, interval, work);
  };

  /** Update both rigs and render. Exposed for the debug handle (tests read pixels after it). */
  renderFrame(now: number, source: FrameSource): void {
    const s = this.settings;
    const est = this.pipeline.displayEstimate(now, source.mirrored);
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
      const camera =
        s.envFromCamera && frame !== null
          ? {
              rgba: frame.rgba,
              width: frame.width,
              height: frame.height,
              mirrored: source.mirrored,
              vfovDeg: 55,
              weight: 0.85,
            }
          : null;
      this.stage.luminaRig.update(est, { ...base, camera });
    }
    if (!prefersReducedMotion()) this.seed = (this.seed + 0.6180339) % 1;
    const view = est === null ? 'fixed' : s.view;
    this.stage.render(view, this.settings.split, {
      grain: s.grain * MAX_GRAIN,
      blur: s.blur * MAX_BLUR,
      seed: this.seed,
    });
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

  backendLabel(): string {
    const st = this.pipeline.neural.status;
    if (this.pipeline.activeId !== 'neural') return 'main thread';
    return st.state === 'ready' ? st.backend : st.state;
  }
}
