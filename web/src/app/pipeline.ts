/**
 * The lighting pipeline between a frame source and the renderer:
 * grab (un-mirrored, model size) -> estimator (rate-gated, one in flight) -> temporal filter ->
 * display frame (mirrored once if the video is mirrored).
 */
import { FrameGrabber, type FrameSample } from '../capture/frame';
import type { FrameSource } from '../capture/source';
import { ImageStatsEstimator } from '../estimators/imageStats';
import { NeuralEstimator, type NeuralStatus } from '../estimators/neural';
import { StudioEstimator, STUDIO_SH } from '../estimators/studio';
import type { Estimator, EstimatorId } from '../estimators/types';
import { forDisplay, makeEstimate, type LightingEstimate } from '../lighting/estimate';
import { paramsForResponsiveness, VectorOneEuro } from '../lighting/filter';
import { shFromFlat } from '../lighting/sh';
import { luminance } from '../lighting/shBuild';
import type { LuminaEstimator } from './settings';
import { RateGate, Rolling } from './scheduler';

export class LightingPipeline {
  readonly studio = new StudioEstimator();
  readonly stats = new ImageStatsEstimator();
  readonly neural: NeuralEstimator;
  readonly studioEstimate: LightingEstimate = makeEstimate(STUDIO_SH, 1);
  selected: LuminaEstimator = 'neural';
  /** Which estimator produced the current raw estimate. */
  activeId: EstimatorId | null = null;
  lastFrame: FrameSample | null = null;
  /** The last frame the network saw and the raw (unfiltered, sensor-frame) SH it returned. */
  lastNeural: { rgba: Uint8ClampedArray; sh: number[][] } | null = null;
  readonly inferenceMs = new Rolling(120);
  inferenceCount = 0;
  private readonly gate = new RateGate();
  private readonly grabber = new FrameGrabber();
  private readonly filter = new VectorOneEuro(paramsForResponsiveness(0.5));
  private raw: LightingEstimate | null = null;
  private rawDisplayFrame = false;
  private confidence = 0;
  private filtered: LightingEstimate | null = null;
  private generation = 0;

  constructor(onNeuralStatus: (s: NeuralStatus) => void) {
    this.neural = new NeuralEstimator(onNeuralStatus);
  }

  setResponsiveness(r: number): void {
    this.filter.params = paramsForResponsiveness(r);
  }

  /** The estimator that runs now: the network once it is ready, otherwise image statistics. */
  current(): Estimator {
    if (this.selected === 'neural' && this.neural.status.state === 'ready') return this.neural;
    return this.stats;
  }

  /** Call once per rendered frame. Starts an estimate when the rate gate allows. */
  tick(nowMs: number, source: FrameSource): void {
    if (!this.gate.tryStart(nowMs)) return;
    this.run(source)
      .catch((err: unknown) => {
        console.warn('estimate failed', err);
      })
      .finally(() => {
        this.gate.finish();
      });
  }

  private async run(source: FrameSource): Promise<void> {
    const gen = this.generation;
    const frame = await this.grabber.grab(source.element);
    if (frame === null || gen !== this.generation) return;
    this.lastFrame = frame;
    const est = this.current();
    const t0 = performance.now();
    const e = await est.estimate(frame);
    if (gen !== this.generation) return; // the source changed while this frame was in flight
    const done = performance.now();
    const ms = est === this.neural ? this.neural.lastRunMs : done - t0;
    this.inferenceMs.push(ms, done);
    this.inferenceCount++;
    this.raw = e;
    if (est === this.neural) this.lastNeural = { rgba: frame.rgba, sh: e.sh.map((c) => [...c]) };
    this.rawDisplayFrame = est.displayFrame;
    this.activeId = est.id;
  }

  /** Filtered estimate in the display frame, or null before the first estimate. */
  displayEstimate(nowMs: number, mirrored: boolean): LightingEstimate | null {
    const raw = this.raw;
    if (raw === null) return null;
    const scale = Math.max(luminance(raw.sh[0] ?? [0, 0, 0]), 1e-4);
    const out = this.filter.filter(raw.sh.flat(), nowMs / 1000, scale);
    this.confidence += 0.1 * (raw.confidence - this.confidence);
    const est = makeEstimate(shFromFlat(out), this.confidence);
    this.filtered = this.rawDisplayFrame ? est : forDisplay(est, mirrored);
    return this.filtered;
  }

  /** Last value returned by displayEstimate. */
  get latest(): LightingEstimate | null {
    return this.filtered;
  }

  /** Forget the current estimate (new source). */
  reset(): void {
    this.generation++;
    this.raw = null;
    this.filtered = null;
    this.confidence = 0;
    this.filter.reset();
    this.inferenceMs.clear();
    this.lastFrame = null;
    this.lastNeural = null;
  }

  dispose(): void {
    this.neural.dispose();
  }
}
