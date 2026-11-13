/**
 * The light reading card: a grey ball lit by the estimate (CPU-rendered with the same maths as the
 * Python reference), an arrow toward the light, colour temperature on a warm-to-cool scale,
 * confidence, inference time and frame rate. Updated a few times a second, not every frame, so it
 * stays readable.
 */
import { colorTemperature } from '../lighting/colorTemperature';
import type { LightingEstimate } from '../lighting/estimate';
import { deriveLightRig } from '../lighting/lightRig';
import { irradiance, type RGB, type Vec3 } from '../lighting/sh';
import { renderProbe } from '../lighting/probeImage';
import { linearToSrgb8 } from '../lighting/srgb';
import { byId, svgById } from './dom';
import {
  describeDirection,
  describeQuality,
  formatKelvin,
  formatMs,
  kelvinPosition,
} from './describe';

export interface ReadingStats {
  inferenceMs: number | null;
  backend: string;
  inferencePerSecond: number | null;
  fps: number | null;
}

const UPDATE_MS = 200;

function css(c: RGB): string {
  return `rgb(${linearToSrgb8(c[0])} ${linearToSrgb8(c[1])} ${linearToSrgb8(c[2])})`;
}

export class ReadingCard {
  private readonly probe = byId('probe', HTMLCanvasElement);
  private readonly ctx: CanvasRenderingContext2D | null;
  private last = Number.NEGATIVE_INFINITY;
  private readonly size: number;

  constructor() {
    this.size = Math.round(60 * Math.min(window.devicePixelRatio || 1, 2));
    this.probe.width = this.size;
    this.probe.height = this.size;
    this.ctx = this.probe.getContext('2d');
    const card = byId('reading', HTMLElement);
    const expand = byId('reading-expand', HTMLButtonElement);
    expand.addEventListener('click', () => {
      const open = card.classList.toggle('is-open');
      expand.setAttribute('aria-expanded', String(open));
      expand.textContent = open ? 'Less' : 'Details';
    });
  }

  update(nowMs: number, est: LightingEstimate | null, stats: ReadingStats): void {
    if (nowMs - this.last < UPDATE_MS) return;
    this.last = nowMs;
    this.updateStats(est, stats);
    if (est === null) return;
    if (this.ctx !== null) {
      const img = new ImageData(renderProbe(est.sh, this.size, 0.8), this.size, this.size);
      this.ctx.putImageData(img, 0, 0);
    }
    this.updateArrow(est);
    byId('reading-direction', HTMLParagraphElement).textContent = describeDirection(est.direction);
    byId('reading-quality', HTMLParagraphElement).textContent = describeQuality(est.directionality);
    const rig = deriveLightRig(est.sh);
    const colour = rig.light?.irradiance ?? est.sh[0] ?? [0, 0, 0];
    const k = colorTemperature(colour);
    byId('kelvin-value', HTMLSpanElement).textContent = formatKelvin(k);
    byId('kelvin-marker', HTMLSpanElement).style.setProperty(
      '--k',
      `${(k === null ? 0.5 : kelvinPosition(k)) * 100}%`,
    );
    this.updateBrandBall(est);
  }

  private updateStats(est: LightingEstimate | null, s: ReadingStats): void {
    byId('stat-confidence', HTMLElement).textContent =
      est === null ? '–' : `${Math.round(est.confidence * 100)}%`;
    const rate = s.inferencePerSecond === null ? '' : `, ${s.inferencePerSecond.toFixed(0)}/s`;
    byId('stat-inference', HTMLElement).textContent =
      s.inferenceMs === null ? '–' : `${formatMs(s.inferenceMs)}${rate}`;
    byId('stat-inference', HTMLElement).title = `Backend: ${s.backend}`;
    byId('stat-fps', HTMLElement).textContent = s.fps === null ? '–' : `${Math.round(s.fps)} fps`;
  }

  private updateArrow(est: LightingEstimate): void {
    const line = svgById('arrow-line', SVGLineElement);
    const tip = svgById('arrow-tip', SVGCircleElement);
    const d = est.direction;
    const len = d === null ? 0 : 30 * Math.hypot(d[0], d[1]) * Math.min(1, est.directionality * 3);
    const ang = d === null ? 0 : Math.atan2(-d[1], d[0]);
    const x = len * Math.cos(ang);
    const y = len * Math.sin(ang);
    line.setAttribute('x2', x.toFixed(1));
    line.setAttribute('y2', y.toFixed(1));
    tip.setAttribute('cx', x.toFixed(1));
    tip.setAttribute('cy', y.toFixed(1));
    tip.classList.toggle('is-behind', d !== null && d[2] < 0);
    tip.style.display = d === null ? 'none' : '';
  }

  /** Light the wordmark's ball with the estimate: highlight where the light is. */
  private updateBrandBall(est: LightingEstimate): void {
    const root = document.documentElement.style;
    const d = est.direction ?? [0, 0, 1];
    const at = (n: Vec3): string => {
      const len = Math.hypot(n[0], n[1], n[2]) || 1;
      const e = irradiance(est.sh, [n[0] / len, n[1] / len, n[2] / len]);
      const k = 0.8 / Math.PI;
      return css([e[0] * k, e[1] * k, e[2] * k]);
    };
    root.setProperty('--ball-hi', at([d[0], d[1], Math.max(d[2], 0.2)]));
    root.setProperty('--ball-mid', at([0, 0, 1]));
    root.setProperty('--ball-lo', at([-d[0], -d[1], 0.2]));
    root.setProperty('--ball-x', `${50 + 30 * d[0]}%`);
    root.setProperty('--ball-y', `${50 - 30 * d[1]}%`);
  }
}
