/**
 * Direct manipulation: the comparison divider, and placing the object (drag to move, pinch or
 * wheel to resize, arrow keys and +/- on the focused canvas).
 */
import type { Placement, Stage } from '../render/stage';
import { byId } from './dom';

export function bindSplit(initial: number, onChange: (x: number) => void): (x: number) => void {
  const split = byId('split', HTMLDivElement);
  const handle = byId('split-handle', HTMLDivElement);
  let value = initial;
  const set = (x: number): void => {
    value = Math.min(Math.max(x, 0), 1);
    split.style.setProperty('--split', `${value * 100}%`);
    const pct = Math.round(value * 100);
    handle.setAttribute('aria-valuenow', String(pct));
    handle.setAttribute('aria-valuetext', `${pct}% fixed light, ${100 - pct}% Lumina`);
    onChange(value);
  };
  handle.addEventListener('pointerdown', (e) => {
    handle.setPointerCapture(e.pointerId);
  });
  handle.addEventListener('pointermove', (e) => {
    if (!handle.hasPointerCapture(e.pointerId)) return;
    const r = split.getBoundingClientRect();
    set((e.clientX - r.left) / r.width);
  });
  handle.addEventListener('keydown', (e) => {
    const step = e.shiftKey ? 0.2 : 0.05;
    const next: Record<string, number> = {
      ArrowLeft: value - step,
      ArrowDown: value - step,
      ArrowRight: value + step,
      ArrowUp: value + step,
      Home: 0,
      End: 1,
    };
    const v = next[e.key];
    if (v === undefined) return;
    e.preventDefault();
    set(v);
  });
  set(initial);
  return set;
}

interface Pointer {
  x: number;
  y: number;
}

function ndc(canvas: HTMLCanvasElement, p: Pointer): [number, number] {
  const r = canvas.getBoundingClientRect();
  return [((p.x - r.left) / r.width) * 2 - 1, 1 - ((p.y - r.top) / r.height) * 2];
}

/** Placement by pointer, wheel and keyboard. Calls onChange with each new placement. */
export function bindPlacement(stage: Stage, onChange: (p: Placement) => void): void {
  const canvas = stage.canvas;
  const pointers = new Map<number, Pointer>();
  let grab: { dx: number; dz: number } | null = null;
  let pinch: { dist: number; scale: number } | null = null;
  const apply = (p: Placement): void => {
    stage.setPlacement(p);
    onChange(stage.placement);
  };
  const spread = (): number => {
    const [a, b] = [...pointers.values()];
    return a && b ? Math.hypot(a.x - b.x, a.y - b.y) : 0;
  };
  canvas.addEventListener('pointerdown', (e) => {
    canvas.setPointerCapture(e.pointerId);
    pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
    if (pointers.size === 2) {
      grab = null;
      pinch = { dist: spread(), scale: stage.placement.scale };
      return;
    }
    const hit = stage.pickGround(...ndc(canvas, { x: e.clientX, y: e.clientY }));
    grab = hit && { dx: stage.placement.x - hit.x, dz: stage.placement.z - hit.z };
  });
  canvas.addEventListener('pointermove', (e) => {
    if (!pointers.has(e.pointerId)) return;
    pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
    if (pinch !== null && pointers.size === 2 && pinch.dist > 0) {
      apply({ ...stage.placement, scale: (pinch.scale * spread()) / pinch.dist });
      return;
    }
    const hit = stage.pickGround(...ndc(canvas, { x: e.clientX, y: e.clientY }));
    if (grab !== null && hit !== null) {
      apply({ ...stage.placement, x: hit.x + grab.dx, z: hit.z + grab.dz });
    }
  });
  const end = (e: PointerEvent): void => {
    pointers.delete(e.pointerId);
    if (pointers.size < 2) pinch = null;
    if (pointers.size === 0) grab = null;
  };
  canvas.addEventListener('pointerup', end);
  canvas.addEventListener('pointercancel', end);
  canvas.addEventListener(
    'wheel',
    (e) => {
      e.preventDefault();
      apply({ ...stage.placement, scale: stage.placement.scale * Math.exp(-e.deltaY * 0.0015) });
    },
    { passive: false },
  );
  canvas.addEventListener('keydown', (e) => {
    const p = stage.placement;
    const moves: Record<string, Placement> = {
      ArrowLeft: { ...p, x: p.x - 0.1 },
      ArrowRight: { ...p, x: p.x + 0.1 },
      ArrowUp: { ...p, z: p.z - 0.15 },
      ArrowDown: { ...p, z: p.z + 0.15 },
      '+': { ...p, scale: p.scale * 1.1 },
      '=': { ...p, scale: p.scale * 1.1 },
      '-': { ...p, scale: p.scale / 1.1 },
    };
    const next = moves[e.key];
    if (next === undefined) return;
    e.preventDefault();
    apply(next);
  });
}
