/** Typed element lookup. A missing element is a build bug, so it throws. */
export function byId<T extends HTMLElement>(id: string, type: abstract new () => T): T {
  const el = document.getElementById(id);
  if (!(el instanceof type)) throw new Error(`#${id} is missing or not a ${type.name}`);
  return el;
}

export function svgById<T extends SVGElement>(id: string, type: abstract new () => T): T {
  const el = document.getElementById(id);
  if (!(el instanceof type)) throw new Error(`#${id} is missing or not a ${type.name}`);
  return el;
}

export function prefersReducedMotion(): boolean {
  return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
}

/** Checked value of a radio group by name. */
export function radioValue(name: string): string | null {
  const el = document.querySelector<HTMLInputElement>(`input[name="${name}"]:checked`);
  return el?.value ?? null;
}

export function setRadio(name: string, value: string): void {
  const el = document.querySelector<HTMLInputElement>(`input[name="${name}"][value="${value}"]`);
  if (el !== null) el.checked = true;
}

let toastTimer: number | undefined;

/** Plain-language status message, shown for a few seconds. */
export function showToast(message: string, ms = 7000): void {
  const el = byId('toast', HTMLDivElement);
  el.textContent = message;
  el.hidden = false;
  window.clearTimeout(toastTimer);
  toastTimer = window.setTimeout(() => {
    el.hidden = true;
  }, ms);
}
