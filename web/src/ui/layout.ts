/**
 * Keeps floating panels from overlapping: publishes the measured height of the dock and the
 * bottom edge of the reading card as CSS variables on #app, which the stylesheets use to place
 * the adjustments panel, the comparison tags and the AR status pill. Heights change with the
 * viewport, the language of the reading and the buttons that are shown, so they are measured
 * rather than assumed.
 */
import { byId } from './dom';

/** Compact layout: narrow phones in portrait, and any short (landscape phone) viewport. */
export const COMPACT_QUERY = '(max-width: 759px), (max-height: 520px)';

export function isCompact(): boolean {
  return typeof window.matchMedia === 'function' && window.matchMedia(COMPACT_QUERY).matches;
}

export function bindLayout(): void {
  const app = byId('app', HTMLDivElement);
  const dock = byId('dock', HTMLElement);
  const reading = byId('reading', HTMLElement);
  const measure = (): void => {
    const dockH = dock.hidden ? 0 : dock.getBoundingClientRect().height;
    const readingBottom = reading.hidden ? 0 : reading.getBoundingClientRect().bottom;
    app.style.setProperty('--dock-h', `${Math.round(dockH)}px`);
    app.style.setProperty('--reading-bottom', `${Math.round(readingBottom)}px`);
  };
  if (typeof ResizeObserver === 'function') {
    const ro = new ResizeObserver(measure);
    ro.observe(dock);
    ro.observe(reading);
  }
  // `hidden` toggles do not always resize the observed box (display: none -> 0 is reported, but
  // be safe): re-measure when attributes change too.
  const mo = new MutationObserver(measure);
  mo.observe(dock, { attributes: true, attributeFilter: ['hidden'] });
  mo.observe(reading, { attributes: true, attributeFilter: ['hidden', 'class'] });
  window.addEventListener('resize', measure);
  measure();
}
