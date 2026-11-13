/** Binds the dock, the adjustments panel and the estimator choice to the settings. */
import type { LuminaEstimator, Settings } from '../app/settings';
import type { ObjectId } from '../render/objects';
import type { View } from '../render/stage';
import { byId, radioValue, setRadio } from './dom';

export interface ControlHandlers {
  onChange: (s: Settings) => void;
  onSwitchCamera: () => void;
  onChangeSource: () => void;
  onDeviceTilt: (on: boolean) => void;
}

const RANGES = [
  ['set-responsiveness', 'responsiveness'],
  ['set-shadow', 'shadow'],
  ['set-tilt', 'tiltDeg'],
  ['set-grain', 'grain'],
  ['set-blur', 'blur'],
] as const;

function bindSettingsPanel(): void {
  const toggle = byId('settings-toggle', HTMLButtonElement);
  const panel = byId('settings', HTMLElement);
  const setOpen = (open: boolean): void => {
    panel.hidden = !open;
    toggle.setAttribute('aria-expanded', String(open));
  };
  toggle.addEventListener('click', () => {
    setOpen(panel.hidden !== false);
  });
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && !panel.hidden) {
      setOpen(false);
      toggle.focus();
    }
  });
  document.addEventListener('pointerdown', (e) => {
    const t = e.target as Node | null;
    if (!panel.hidden && t !== null && !panel.contains(t) && !toggle.contains(t)) setOpen(false);
  });
}

/** Returns a function that pushes programmatic changes (fallbacks, device tilt) into the UI. */
export function bindControls(
  initial: Settings,
  h: ControlHandlers,
): (patch: Partial<Settings>) => void {
  let s = { ...initial };
  const emit = (): void => {
    h.onChange({ ...s });
  };
  const sync = (): void => {
    setRadio('view', s.view);
    setRadio('object', s.object);
    setRadio('estimator', s.estimator);
    for (const [id, key] of RANGES) byId(id, HTMLInputElement).value = String(s[key]);
    byId('set-device-tilt', HTMLInputElement).checked = s.deviceTilt;
    byId('set-env-camera', HTMLInputElement).checked = s.envFromCamera;
    byId('set-tilt', HTMLInputElement).disabled = s.deviceTilt;
  };
  const onRadio = (name: string, apply: (v: string) => void): void => {
    document.querySelectorAll<HTMLInputElement>(`input[name="${name}"]`).forEach((el) => {
      el.addEventListener('change', () => {
        const v = radioValue(name);
        if (v !== null) apply(v);
        emit();
      });
    });
  };
  onRadio('view', (v) => (s.view = v as View));
  onRadio('object', (v) => (s.object = v as ObjectId));
  onRadio('estimator', (v) => (s.estimator = v as LuminaEstimator));
  for (const [id, key] of RANGES) {
    byId(id, HTMLInputElement).addEventListener('input', (e) => {
      s = { ...s, [key]: Number((e.target as HTMLInputElement).value) };
      emit();
    });
  }
  byId('set-env-camera', HTMLInputElement).addEventListener('change', (e) => {
    s.envFromCamera = (e.target as HTMLInputElement).checked;
    emit();
  });
  byId('set-device-tilt', HTMLInputElement).addEventListener('change', (e) => {
    h.onDeviceTilt((e.target as HTMLInputElement).checked);
  });
  byId('switch-camera', HTMLButtonElement).addEventListener('click', h.onSwitchCamera);
  byId('change-source', HTMLButtonElement).addEventListener('click', h.onChangeSource);
  bindSettingsPanel();
  sync();
  return (patch) => {
    s = { ...s, ...patch };
    sync();
    emit();
  };
}
