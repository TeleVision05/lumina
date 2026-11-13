/**
 * Ground tilt from the phone's orientation, when the visitor turns it on (iOS asks for permission,
 * which must happen inside the click). Converts the device pitch to "camera looks down by N
 * degrees", smoothed, clamped to the ground-tilt slider's range.
 */

export type TiltResult = 'on' | 'denied' | 'unsupported';

interface OrientationWithPermission {
  requestPermission?: () => Promise<'granted' | 'denied'>;
}

/** Camera pitch below horizontal, degrees, from DeviceOrientationEvent angles and screen angle. */
export function pitchFromOrientation(beta: number, gamma: number, screenAngle: number): number {
  const landscape = Math.abs(screenAngle) === 90;
  const raw = landscape ? 90 - Math.abs(gamma) : 90 - beta;
  return Math.min(Math.max(raw, -10), 60);
}

let handler: ((e: DeviceOrientationEvent) => void) | null = null;

export async function startDeviceTilt(onTilt: (deg: number) => void): Promise<TiltResult> {
  if (typeof DeviceOrientationEvent === 'undefined') return 'unsupported';
  const ctor = DeviceOrientationEvent as unknown as OrientationWithPermission;
  if (typeof ctor.requestPermission === 'function') {
    try {
      if ((await ctor.requestPermission()) !== 'granted') return 'denied';
    } catch (err) {
      console.warn('device orientation permission failed', err);
      return 'denied';
    }
  }
  let smoothed: number | null = null;
  stopDeviceTilt();
  handler = (e) => {
    if (e.beta === null || e.gamma === null) return;
    const angle = typeof screen.orientation === 'undefined' ? 0 : screen.orientation.angle;
    const pitch = pitchFromOrientation(e.beta, e.gamma, angle);
    smoothed = smoothed === null ? pitch : smoothed + 0.15 * (pitch - smoothed);
    onTilt(smoothed);
  };
  window.addEventListener('deviceorientation', handler);
  return 'on';
}

export function stopDeviceTilt(): void {
  if (handler !== null) window.removeEventListener('deviceorientation', handler);
  handler = null;
}
