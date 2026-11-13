/**
 * Frame sources: the live camera, a bundled sample clip, or a file the visitor drops in. Every
 * source drives one element that is both the visible backdrop and the input to the estimators.
 *
 * Mirroring: a front ("selfie") camera is shown mirrored, because people expect a mirror. The
 * element's pixels are never mirrored (only its CSS), so estimators always see the sensor image;
 * the estimate is mirrored once, in lighting/estimate.ts forDisplay, using `mirrored` from here.
 */
import { CaptureError, classifyCaptureError, preflightProblem } from './captureErrors';
import type { FrameElement } from './frame';

export type SourceKind = 'camera' | 'sample' | 'file';
export type Facing = 'user' | 'environment';

export interface FrameSource {
  readonly kind: SourceKind;
  readonly element: FrameElement;
  /** True when the element is displayed mirrored left-right. */
  readonly mirrored: boolean;
  /** Short plain-language name, e.g. "Front camera" or "beach.mp4". */
  readonly label: string;
  stop(): void;
}

export interface CameraSource extends FrameSource {
  readonly kind: 'camera';
  readonly facing: Facing | null;
  readonly deviceId: string | null;
}

/**
 * Mirror only the user-facing camera. Laptop webcams usually report no facingMode, and they face
 * the user, so unknown means mirrored. Rear cameras and media files are never mirrored.
 */
export function shouldMirror(kind: SourceKind, facingMode: string | undefined): boolean {
  return kind === 'camera' && facingMode !== 'environment';
}

function cameraLabel(facing: Facing | null): string {
  if (facing === 'environment') return 'Rear camera';
  if (facing === 'user') return 'Front camera';
  return 'Camera';
}

export interface CameraRequest {
  facing: Facing;
  deviceId?: string | undefined;
}

function constraintsFor(req: CameraRequest): MediaStreamConstraints {
  const video: MediaTrackConstraints = { width: { ideal: 1280 }, height: { ideal: 720 } };
  if (req.deviceId !== undefined) video.deviceId = { exact: req.deviceId };
  else video.facingMode = { ideal: req.facing };
  return { video, audio: false };
}

/** navigator.mediaDevices is undefined on insecure pages, whatever the DOM types say. */
function mediaDevices(): MediaDevices | undefined {
  return (navigator as Partial<Navigator>).mediaDevices;
}

/** Ask for the camera and start it. Throws CaptureError with a plain-language problem. */
export async function openCamera(req: CameraRequest): Promise<CameraSource> {
  const problem = preflightProblem({
    isSecureContext: window.isSecureContext,
    hasGetUserMedia: typeof mediaDevices()?.getUserMedia === 'function',
  });
  if (problem !== null) throw new CaptureError(problem);
  let stream: MediaStream;
  try {
    stream = await navigator.mediaDevices.getUserMedia(constraintsFor(req));
  } catch (err) {
    console.error('getUserMedia failed', err);
    throw new CaptureError(classifyCaptureError(err), { cause: err });
  }
  const track = stream.getVideoTracks()[0];
  const settings = track?.getSettings() ?? {};
  const facing: Facing | null =
    settings.facingMode === 'user' || settings.facingMode === 'environment'
      ? settings.facingMode
      : null;
  const video = document.createElement('video');
  video.muted = true;
  video.playsInline = true;
  video.autoplay = true;
  video.srcObject = stream;
  await video.play().catch((err: unknown) => {
    console.warn('camera video play() was refused; waiting for data instead', err);
  });
  return {
    kind: 'camera',
    element: video,
    mirrored: shouldMirror('camera', settings.facingMode),
    label: cameraLabel(facing),
    facing,
    deviceId: settings.deviceId ?? null,
    stop: () => {
      stream.getTracks().forEach((t) => {
        t.stop();
      });
      video.srcObject = null;
    },
  };
}

/** Number of cameras the browser lists (labels need permission; the count does not). */
export async function countCameras(): Promise<number> {
  if (typeof mediaDevices()?.enumerateDevices !== 'function') return 0;
  const devices = await navigator.mediaDevices.enumerateDevices();
  return devices.filter((d) => d.kind === 'videoinput').length;
}

/** The request that switches away from `current`: the other facing, or the next device. */
export async function nextCameraRequest(current: CameraSource): Promise<CameraRequest> {
  if (current.facing !== null) {
    return { facing: current.facing === 'user' ? 'environment' : 'user' };
  }
  const devices = (await navigator.mediaDevices.enumerateDevices()).filter(
    (d) => d.kind === 'videoinput',
  );
  const i = devices.findIndex((d) => d.deviceId === current.deviceId);
  const next = devices[(i + 1) % Math.max(devices.length, 1)];
  return { facing: 'user', deviceId: next?.deviceId };
}
