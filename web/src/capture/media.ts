/**
 * Sources from media files: the bundled sample clips and files the visitor drops or picks. Files
 * stay on the device (object URLs); nothing is uploaded.
 */
import type { FrameSource, SourceKind } from './source';

export class MediaLoadError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, options);
    this.name = 'MediaLoadError';
  }
}

export const MEDIA_COPY = {
  notMedia: 'That file is not an image or a video. Choose a JPEG, PNG, WebP, MP4 or WebM file.',
  videoFailed:
    'This video could not be played in this browser. Try an MP4 or WebM file, or the sample clip.',
  imageFailed: 'This image could not be opened. Try a JPEG, PNG or WebP file.',
  noSamples: 'No sample clip is included in this build. Use your camera or open your own file.',
} as const;

/** Sample clips found in public/samples/ at build time (see vite/plugins.ts). */
export function sampleClips(): readonly string[] {
  return typeof __LUMINA_SAMPLES__ === 'undefined' ? [] : __LUMINA_SAMPLES__;
}

export function sampleUrl(name: string, base: string = import.meta.env.BASE_URL): string {
  return `${base}samples/${encodeURIComponent(name)}`;
}

/** Start a looping, muted video from a URL. Resolves once the first frame is available. */
export async function openVideo(
  url: string,
  kind: SourceKind,
  label: string,
): Promise<FrameSource> {
  const video = document.createElement('video');
  video.muted = true;
  video.loop = true;
  video.playsInline = true;
  video.autoplay = true;
  video.preload = 'auto';
  const loaded = new Promise<void>((resolve, reject) => {
    video.addEventListener('loadeddata', () => {
      resolve();
    });
    video.addEventListener('error', () => {
      reject(new MediaLoadError(MEDIA_COPY.videoFailed, { cause: video.error }));
    });
  });
  video.src = url;
  await loaded;
  await video.play().catch((err: unknown) => {
    console.warn('video play() was refused', err);
  });
  return {
    kind,
    element: video,
    mirrored: false,
    label,
    stop: () => {
      video.pause();
      video.removeAttribute('src');
      video.load();
      if (url.startsWith('blob:')) URL.revokeObjectURL(url);
    },
  };
}

async function openImage(url: string, label: string): Promise<FrameSource> {
  const img = new Image();
  img.decoding = 'async';
  img.src = url;
  try {
    await img.decode();
  } catch (err) {
    URL.revokeObjectURL(url);
    throw new MediaLoadError(MEDIA_COPY.imageFailed, { cause: err });
  }
  return {
    kind: 'file',
    element: img,
    mirrored: false,
    label,
    stop: () => {
      URL.revokeObjectURL(url);
    },
  };
}

/** Open a dropped or picked file as a source. */
export async function openFile(file: File): Promise<FrameSource> {
  const isImage = file.type.startsWith('image/');
  const isVideo = file.type.startsWith('video/');
  if (!isImage && !isVideo) throw new MediaLoadError(MEDIA_COPY.notMedia);
  const url = URL.createObjectURL(file);
  if (isImage) return openImage(url, file.name);
  try {
    return await openVideo(url, 'file', file.name);
  } catch (err) {
    URL.revokeObjectURL(url);
    throw err;
  }
}
