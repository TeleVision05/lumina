/**
 * Camera failures in plain language. The raw error goes to the console; the visitor sees what
 * happened and what to do next.
 */

export type CaptureProblem =
  'insecure' | 'unsupported' | 'denied' | 'not-found' | 'in-use' | 'unknown';

export interface ProblemCopy {
  title: string;
  body: string;
}

export const PROBLEM_COPY: Record<CaptureProblem, ProblemCopy> = {
  insecure: {
    title: 'The camera needs a secure page',
    body: 'Browsers only share a camera with pages served over https. Open the https address of this page, or try the sample clip or your own file instead.',
  },
  unsupported: {
    title: 'This browser cannot share a camera',
    body: 'Try a current version of Chrome, Edge, Safari or Firefox, or use the sample clip or your own file.',
  },
  denied: {
    title: 'Camera access was turned down',
    body: 'To try it live, allow the camera for this site in your browser’s address bar or site settings, then choose Use my camera again. The sample clip and your own files work without it.',
  },
  'not-found': {
    title: 'No camera found',
    body: 'Connect a camera and choose Use my camera again, or use the sample clip or your own file.',
  },
  'in-use': {
    title: 'The camera is busy',
    body: 'Another app or tab may be using it. Close it, then choose Use my camera again.',
  },
  unknown: {
    title: 'The camera did not start',
    body: 'Reload the page and try again, or use the sample clip or your own file.',
  },
};

/** Which plain-language problem an exception from getUserMedia means. */
export function classifyCaptureError(err: unknown): CaptureProblem {
  const name = err instanceof Error || err instanceof DOMException ? err.name : '';
  switch (name) {
    case 'NotAllowedError':
    case 'PermissionDeniedError':
    case 'SecurityError':
      return 'denied';
    case 'NotFoundError':
    case 'DevicesNotFoundError':
    case 'OverconstrainedError':
      return 'not-found';
    case 'NotReadableError':
    case 'TrackStartError':
    case 'AbortError':
      return 'in-use';
    case 'TypeError':
      return 'unsupported';
    default:
      return 'unknown';
  }
}

/** A problem known before asking for the camera, or null when it is worth asking. */
export function preflightProblem(env: {
  isSecureContext: boolean;
  hasGetUserMedia: boolean;
}): CaptureProblem | null {
  if (!env.isSecureContext) return 'insecure';
  if (!env.hasGetUserMedia) return 'unsupported';
  return null;
}

export class CaptureError extends Error {
  constructor(
    public readonly problem: CaptureProblem,
    options?: ErrorOptions,
  ) {
    super(PROBLEM_COPY[problem].title, options);
    this.name = 'CaptureError';
  }
}
