/** The start card: choose a source, see plain-language problems, drop a file anywhere. */
import { MEDIA_COPY, sampleClips } from '../capture/media';
import { byId } from './dom';

export interface IntroHandlers {
  camera: () => void;
  sample: () => void;
  file: (f: File) => void;
}

export interface IntroView {
  showProblem: (title: string, body: string) => void;
  clearProblem: () => void;
  setBusy: (busy: boolean) => void;
}

function bindDrop(onFile: (f: File) => void): void {
  const app = byId('app', HTMLDivElement);
  let depth = 0;
  window.addEventListener('dragenter', (e) => {
    if (!e.dataTransfer?.types.includes('Files')) return;
    depth++;
    app.classList.add('is-dragging');
  });
  window.addEventListener('dragleave', () => {
    depth = Math.max(0, depth - 1);
    if (depth === 0) app.classList.remove('is-dragging');
  });
  window.addEventListener('dragover', (e) => {
    e.preventDefault();
  });
  window.addEventListener('drop', (e) => {
    e.preventDefault();
    depth = 0;
    app.classList.remove('is-dragging');
    const f = e.dataTransfer?.files[0];
    if (f !== undefined) onFile(f);
  });
}

export function bindIntro(h: IntroHandlers): IntroView {
  const problem = byId('problem', HTMLDivElement);
  const buttons = [
    byId('start-camera', HTMLButtonElement),
    byId('start-sample', HTMLButtonElement),
  ];
  const view: IntroView = {
    showProblem: (title, body) => {
      byId('problem-title', HTMLHeadingElement).textContent = title;
      byId('problem-body', HTMLParagraphElement).textContent = body;
      problem.hidden = false;
    },
    clearProblem: () => {
      problem.hidden = true;
    },
    setBusy: (busy) => {
      for (const b of buttons) b.toggleAttribute('aria-busy', busy);
      byId('start-camera', HTMLButtonElement).textContent = busy ? 'Starting…' : 'Use my camera';
    },
  };
  const sample = byId('start-sample', HTMLButtonElement);
  if (sampleClips().length === 0) {
    // No clip in this build: no dead button, just say so next to the other choices.
    sample.hidden = true;
    byId('intro-note', HTMLParagraphElement).textContent =
      `${MEDIA_COPY.noSamples} You can also drop an image or video anywhere on this page.`;
  }
  byId('start-camera', HTMLButtonElement).addEventListener('click', h.camera);
  sample.addEventListener('click', h.sample);
  const input = byId('file-input', HTMLInputElement);
  input.addEventListener('change', () => {
    const f = input.files?.[0];
    if (f !== undefined) h.file(f);
    input.value = '';
  });
  bindDrop(h.file);
  return view;
}
