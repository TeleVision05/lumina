/** The "How it works" dialog: what Lumina does, step by step, and what it honestly cannot do. */
import { byId } from './dom';

const CONTENT = `
<h2 id="how-title">How it works</h2>
<ol>
  <li>Each camera frame is cropped to 4:3 and shrunk to 128 by 96 pixels.</li>
  <li>A small convolutional network, trained on synthetic views of CC0 HDRI panoramas from
  Poly Haven, predicts the room’s lighting as nine spherical-harmonic coefficients per colour: a
  smooth description of the light arriving from every direction. It runs in a background worker,
  on WebGPU where available and WebAssembly otherwise, about twelve times a second.</li>
  <li>A temporal filter steadies the estimate while the light is still and lets it move quickly
  when the light changes.</li>
  <li>The estimate becomes a soft ambient light probe plus one directional light, which casts the
  shadow. The directional light’s energy is taken out of the probe so nothing is counted twice,
  and the shadow fades out when the light is flat.</li>
  <li>Glossy objects reflect an environment built from the same estimate, with the camera picture
  placed in front. That reflection is an approximation, not a measurement.</li>
</ol>
<p>Image statistics is the simple alternative reader: the average colour of the picture plus the
direction of its brightest region. Compare it with the network to see what learning adds.</p>
<h2>What it cannot do</h2>
<ul>
  <li>It cannot know how bright the room really is. Your camera adjusts its exposure and white
  balance, so Lumina estimates the light as your camera sees it.</li>
  <li>Nine coefficients only describe soft, broad light. Hard shadows and sharp reflections are
  approximations built from one assumed main light.</li>
  <li>Light from behind the camera is inferred, not seen. Lights out of view and bright windows are
  the hardest cases.</li>
  <li>The object stands on an assumed floor seen from an assumed camera angle. There is no tracking,
  so it does not stay put when the camera moves.</li>
  <li>The training panoramas are mostly outdoors and professionally shot, so rooms lit by a mix of
  lamps are harder.</li>
</ul>
<h2>Privacy</h2>
<p>Your camera picture is processed in this tab and never uploaded. The page loads no third-party
scripts or fonts.</p>`;

export function bindHowItWorks(): void {
  const dialog = byId('how', HTMLDialogElement);
  byId('how-body', HTMLDivElement).innerHTML = CONTENT;
  const opener = byId('how-open', HTMLButtonElement);
  opener.addEventListener('click', () => {
    dialog.showModal();
  });
  dialog.addEventListener('close', () => {
    opener.focus();
  });
  dialog.addEventListener('click', (e) => {
    if (e.target === dialog) dialog.close();
  });
}
