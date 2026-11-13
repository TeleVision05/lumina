/**
 * Lumina entry point. See docs/CONVENTIONS.md before touching lighting code and docs/WEB.md for
 * how the app is put together.
 */
import './styles/base.css';
import './styles/layout.css';
import './styles/panels.css';
import { App } from './app/app';
import { attachDebug } from './app/debug';

function start(): void {
  try {
    const app = new App();
    attachDebug(app);
  } catch (err) {
    console.error('Lumina failed to start', err);
    const problem = document.getElementById('problem');
    const title = document.getElementById('problem-title');
    const body = document.getElementById('problem-body');
    if (problem !== null && title !== null && body !== null) {
      title.textContent = 'This browser cannot show the 3D view';
      body.textContent =
        'Lumina needs WebGL 2. Try a current version of Chrome, Edge, Safari or Firefox, and check that hardware acceleration is turned on.';
      problem.hidden = false;
    }
  }
}

start();
