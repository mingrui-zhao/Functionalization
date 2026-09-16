// Recorded metal drawer runners, CC0; source and processing in CREDITS.md.
// Independent of the 3D demo so a model/CDN failure cannot disable mute.
(() => {
const button = document.getElementById('sndToggle');
const projectVideo = document.querySelector('.video video');
const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
const preferenceKey = 'functionalization-rail-muted';
let muted = reducedMotion.matches;
try { muted = localStorage.getItem(preferenceKey) === null
  ? muted : localStorage.getItem(preferenceKey) === 'true'; } catch { /* Private browsing. */ }

let context, loading, buffers, master;
let voices = [], idleTimer;
let lastY = window.scrollY, lastMove = 0, gestureStart = 0, direction = 1;

function updateButton() {
  const ready = !muted && !!buffers && context?.state === 'running';
  button.innerHTML = ready ? '🔉' : '🔈 <span class="snd-label">Enable sound</span>';
  button.title = button.ariaLabel = ready ? 'Mute rail sound' : 'Enable rail sound';
  button.setAttribute('aria-pressed', String(ready));
}
updateButton();

function allowed() {
  return !muted && !document.hidden &&
    (!projectVideo || projectVideo.paused || projectVideo.ended);
}

// Overlap the ends of the recording for a continuous, click-free loop.
function makeLoop(recording, reverse = false) {
  const input = recording.getChannelData(0).slice();
  if (reverse) input.reverse();
  const overlap = Math.round(recording.sampleRate * 0.08);
  const size = input.length - overlap;
  const loop = context.createBuffer(1, size, recording.sampleRate);
  const output = loop.getChannelData(0);
  output.set(input.subarray(overlap, size));
  for (let i = 0; i < overlap; i++) {
    const mix = 0.5 - 0.5 * Math.cos(Math.PI * i / (overlap - 1));
    output[size - overlap + i] = input[size + i] * (1 - mix) + input[i] * mix;
  }
  return loop;
}

async function unlock(event) {
  if (!event.isTrusted || !allowed()) return;
  const AudioContext = window.AudioContext || window.webkitAudioContext;
  if (!AudioContext) { button.hidden = true; return; }
  try {
    if (!context) {
      context = new AudioContext();
      master = context.createGain();
      master.gain.value = 0;
      master.connect(context.destination);
      context.addEventListener('statechange', updateButton);
    }
    // Called synchronously from a real gesture for Safari/iOS autoplay rules.
    if (context.state === 'suspended') await context.resume();
    if (!buffers && !loading) {
      loading = (async () => {
        // Embedded PCM recording avoids file:// fetch/CORS restrictions.
        const bytes = Uint8Array.from(atob(window.functionalizationRailRecording), c => c.charCodeAt(0));
        const recording = await context.decodeAudioData(bytes.buffer);
        buffers = [makeLoop(recording), makeLoop(recording, true)];
      })();
    }
    await loading;
    updateButton();
  } catch (error) {
    loading = null;
    stop();
    updateButton();
    console.warn('Rail sound could not start:', error);
    button.title = button.ariaLabel = 'Retry rail sound';
  }
}

// A small pitch change suggests direction; both tracks are the real recording.
function start() {
  clearTimeout(idleTimer);
  if (voices.length) return;
  voices = buffers.map((buffer, i) => {
    const source = context.createBufferSource();
    const gain = context.createGain();
    source.buffer = buffer;
    source.loop = true;
    source.playbackRate.value = i ? 0.94 : 1;
    gain.gain.value = 0;
    source.connect(gain).connect(master);
    source.start();
    return { source, gain };
  });
}

function stop() {
  clearTimeout(idleTimer);
  if (!context || !voices.length) return;
  master.gain.setTargetAtTime(0, context.currentTime, 0.045);
  idleTimer = setTimeout(() => {
    for (const { source, gain } of voices) {
      source.stop(); source.disconnect(); gain.disconnect();
    }
    voices = [];
    master.gain.value = 0;
  }, 240);
}

function onScroll() {
  const now = performance.now();
  const dy = window.scrollY - lastY;
  lastY = window.scrollY;
  if (!allowed() || !buffers || context.state !== 'running') { stop(); return; }
  if (Math.abs(dy) < 1) return;
  const elapsed = now - lastMove;
  if (elapsed > 250) gestureStart = now;
  const speed = Math.min(1, Math.abs(dy) / Math.max(16, Math.min(elapsed, 80)) / 1.4);
  lastMove = now;
  direction = dy > 0 ? 1 : -1;
  start();
  const t = context.currentTime;
  // Gentle volume cap, plus a quieter bed during prolonged scrolling.
  const fatigue = now - gestureStart > 6000 ? 0.65 : 1;
  master.gain.setTargetAtTime((0.4 + 0.6 * Math.sqrt(speed)) * fatigue, t, 0.035);
  for (let i = 0; i < voices.length; i++) {
    voices[i].gain.gain.setTargetAtTime((direction > 0) === (i === 0) ? 1 : 0, t, 0.065);
  }
  idleTimer = setTimeout(stop, 100);
}

button.addEventListener('click', async event => {
  // The first click enables audio even if the browser has blocked autoplay.
  const ready = !muted && !!buffers && context?.state === 'running';
  muted = ready;
  try { localStorage.setItem(preferenceKey, String(muted)); } catch { /* Optional. */ }
  updateButton();
  if (muted) { stop(); return; }
  await unlock(event);
  if (!allowed() || !buffers || context.state !== 'running') return;
  // A short real slide confirms activation without requiring another scroll.
  start();
  master.gain.setTargetAtTime(0.85, context.currentTime, 0.035);
  voices[0].gain.gain.setTargetAtTime(1, context.currentTime, 0.035);
  voices[1].gain.gain.setTargetAtTime(0, context.currentTime, 0.035);
  idleTimer = setTimeout(stop, 450);
});
// Scrolling cannot unlock audio in every browser. The first click/tap or key
// press does; wheel is also offered for browsers that permit it.
for (const name of ['pointerdown', 'keydown', 'wheel']) {
  window.addEventListener(name, event => {
    if (!button.contains(event.target)) void unlock(event);
  }, { passive: true });
}
window.addEventListener('scroll', onScroll, { passive: true });
window.addEventListener('pagehide', stop);
window.addEventListener('blur', stop);
document.addEventListener('visibilitychange', () => { if (document.hidden) stop(); });
projectVideo?.addEventListener('play', stop);
})();
