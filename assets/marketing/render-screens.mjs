// Renders the screenshots that go into the marketing artwork: the PRODUCTION page (ui/dist/index.html, the bytes
// embedded in shape_arrange.py) in headless Chrome, with a MOCK host bridge and made-up data. Same approach as
// ui/scripts/verify-browser.mjs. Nothing here talks to OrcaSlicer, a printer or the network.
//
//   cd ui && npm ci --ignore-scripts && npx vite build && cd ..
//   node assets/marketing/render-screens.mjs [--out assets/marketing/source/screens] [--chrome <path>]
//   uv run --python 3.12 --with numpy --with pillow python assets/marketing/compose.py
import { spawn } from 'node:child_process';
import { createServer } from 'node:http';
import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import assert from 'node:assert/strict';

const args = process.argv.slice(2);
const opt = (name, fallback) => (args.includes(name) ? args[args.indexOf(name) + 1] : fallback);
const here = dirname(fileURLToPath(import.meta.url));
const out = resolve(opt('--out', join(here, 'source', 'screens')));
const chromePath = opt('--chrome', '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome');
// What is rendered: CSS size (captured at 2x), file suffix, states and themes. English and pt-BR each time.
// The sizes follow the empty square each screenshot goes into (about 1.04:1 in landscape artwork, 1.01:1 in the
// 1:1 covers) and how large the artwork is shown: two columns for the store images, one column with larger
// text for the README heroes, the social preview and the 1:1 covers. 1040 wide is the plain README screenshot.
const PLAN = [
  { size: [700, 674], suffix: '', states: ['filled', 'outlines', 'filling', 'arranged'], themes: ['dark'] },
  { size: [790, 761], suffix: '', states: ['advanced'], themes: ['dark'] }, // room for the open Advanced section
  { size: [460, 443], suffix: '-medium', states: ['form'], themes: ['dark', 'light'] },
  { size: [420, 414], suffix: '-small', states: ['form'], themes: ['dark'] },
  { size: [1040, 410], suffix: '-wide', states: ['filled'], themes: ['dark', 'light'] },
  { size: [1040, 730], suffix: '-wide', states: ['advanced'], themes: ['dark', 'light'] },
];
await mkdir(out, { recursive: true });

const html = await readFile(join(here, '..', '..', 'ui', 'dist', 'index.html'));
const CSP = "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data:; font-src data:; connect-src 'none'";
const server = createServer((req, res) => {
  if (req.url.split('?')[0] !== '/panel.html') { res.writeHead(404); res.end('no'); return; }
  res.writeHead(200, { 'content-type': 'text/html; charset=utf-8', 'content-security-policy': CSP });
  res.end(html);
});
await new Promise((r) => server.listen(0, '127.0.0.1', r));
const url = `http://127.0.0.1:${server.address().port}/panel.html`;

const profile = await mkdtemp(join(tmpdir(), 'orca-organizer-marketing-'));
const browser = spawn(chromePath, ['--headless=new', '--disable-extensions', '--use-mock-keychain', '--password-store=basic', '--disable-gpu',
  '--disable-background-networking', '--no-first-run', '--remote-debugging-port=0', `--user-data-dir=${profile}`, 'about:blank'], { stdio: 'ignore' });
const delay = (ms) => new Promise((r) => setTimeout(r, ms));
let socket;
try {
  let port;
  for (let i = 0; i < 1200 && !port; i++) { try { port = (await readFile(join(profile, 'DevToolsActivePort'), 'utf8')).split('\n')[0]; } catch { await delay(25); } }
  assert(port, 'Chrome did not start');
  const target = await (await fetch(`http://127.0.0.1:${port}/json/new?about:blank`, { method: 'PUT' })).json();
  socket = new WebSocket(target.webSocketDebuggerUrl);
  await new Promise((res, rej) => { socket.onopen = res; socket.onerror = rej; });
  let nextId = 0;
  const pending = new Map();
  socket.onmessage = ({ data }) => {
    const m = JSON.parse(data);
    if (m.id && pending.has(m.id)) { const p = pending.get(m.id); pending.delete(m.id); m.error ? p.reject(new Error(JSON.stringify(m.error))) : p.resolve(m.result); }
  };
  const call = (method, params = {}) => new Promise((res, rej) => { const id = ++nextId; pending.set(id, { resolve: res, reject: rej }); socket.send(JSON.stringify({ id, method, params })); });
  const evaluate = async (expression) => {
    const r = await call('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true });
    if (r.exceptionDetails) throw new Error(JSON.stringify(r.exceptionDetails));
    return r.result.value;
  };
  await call('Page.enable'); await call('Runtime.enable');
  await call('Page.addScriptToEvaluateOnNewDocument', { source: 'window.orca = { postMessage: () => {}, onMessage: (f) => { window.__deliver = f; } };' });
  const deliver = (m) => evaluate(`window.__deliver(${JSON.stringify(m)}); new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))`);
  const click = async (sel) => {
    const { x, y } = await evaluate(`(() => { const el = document.querySelector(${JSON.stringify(sel)}); el.scrollIntoView({ block: 'center' }); const r = el.getBoundingClientRect(); return { x: r.left + r.width / 2, y: r.top + r.height / 2 }; })()`);
    for (const type of ['mouseMoved', 'mousePressed', 'mouseReleased']) await call('Input.dispatchMouseEvent', { type, x, y, button: 'left', clickCount: 1 });
    await delay(350);
  };

  // ---- made-up data: a plate of brackets, clips and gears
  const name = 'bracket-set';
  const path = '/example/user/OrcaSlicer/plugins/storage/bracket-set-arranged-20261116-101500-3fa2c1.3mf';
  const ref = { stem: name, plate: 1 };
  const base = (lang) => ({ sync: true, lang, job: null, busy: false, stale: '', stale_ref: null, context: 'x', ctx: { plate: 1, objects: 14, name, problem: null } });
  const result = { ...ref, code: 'done', limit: false, max: 300, tower: false, clearance: null, keep: false, unplaced: [], open: 'asked' };
  const start = async (l, action) => { await deliver(base(l)); await deliver({ job: 'j1', start: true, busy: true, text: 'r', lang: l, msg: { ...ref, action, code: 'reading' } }); };
  const S = {
    form: async (l) => deliver(base(l)),
    advanced: async (l) => { await deliver(base(l)); await click('#adv'); },
    outlines: async (l) => { await start(l, 'arrange'); await deliver({ job: 'j1', busy: true, text: 'p', msg: { ...ref, action: 'arrange', code: 'progress', phase: 'footprint', i: 8, n: 14, count: null, subject: 'Cable clip' } }); },
    filling: async (l) => { await start(l, 'fill'); await deliver({ job: 'j1', busy: true, text: 'p', msg: { ...ref, action: 'fill', code: 'progress', phase: 'fill', i: 17, n: null, count: 17, subject: '' } }); },
    arranged: async (l) => { await start(l, 'arrange'); await deliver({ job: 'j1', busy: false, done: true, path, text: 'd', msg: { ...result, action: 'arrange', placed: 14, added: 0, by_object: {} } }); },
    filled: async (l) => { await start(l, 'fill'); await deliver({ job: 'j1', busy: false, done: true, path, text: 'd', msg: { ...result, action: 'fill', placed: 14, added: 23, by_object: { 'Corner bracket': 6, 'Cable clip': 9, 'Gear 12T': 8 } } }); },
  };

  for (const { size: [width, height], suffix, states, themes } of PLAN) for (const theme of themes) for (const [tag, hostLang] of [['en', 'en_US'], ['pt-BR', 'pt_BR']]) for (const state of states) {
    await call('Emulation.setDeviceMetricsOverride', { width, height, deviceScaleFactor: 2, mobile: false });
    await call('Emulation.setEmulatedMedia', { features: [{ name: 'prefers-color-scheme', value: theme }, { name: 'prefers-reduced-motion', value: 'reduce' }] });
    await call('Page.navigate', { url: url + `?${Date.now()}` });
    for (let i = 0; i < 200; i++) { if (await evaluate('!!window.__deliver && !!document.querySelector("#go")').catch(() => false)) break; await delay(25); }
    await S[state](hostLang);
    await evaluate('document.fonts.ready.then(() => true)');
    await delay(600);
    const shot = await call('Page.captureScreenshot', { format: 'png' });
    const file = `screen-${state}-${theme}-${tag}${suffix}.png`;
    await writeFile(join(out, file), Buffer.from(shot.data, 'base64'));
    console.log(file);
  }
} finally {
  socket?.close();
  browser.kill('SIGTERM');
  await new Promise((r) => (browser.exitCode !== null ? r() : browser.once('exit', r)));
  server.close();
  await rm(profile, { recursive: true, force: true });
}
