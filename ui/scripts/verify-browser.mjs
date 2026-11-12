// Exercises the PRODUCTION page (dist/index.html, the same bytes embedded in shape_arrange.py) in an owned
// headless Chrome, under the strict CSP, against a MOCK of the host bridge. Real-time CDP (virtual time breaks
// requestAnimationFrame transitions). Writes MOCK screenshots and measurements into --out.
//
//   node scripts/verify-browser.mjs --out /path/to/evidence [--chrome "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"]
//
// Not Orca: no real WebView, theme, app_language(), clipboard permission or plugin. Those stay with the user.
import { spawn } from 'node:child_process';
import { createHash } from 'node:crypto';
import { createServer } from 'node:http';
import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import assert from 'node:assert/strict';

const args = process.argv.slice(2);
const opt = (name, fallback) => (args.includes(name) ? args[args.indexOf(name) + 1] : fallback);
const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const out = resolve(opt('--out', join(root, 'evidence')));
const chromePath = opt('--chrome', '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome');
await mkdir(out, { recursive: true });

// ---- the artifact under test, and proof that it is the embedded one
const html = await readFile(join(root, 'dist', 'index.html'));
const sha = createHash('sha256').update(html).digest('hex');
const py = await readFile(join(root, '..', 'shape_arrange.py'), 'utf8');
assert.equal(/PANEL_HTML_SHA256 = "([0-9a-f]{64})"/.exec(py)[1], sha, 'dist/index.html is not the embedded page: run npm run build');

const CSP = "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data:; connect-src 'none'";
const requests = [];
const server = createServer((req, res) => {
  requests.push(req.url);
  if (req.url.split('?')[0] === '/panel.html') {
    res.writeHead(200, { 'content-type': 'text/html; charset=utf-8', 'content-security-policy': CSP });
    res.end(html);
  } else { res.writeHead(404); res.end('no'); }
});
await new Promise((r) => server.listen(0, '127.0.0.1', r));
const url = `http://127.0.0.1:${server.address().port}/panel.html`;

// ---- owned Chrome
const profile = await mkdtemp(join(tmpdir(), 'orca-organizer-cdp-'));
const browser = spawn(chromePath, ['--headless=new', '--disable-extensions', '--disable-component-extensions-with-background-pages', '--use-mock-keychain', '--password-store=basic', '--disable-gpu',
  '--disable-background-networking', '--no-first-run', '--remote-debugging-port=0', `--user-data-dir=${profile}`, 'about:blank'],
  { stdio: ['ignore', 'ignore', 'pipe'] });
let chromeLog = '';
browser.stderr.on('data', (c) => { chromeLog += c; });
const delay = (ms) => new Promise((r) => setTimeout(r, ms));
let socket;
const notes = [];
try {
  let port;
  for (let i = 0; i < 1200 && !port; i++) { try { port = (await readFile(join(profile, 'DevToolsActivePort'), 'utf8')).split('\n')[0]; } catch { await delay(25); } }
  assert(port, 'Chrome did not start');
  const target = await (await fetch(`http://127.0.0.1:${port}/json/new?about:blank`, { method: 'PUT' })).json();
  socket = new WebSocket(target.webSocketDebuggerUrl);
  await new Promise((res, rej) => { socket.onopen = res; socket.onerror = rej; });
  let nextId = 0;
  const pending = new Map();
  const events = [];
  socket.onmessage = ({ data }) => {
    const m = JSON.parse(data);
    if (m.id && pending.has(m.id)) { const p = pending.get(m.id); pending.delete(m.id); m.error ? p.reject(new Error(JSON.stringify(m.error))) : p.resolve(m.result); }
    else if (m.method) events.push(m);
  };
  const call = (method, params = {}) => new Promise((resolve_, reject) => { const id = ++nextId; pending.set(id, { resolve: resolve_, reject }); socket.send(JSON.stringify({ id, method, params })); });
  const evaluate = async (expression) => {
    const r = await call('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true });
    if (r.exceptionDetails) throw new Error(JSON.stringify(r.exceptionDetails));
    return r.result.value;
  };
  await call('Page.enable'); await call('Runtime.enable'); await call('Log.enable'); await call('Network.enable');

  // The mock host bridge: what Orca injects. Posts are recorded, messages are delivered by the driver.
  await call('Page.addScriptToEvaluateOnNewDocument', { source: `
    window.__posts = []; window.__violations = [];
    window.orca = { postMessage: (m) => window.__posts.push(m), onMessage: (f) => { window.__deliver = f; } };
    document.addEventListener('securitypolicyviolation', (e) => window.__violations.push(e.violatedDirective + ' ' + e.blockedURI));` });

  async function open({ theme, width, height = 360, scale = 1, language = 'en-US', reducedMotion = false }) {
    await call('Emulation.setDeviceMetricsOverride', { width, height, deviceScaleFactor: scale, mobile: false });
    await call('Emulation.setEmulatedMedia', { features: [{ name: 'prefers-color-scheme', value: theme }, { name: 'prefers-reduced-motion', value: reducedMotion ? 'reduce' : 'no-preference' }] });
    await call('Emulation.setUserAgentOverride', { userAgent: await evaluate('navigator.userAgent'), acceptLanguage: language });
    await call('Page.navigate', { url: url + `?${Date.now()}` });
    for (let i = 0; i < 200; i++) { if (await evaluate('!!window.__deliver && !!document.querySelector("#go")').catch(() => false)) break; await delay(25); }
    assert(await evaluate('!!window.__deliver'), 'page did not register its bridge handler');
  }
  const deliver = (m) => evaluate(`window.__deliver(${JSON.stringify(m)}); new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))`);
  const posts = () => evaluate('window.__posts.splice(0)');
  const active = () => evaluate('(document.activeElement && document.activeElement.id) || (document.activeElement === document.body ? "body" : document.activeElement.tagName)');

  // ---- measurements
  const MEASURE = `(() => {
    const doc = document.documentElement;
    const rgb = (c) => { const m = /rgba?\\(([^)]+)\\)/.exec(c); if (!m) return null; const p = m[1].split(/[ ,\\/]+/).map(Number); return { r: p[0], g: p[1], b: p[2], a: p[3] == null ? 1 : p[3] }; };
    const lum = ({ r, g, b }) => { const f = (v) => { v /= 255; return v <= .03928 ? v / 12.92 : Math.pow((v + .055) / 1.055, 2.4); }; return .2126 * f(r) + .7152 * f(g) + .0722 * f(b); };
    const over = (top, below) => ({ r: top.r * top.a + below.r * (1 - top.a), g: top.g * top.a + below.g * (1 - top.a), b: top.b * top.a + below.b * (1 - top.a), a: 1 });
    const background = (el) => { const stack = []; for (let e = el; e; e = e.parentElement) { const c = rgb(getComputedStyle(e).backgroundColor); if (c && c.a > 0) { stack.push(c); if (c.a === 1) break; } } let base = { r: 255, g: 255, b: 255, a: 1 }; if (!stack.length || stack[stack.length - 1].a < 1) base = rgb(getComputedStyle(document.body).backgroundColor) || base; return stack.reduceRight((acc, c) => over(c, acc), base); };
    const ratio = (a, b) => { const [x, y] = [lum(a), lum(b)].sort((p, q) => q - p); return (x + .05) / (y + .05); };
    const text = [...document.querySelectorAll('body *')].filter((el) => [...el.childNodes].some((n) => n.nodeType === 3 && n.textContent.trim()) && el.getClientRects().length);
    const low = [];
    for (const el of text) { const cs = getComputedStyle(el); const fg = rgb(cs.color); const bg = background(el); const r = ratio(over(fg, bg), bg); const size = parseFloat(cs.fontSize); const large = size >= 24 || (size >= 18.66 && +cs.fontWeight >= 700); if (r < (large ? 3 : 4.5)) low.push({ el: el.id || el.className || el.tagName, ratio: +r.toFixed(2), text: el.textContent.trim().slice(0, 30) }); }
    const offenders = [...document.querySelectorAll('body *')].filter((el) => { const r = el.getBoundingClientRect(); return el.getClientRects().length && (r.right > doc.clientWidth + 0.5 || r.left < -0.5) && !el.closest('.p-select-overlay'); }).slice(0, 6).map((el) => (el.id || el.className || el.tagName) + ':' + Math.round(el.getBoundingClientRect().right));
    const go = document.querySelector('#go'); const gcs = go && getComputedStyle(go);
    const sizes = [...new Set(text.map((el) => getComputedStyle(el).fontSize))].sort();
    const tooSmall = text.filter((el) => parseFloat(getComputedStyle(el).fontSize) < 12).map((el) => el.id || el.className);
    return { viewport: innerWidth + 'x' + innerHeight, dpr: devicePixelRatio, scrollWidth: doc.scrollWidth, clientWidth: doc.clientWidth,
      horizontalOverflow: doc.scrollWidth > doc.clientWidth + 0.5, offenders, domNodes: document.querySelectorAll('*').length,
      theme: doc.dataset.orcaUiTheme, lang: doc.lang, title: document.title, fontFamily: getComputedStyle(document.body).fontFamily,
      fontSizes: sizes, textBelow12px: tooSmall, lowContrast: low,
      goFont: gcs && gcs.fontSize, goRadius: gcs && gcs.borderRadius, goBg: gcs && gcs.backgroundColor, goColor: gcs && gcs.color, goHeight: go && Math.round(go.getBoundingClientRect().height),
      transitionOnGo: gcs && gcs.transitionDuration, violations: window.__violations.slice() };
  })()`;

  // ---- scenarios (the host's real protocol, delivered by the driver)
  const longName = 'Benchy-with-a-really-long-project-file-name-to-check-wrapping';
  const longPath = '/Users/someone/Library/Application Support/OrcaSlicer/plugins/storage/' + longName + '-arranged-20261111-195500-1f3ffe.3mf';
  const ctx = (extra = {}) => ({ plate: 2, objects: 7, name: longName, problem: null, ...extra });
  const base = (lang) => ({ sync: true, lang, job: null, busy: false, stale: '', stale_ref: null, context: 'x' });
  const ref = { stem: longName, plate: 2 };
  const done = { ...ref, action: 'fill', code: 'done', placed: 7, added: 12, by_object: { 'Benchy-hull': 8, 'Cube': 4 }, limit: false, max: 300, tower: true, clearance: 12.5, keep: false, unplaced: ['Part A', 'Part B'], open: 'asked' };
  const S = {
    loading: async () => {},
    idle: async (l) => deliver({ ...base(l), ctx: ctx() }),
    empty: async (l) => deliver({ ...base(l), ctx: ctx({ objects: 0 }) }),
    unreadable: async (l) => deliver({ ...base(l), ctx: ctx({ plate: null, objects: null, name: null, problem: 'no project is open' }) }),
    stale: async (l) => deliver({ ...base(l), ctx: ctx({ plate: 1, name: 'Other' }), stale: 'x', stale_ref: { ...ref, action: 'arrange' }, job: 'j', text: 't', msg: done, path: longPath }),
    validation: async (l) => {  // real keyboard input: select the value, delete it, then press Arrange
      await S.idle(l);
      await evaluate('(() => { const i = document.querySelector("#spacing"); i.focus(); i.select(); })()');
      for (const type of ['keyDown', 'keyUp']) await call('Input.dispatchKeyEvent', { type, key: 'Backspace', code: 'Backspace', windowsVirtualKeyCode: 8 });
      await evaluate('document.querySelector("#go").focus()');
      for (const type of ['keyDown', 'keyUp']) await call('Input.dispatchKeyEvent', { type, key: 'Enter', code: 'Enter', windowsVirtualKeyCode: 13, text: '\r' });
      await delay(150);
    },
    running: async (l) => { await S.idle(l); await deliver({ job: 'j1', start: true, busy: true, text: 'r', lang: l, msg: { ...ref, action: 'arrange', code: 'reading' } });
      await deliver({ job: 'j1', busy: true, text: 'p', msg: { ...ref, action: 'arrange', code: 'progress', phase: 'place', i: 6, n: 14, count: null, subject: 'Benchy-hull copy 6' } }); },
    runningFill: async (l) => { await S.idle(l); await deliver({ job: 'j1', start: true, busy: true, text: 'r', lang: l, msg: { ...ref, action: 'fill', code: 'reading' } });
      await deliver({ job: 'j1', busy: true, text: 'p', msg: { ...ref, action: 'fill', code: 'progress', phase: 'fill', i: 5, n: null, count: 5, subject: '' } }); },
    done: async (l) => { await S.running(l); await deliver({ job: 'j1', busy: false, done: true, path: longPath, text: 'd', msg: done }); },
    error: async (l) => { await S.idle(l); await deliver({ ...base(l), job: 'j2', error: true, text: 'e', msg: { ...ref, action: 'arrange', code: 'failed', cause: { code: 'plate_empty', plate: 2 }, detail: 'x' } }); },
    errorUnknown: async (l) => { await S.idle(l); await deliver({ ...base(l), job: 'j2', error: true, text: 'e', msg: { ...ref, action: 'fill', code: 'failed', cause: { code: 'unexpected' }, detail: 'zipfile.BadZipFile: File is not a zip file' } }); },
    offline: async (l) => { await S.idle(l); await delay(14500); },
  };

  const THEMES = ['light', 'dark'];
  const LANGS = [['en', 'en_US'], ['pt-BR', 'pt_BR']];
  const WIDTHS = [['dock280', { width: 280, scale: 1 }], ['tab800', { width: 800, height: 700, scale: 1 }], ['zoom200-dock', { width: 140, height: 360, scale: 2 }], ['zoom200-tab', { width: 400, height: 700, scale: 2 }]];
  const MATRIX_STATES = ['idle', 'running', 'done', 'error'];
  const EXTRA_STATES = ['loading', 'empty', 'unreadable', 'stale', 'validation', 'runningFill', 'errorUnknown', 'offline'];
  const results = [];
  const reducedMotionWait = 500;
  const shot = async (name, label) => {
    const metrics = await evaluate('(() => ({ w: document.documentElement.scrollWidth, h: document.documentElement.scrollHeight }))()');
    const r = await call('Page.captureScreenshot', { format: 'png', captureBeyondViewport: true, clip: { x: 0, y: 0, width: Math.max(metrics.w, 1), height: Math.min(Math.max(metrics.h, 1), 4000), scale: 1 } });
    await writeFile(join(out, `MOCK-${name}.png`), Buffer.from(r.data, 'base64'));
    return label;
  };
  const run = async (state, theme, [langTag, hostLang], [wName, w]) => {
    await open({ theme, language: langTag === 'en' ? 'en-US' : 'pt-BR', ...w });
    if (langTag === 'pt-BR' && state === 'loading') { /* browser language fallback is what is exercised here */ }
    await S[state](state === 'loading' ? undefined : hostLang);
    await delay(reducedMotionWait); // let enter transitions (Message, Accordion) settle before measuring
    const expected = { offline: '#offline', error: '#status-error', errorUnknown: '#status-error', validation: '#spacing-error', stale: '#stale',
      unreadable: '.context-problem', running: '#cancel', runningFill: '#cancel', done: '#result', loading: '[role=progressbar]', empty: '#status' };
    if (expected[state]) assert(await evaluate(`!!document.querySelector(${JSON.stringify(expected[state])})`), `${state}: ${expected[state]} is not on screen`);
    const m = await evaluate(MEASURE);
    const name = `organizer-${state}-${theme}-${langTag}-${wName}`;
    await shot(name);
    results.push({ name, state, theme, lang: langTag, width: wName, ...m });
    process.stdout.write(`${name} overflow=${m.horizontalOverflow} lowContrast=${m.lowContrast.length} violations=${m.violations.length}\n`);
  };

  if (!process.env.ONLY_INTERACTION) {
    for (const theme of THEMES) for (const lang of LANGS) for (const w of WIDTHS) for (const state of MATRIX_STATES) await run(state, theme, lang, w);
    for (const theme of THEMES) for (const lang of LANGS) for (const state of EXTRA_STATES) await run(state, theme, lang, WIDTHS[0]);
    // the widest stress: dock at zoom 200% with the long states
    for (const theme of THEMES) for (const state of ['errorUnknown', 'done', 'offline']) await run(state, theme, LANGS[1], WIDTHS[2]);
  }

  // ---- interaction (real browser focus semantics)
  const interaction = {};
  await open({ theme: 'light', width: 280, language: 'en-US' });
  await S.idle('en_US');
  const stops = []; const rings = [];
  await evaluate('document.body.focus()');
  for (let i = 0; i < 6; i++) {
    await call('Input.dispatchKeyEvent', { type: 'keyDown', key: 'Tab', code: 'Tab', windowsVirtualKeyCode: 9 });
    await call('Input.dispatchKeyEvent', { type: 'keyUp', key: 'Tab', code: 'Tab', windowsVirtualKeyCode: 9 });
    stops.push(await active());
    await delay(350); // outline-color is transitioned by PrimeVue: measure when settled
    rings.push(await evaluate('(() => { const e = document.activeElement; const cs = getComputedStyle(e); const t = e.id === "spacing" ? e : e; return { id: e.id, outline: cs.outlineStyle + " " + cs.outlineWidth + " " + cs.outlineColor, outlineOffset: cs.outlineOffset, border: cs.borderTopColor, boxShadow: cs.boxShadow }; })()'));
  }
  interaction.tabOrder = stops; interaction.focusRings = rings;
  assert.deepEqual(stops.slice(0, 4), ['spacing', 'go', 'fill', 'adv'], 'tab order: spacing, Arrange, Fill, Advanced');
  const focusRing = await evaluate('(() => { const e = document.activeElement; const cs = getComputedStyle(e); return { id: e.id, outline: cs.outlineStyle + " " + cs.outlineWidth + " " + cs.outlineColor, boxShadow: cs.boxShadow }; })()');
  interaction.focusRingOnLastStop = focusRing;
  for (const r of rings.filter((x) => x.id)) {
    const m = /rgba?\((\d+), (\d+), (\d+)/.exec(r.outline);
    assert(m && !(+m[1] === 245 && +m[2] === 246 && +m[3] === 248) && !r.outline.includes(', 0)'), `focus ring of #${r.id} is invisible: ${r.outline}`);
  }

  // focus survives refreshes
  await evaluate('document.querySelector("#spacing").focus()');
  for (let i = 0; i < 5; i++) await deliver({ ...base('en_US'), ctx: ctx({ objects: 7 + i }) });
  interaction.focusAfterPolls = await active();
  assert.equal(interaction.focusAfterPolls, 'spacing');

  // real click on Arrange: focus moves to Cancel; when the job ends, to Open
  await evaluate('window.__posts.length = 0');
  const center = async (sel) => evaluate(`(() => { const el = document.querySelector(${JSON.stringify(sel)}); el.scrollIntoView({ block: 'center' }); const r = el.getBoundingClientRect(); return { x: r.left + r.width / 2, y: r.top + r.height / 2 }; })()`);
  const click = async (sel) => { const { x, y } = await center(sel); for (const type of ['mouseMoved', 'mousePressed', 'mouseReleased']) await call('Input.dispatchMouseEvent', { type, x, y, button: 'left', clickCount: 1 }); await delay(80); };
  await click('#go');
  interaction.postsAfterClick = await posts();
  assert.equal(interaction.postsAfterClick.at(-1).action, 'arrange');
  await deliver({ job: 'j9', start: true, busy: true, text: 'r', msg: { ...ref, action: 'arrange', code: 'reading' } });
  await delay(120);
  interaction.focusWhileRunning = await active();
  assert.equal(interaction.focusWhileRunning, 'cancel', 'focus moves to Cancel when the run starts');
  await deliver({ job: 'j9', busy: false, done: true, path: longPath, text: 'd', msg: { ...done, action: 'arrange' } });
  await delay(120);
  interaction.focusAfterDone = await active();
  assert.equal(interaction.focusAfterDone, 'open', 'focus moves to Open when the result appears');
  // Enter on Open posts open with the job id
  await evaluate('window.__posts.length = 0');
  await call('Input.dispatchKeyEvent', { type: 'keyDown', key: 'Enter', code: 'Enter', windowsVirtualKeyCode: 13, text: '\r' });
  await call('Input.dispatchKeyEvent', { type: 'keyUp', key: 'Enter', code: 'Enter', windowsVirtualKeyCode: 13 });
  interaction.openByKeyboard = (await posts()).filter((p) => p.action !== 'poll');
  assert.deepEqual(interaction.openByKeyboard, [{ action: 'open', job: 'j9' }]);
  // the copy fallback in a real page: clipboard permission is not granted here, so the honest note must show
  interaction.copyButtonShown = await evaluate('!!document.querySelector("#copy")');
  if (interaction.copyButtonShown) { await click('#copy'); await delay(300); interaction.copyNote = await evaluate('document.querySelector("#path-note")?.textContent ?? null'); }

  // keyboard: Advanced, the Select, focus returns to it after choosing
  await open({ theme: 'light', width: 280, language: 'en-US' });
  await S.idle('en_US');
  await click('#adv'); await delay(300);
  interaction.advancedExpanded = await evaluate('document.querySelector("#adv")?.getAttribute("aria-expanded")');
  await evaluate('document.querySelector("#rotation-auto").focus()');
  for (const type of ['keyDown', 'keyUp']) await call('Input.dispatchKeyEvent', { type, key: 'ArrowDown', code: 'ArrowDown', windowsVirtualKeyCode: 40 });
  await delay(250);
  interaction.focusAfterSelect = await active();
  interaction.rotationLabel = await evaluate('document.querySelector("#rotation-keep").checked ? "keep" : "auto"');
  await evaluate('window.__posts.length = 0');
  await click('#go');
  interaction.rotationSent = (await posts()).filter((p) => p.action !== 'poll');
  assert.equal(interaction.rotationSent[0].rotation, 'keep', 'the chosen rotation is sent');
  assert.equal(interaction.focusAfterSelect, 'rotation-keep', 'arrow keys move between the radio buttons');

  // browser language fallback (no host lang): pt-BR browser shows Portuguese
  await open({ theme: 'light', width: 280, language: 'pt-BR' });
  interaction.browserLanguagePt = await evaluate('document.documentElement.lang + "|" + document.querySelector("#context").textContent');
  assert.equal(interaction.browserLanguagePt, 'pt-BR|Lendo o projeto…');
  // reduced motion
  await open({ theme: 'light', width: 280, language: 'en-US', reducedMotion: true });
  await S.idle('en_US');
  interaction.reducedMotion = await evaluate('(() => { const cs = (s) => getComputedStyle(document.querySelector(s)).transitionDuration; return { go: cs("#go"), spacing: cs("#spacing") }; })()');

  const allViolations = results.flatMap((r) => r.violations);
  const consoleProblems = events.filter((e) => (e.method === 'Runtime.exceptionThrown') || (e.method === 'Log.entryAdded' && ['error', 'warning'].includes(e.params.entry.level)) || (e.method === 'Runtime.consoleAPICalled' && ['error', 'warning'].includes(e.params.type)))
    .map((e) => JSON.stringify(e.params).slice(0, 300));
  const summary = {
    artifactSha256: sha, artifactBytes: html.length, csp: CSP, screenshots: results.length,
    overflowCases: results.filter((r) => r.horizontalOverflow).map((r) => r.name),
    lowContrastCases: results.filter((r) => r.lowContrast.length).map((r) => ({ name: r.name, low: r.lowContrast })),
    cspViolations: allViolations, consoleProblems, networkRequestsToServer: [...new Set(requests.map((r) => r.split('?')[0]))],
    interaction,
  };
  await writeFile(join(out, 'results.json'), JSON.stringify({ summary, results }, null, 1));
  await writeFile(join(out, 'summary.json'), JSON.stringify(summary, null, 1));
  console.log(JSON.stringify(summary, null, 1));
} finally {
  socket?.close();
  browser.kill('SIGTERM');
  await new Promise((r) => (browser.exitCode !== null ? r() : browser.once('exit', r)));
  server.close();
  await writeFile(join(out, 'chrome.log'), chromeLog);
  await rm(profile, { recursive: true, force: true });
}
