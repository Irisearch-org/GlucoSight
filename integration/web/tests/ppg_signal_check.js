#!/usr/bin/env node
/*
 * Reproduce the page-side findings in docs/PPG_CAMERA_CAPTURE_FINDINGS.md.
 *
 * Covers:
 *   P1  channel selection — auto picks the brightest channel, because a finger
 *       over the torch transmits red while green sits at the quantisation floor
 *   P1  the dark-frame guard: with DC below 0.05 the AC/DC ratio is inflated by
 *       the denominator and is not comparable to the physiological band
 *
 * The functions are extracted from the shipped page rather than copied, so this
 * exercises integration/web/ppg-capture.html itself:
 *
 *     node integration/web/tests/ppg_signal_check.js
 */
'use strict';

const fs = require('fs');
const path = require('path');

const PAGE = path.join(__dirname, '..', 'ppg-capture.html');
const source = fs.readFileSync(PAGE, 'utf8');

let failures = 0;
function check(label, ok, detail) {
  console.log(`  [${ok ? 'ok  ' : 'FAIL'}] ${label}: ${detail}`);
  if (!ok) failures++;
}

/* ---- pull named function bodies straight out of the page -------------- */

function bodyEnd(src, open) {
  let depth = 0;
  for (let i = open; i < src.length; i++) {
    const c = src[i];
    if (c === '/' && src[i + 1] === '/') { while (i < src.length && src[i] !== '\n') i++; continue; }
    if (c === '/' && src[i + 1] === '*') { i += 2; while (i < src.length && !(src[i] === '*' && src[i + 1] === '/')) i++; i++; continue; }
    if (c === '"' || c === "'" || c === '`') {
      const q = c; i++;
      while (i < src.length) { if (src[i] === '\\') { i += 2; continue; } if (src[i] === q) break; i++; }
      continue;
    }
    if (c === '{') depth++;
    else if (c === '}' && --depth === 0) return i;
  }
  throw new Error('unbalanced braces');
}

function extract(src, name) {
  const start = src.indexOf(`function ${name}(`);
  if (start < 0) throw new Error(`function ${name}() not found in the page`);
  const open = src.indexOf('{', start);
  return src.slice(start, bodyEnd(src, open) + 1);
}

const NAMES = ['meanOf', 'channelKey', 'detrend', 'dcLevel', 'goertzelAmp', 'estimateHR', 'driftAndPulse'];
const code = NAMES.map(n => extract(source, n)).join('\n');
const sandbox = new Function(`${code}\nreturn { ${NAMES.join(', ')} };`)();
const { channelKey, driftAndPulse } = sandbox;

/* ---- deterministic synthetic frames ---------------------------------- */

const FS = 30.3;
const SECONDS = 34;
const N = Math.round(FS * SECONDS);

function lcg(seed) {                 // deterministic, so results are stable
  let s = seed >>> 0;
  return () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296 - 0.5; };
}

function series(dc, pulseAmp, noise, seed) {
  const rnd = lcg(seed), out = [];
  for (let i = 0; i < N; i++) {
    const t = i / FS;
    out.push(dc + pulseAmp * Math.sin(2 * Math.PI * 1.0 * t) + noise * rnd());
  }
  return out;
}

// A finger clamped over the torch: the frame is red because haemoglobin
// absorbs green. Recorded DCs from the findings document.
const greenFrame = series(0.0021, 0.00012, 0.0003, 1);
const redFrame = series(0.5501, 0.0040, 0.0004, 2);

/* ---- P1: channel selection ------------------------------------------- */

console.log('\nP1  channel selection (finger over the torch)');

const recs = Array.from({ length: N }, (_, i) => ({
  r: redFrame[i], g: greenFrame[i], b: 0.012, t: i / FS,
}));
check('auto picks red (the brightest channel)', channelKey(recs, 'auto') === 'r',
  `channelKey(..., 'auto') = '${channelKey(recs, 'auto')}'`);
check('an explicit green selection still returns green', channelKey(recs, 'green') === 'g',
  "the old hardcoded default is what produced the bad numbers");
check('explicit red returns red', channelKey(recs, 'red') === 'r',
  "user override works");

/* ---- P1: the dark-frame guard ---------------------------------------- */

console.log('\nP1  dark-frame inflates AC/DC through the denominator');

const green = driftAndPulse(greenFrame, FS);
const red = driftAndPulse(redFrame, FS);

console.log(`  green: DC ${green.dcMean.toFixed(4)}  AC/DC ${green.acRatio.toFixed(4)}`);
console.log(`  red:   DC ${red.dcMean.toFixed(4)}  AC/DC ${red.acRatio.toFixed(4)}`);

check('green DC is below the 0.05 dark-frame gate', green.dcMean < 0.05,
  `DC ${green.dcMean.toFixed(4)} — the ratio is not comparable to blood's band`);
check('green AC/DC is above the physiological band', green.acRatio > 0.03,
  `${green.acRatio.toFixed(3)} (> 0.03): manufactured by a tiny denominator`);
check('red DC is above the 0.05 gate', red.dcMean >= 0.05,
  `DC ${red.dcMean.toFixed(4)}`);
check('red AC/DC is physiological', red.acRatio >= 0.002 && red.acRatio <= 0.03,
  `${red.acRatio.toFixed(4)} within 0.002-0.03`);

/* ---- the guard's thresholds must still be what the document quotes --- */

console.log('\nGuard thresholds present in the shipped page');
for (const literal of ['dcm < 0.05', 'a < 0.002', 'a <= 0.03']) {
  check(`page contains \`${literal}\``, source.includes(literal),
    'the document and the implementation agree on the bands');
}

console.log();
if (failures) { console.log(`${failures} check(s) FAILED`); process.exit(1); }
console.log('all checks passed');
