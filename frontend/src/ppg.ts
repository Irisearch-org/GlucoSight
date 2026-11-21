import type { PpgCapture } from "./api";

export const PPG_SECONDS = 20;

/** The lit frame must be this much brighter (mean luma, 0–1) than the dark one for the torch to count as on. */
const TORCH_LUMA_DELTA = 0.03;
/** A channel at or above this level (0–1) is clipped. */
const CLIP_LEVEL = 250 / 255;
/** The server discards the first 3 s as settling time; the pulse check judges the same window. */
const SETTLE_SECONDS = 3;
const HR_MIN_BPM = 40;
const HR_MAX_BPM = 180;

export interface FrameInfo {
  elapsed: number;
  fingerOn: boolean;
  wave: number[];
}

export function cameraSupported(): boolean {
  return window.isSecureContext && !!navigator.mediaDevices?.getUserMedia;
}

type RawSample = { t: number; r: number; g: number; b: number };
type Channel = "r" | "g" | "b";
type TorchConstraintSet = MediaTrackConstraintSet & { torch?: boolean };

const sleep = (ms: number) => new Promise<void>((resolve) => window.setTimeout(resolve, ms));

/**
 * POS (plane-orthogonal-to-skin, Wang 2017) on uniformly sampled r/g/b. It uses
 * all three channels, so it detects a pulse more robustly than any one channel;
 * it is only the detector here, the samples sent for glucose stay one raw channel.
 */
function posSignal(rs: number[], gs: number[], bs: number[], fs: number): number[] {
  const n = gs.length;
  const win = Math.max(4, Math.round(fs * 1.6));
  const s1 = gs.map((g, i) => g - bs[i]);
  const s2 = gs.map((g, i) => g + bs[i] - 2 * rs[i]);
  const out = new Array<number>(n).fill(0);
  for (let k = win; k < n; k++) {
    let m1 = 0, m2 = 0;
    for (let i = k - win; i < k; i++) { m1 += s1[i]; m2 += s2[i]; }
    m1 /= win; m2 /= win;
    let v1 = 0, v2 = 0;
    for (let i = k - win; i < k; i++) { v1 += (s1[i] - m1) ** 2; v2 += (s2[i] - m2) ** 2; }
    const d1 = Math.sqrt(v1 / win), d2 = Math.sqrt(v2 / win);
    out[k] = (s1[k] - m1) + (d2 > 1e-12 ? d1 / d2 : 0) * (s2[k] - m2);
  }
  return out;
}

/** Heart rate by autocorrelation over 40–200 bpm lags with parabolic refinement; NaN when there is no periodicity. */
function heartRateBpm(x: number[], fs: number): number {
  const n = x.length;
  if (n < fs * 4) return NaN;
  const mean = x.reduce((a, v) => a + v, 0) / n;
  const s = x.map((v) => v - mean);
  const e0 = s.reduce((a, v) => a + v * v, 0);
  if (e0 <= 0) return NaN;
  const minLag = Math.max(1, Math.floor((fs * 60) / 200));
  const maxLag = Math.min(n - 2, Math.floor((fs * 60) / 40));
  const ac = new Array<number>(maxLag + 2).fill(0);
  for (let lag = minLag; lag <= maxLag + 1 && lag < n; lag++) {
    let acc = 0;
    for (let i = 0; i < n - lag; i++) acc += s[i] * s[i + lag];
    ac[lag] = acc / e0;
  }
  let best = -1, bestLag = minLag;
  for (let lag = minLag; lag <= maxLag; lag++) if (ac[lag] > best) { best = ac[lag]; bestLag = lag; }
  if (best <= 0) return NaN;
  const y0 = ac[bestLag - 1] ?? 0, y1 = ac[bestLag], y2 = ac[bestLag + 1] ?? 0;
  const den = y0 - 2 * y1 + y2;
  const delta = den === 0 ? 0 : (0.5 * (y0 - y2)) / den;
  return (60 * fs) / (bestLag + delta);
}

/**
 * Finger-over-lens PPG capture (the one capture module). Protocol:
 *  1. open the rear camera and switch the torch on, then verify it really lit:
 *     a lens that accepts `torch: true` and ignores it gives a black frame, so the
 *     lit frame must be brighter than the dark one by TORCH_LUMA_DELTA;
 *  2. record the full-frame mean r/g/b, resample onto a uniform grid (the API
 *     rejects >5% timestamp jitter) and keep the brightest channel;
 *  3. check with POS that a 40–180 bpm pulse is present, and count clipped frames.
 */
export class PpgRecorder {
  torchOn = false;
  /** True only when switching the torch on measurably brightened the picture. */
  torchVerified = false;
  private stream: MediaStream | null = null;
  private canvas = document.createElement("canvas");
  private ctx: CanvasRenderingContext2D;
  private samples: RawSample[] = [];
  private wave: number[] = [];
  private recording = false;
  private recordStart = 0;
  private rafId = 0;
  private timeoutId = 0;

  constructor(
    private video: HTMLVideoElement,
    private onFrame: (f: FrameInfo) => void,
    private onDone: (capture: PpgCapture | null, problem?: string) => void,
  ) {
    this.ctx = this.canvas.getContext("2d", { willReadFrequently: true }) as CanvasRenderingContext2D;
  }

  async start(): Promise<void> {
    this.stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: "environment" }, width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 30 } },
      audio: false,
    });
    this.video.srcObject = this.stream;
    await this.video.play().catch(() => undefined);
    await this.verifyTorch(this.stream.getVideoTracks()[0]);
    this.loop();
  }

  private async setTorch(track: MediaStreamTrack, on: boolean): Promise<boolean> {
    try {
      await track.applyConstraints({ advanced: [{ torch: on } as TorchConstraintSet] });
      return true;
    } catch {
      return false;
    }
  }

  /** Lit vs dark mean luma; the camera must still point at the room, not at a finger. */
  private async verifyTorch(track: MediaStreamTrack): Promise<void> {
    const caps = (track.getCapabilities?.() ?? {}) as TorchConstraintSet;
    if (!caps.torch || !(await this.setTorch(track, true))) return;
    await sleep(500);
    const lit = this.meanLuma();
    await this.setTorch(track, false);
    await sleep(500);
    const dark = this.meanLuma();
    this.torchVerified = lit - dark > TORCH_LUMA_DELTA;
    this.torchOn = await this.setTorch(track, true);
  }

  private meanLuma(): number {
    const W = 32, H = 24;
    if (this.canvas.width !== W) { this.canvas.width = W; this.canvas.height = H; }
    this.ctx.drawImage(this.video, 0, 0, W, H);
    const d = this.ctx.getImageData(0, 0, W, H).data;
    let s = 0;
    for (let i = 0; i < d.length; i += 4) s += (d[i] + d[i + 1] + d[i + 2]) / 3;
    return s / (d.length / 4) / 255;
  }

  private meanRGB(): [number, number, number] {
    const W = 64, H = 48;
    if (this.canvas.width !== W) { this.canvas.width = W; this.canvas.height = H; }
    this.ctx.drawImage(this.video, 0, 0, W, H);
    const d = this.ctx.getImageData(0, 0, W, H).data;
    let r = 0, g = 0, b = 0, n = 0;
    for (let i = 0; i < d.length; i += 4) { r += d[i]; g += d[i + 1]; b += d[i + 2]; n++; }
    return [r / n / 255, g / n / 255, b / n / 255];
  }

  private loop = () => {
    const now = performance.now() / 1000;
    const [r, g, b] = this.meanRGB();
    // A finger pressed over a lit lens reads bright and red.
    const fingerOn = r > 0.35;
    if (this.recording) {
      this.samples.push({ t: now, r, g, b });
      this.wave.push(r);
      if (this.wave.length > 150) this.wave.shift();
      this.onFrame({ elapsed: now - this.recordStart, fingerOn, wave: this.wave.slice() });
    } else {
      this.onFrame({ elapsed: 0, fingerOn, wave: [] });
    }
    this.rafId = requestAnimationFrame(this.loop);
  };

  record(): void {
    this.samples = [];
    this.wave = [];
    this.recording = true;
    this.recordStart = performance.now() / 1000;
    this.timeoutId = window.setTimeout(() => this.finish(), PPG_SECONDS * 1000);
  }

  private brightestChannel(): Channel {
    const mean = (k: Channel) => this.samples.reduce((a, p) => a + p[k], 0) / this.samples.length;
    const mr = mean("r"), mg = mean("g"), mb = mean("b");
    return mr >= mg && mr >= mb ? "r" : mg >= mb ? "g" : "b";
  }

  private finish(): void {
    this.recording = false;
    if (this.samples.length < 30) {
      this.onDone(null, `Sinyal terlalu sedikit (${this.samples.length} frame). Ulangi pemindaian.`);
      return;
    }
    const diffs: number[] = [];
    for (let i = 1; i < this.samples.length; i++) diffs.push(this.samples[i].t - this.samples[i - 1].t);
    diffs.sort((a, b) => a - b);
    const fs = Math.round((1 / diffs[Math.floor(diffs.length / 2)]) * 100) / 100;
    const t0 = this.samples[0].t, t1 = this.samples[this.samples.length - 1].t;
    const n = Math.floor((t1 - t0) * fs) + 1;
    const ts = new Array<number>(n);
    const rs = new Array<number>(n), gs = new Array<number>(n), bs = new Array<number>(n);
    let j = 0;
    for (let i = 0; i < n; i++) {
      const t = t0 + i / fs;
      while (j < this.samples.length - 2 && this.samples[j + 1].t < t) j++;
      const a = this.samples[j], b = this.samples[j + 1] ?? a;
      const c = b.t > a.t ? Math.min(1, Math.max(0, (t - a.t) / (b.t - a.t))) : 0;
      ts[i] = t;
      rs[i] = a.r + (b.r - a.r) * c;
      gs[i] = a.g + (b.g - a.g) * c;
      bs[i] = a.b + (b.b - a.b) * c;
    }

    const key = this.brightestChannel();
    const samples = key === "r" ? rs : key === "g" ? gs : bs;
    const clipped = this.samples.filter((p) => p[key] >= CLIP_LEVEL).length;

    const settle = Math.min(n - 1, Math.round(SETTLE_SECONDS * fs));
    const bpm = heartRateBpm(posSignal(rs, gs, bs, fs).slice(settle), fs);
    if (!(bpm >= HR_MIN_BPM && bpm <= HR_MAX_BPM)) {
      this.onDone(null, "Denyut nadi tidak terdeteksi. Pastikan jari menutup lensa dan flash sepenuhnya, lalu tahan tanpa bergerak.");
      return;
    }
    this.onDone({
      samples, fs, timestamps: ts,
      torch_verified: this.torchVerified,
      clip_fraction: clipped / this.samples.length,
    });
  }

  dispose(): void {
    cancelAnimationFrame(this.rafId);
    window.clearTimeout(this.timeoutId);
    this.recording = false;
    this.stream?.getTracks().forEach((t) => t.stop());
    this.video.srcObject = null;
  }
}
