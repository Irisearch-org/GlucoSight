import type { PpgCapturePayload } from "./api";

export const PPG_SECONDS = 20;

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

/**
 * Finger-over-lens PPG capture: averages the full frame to r/g/b and
 * resamples onto a uniform grid, matching the engine already proven in
 * integration/web/index.html (the API rejects >5% timestamp jitter).
 */
export class PpgRecorder {
  torchOn = false;
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
    private onDone: (capture: PpgCapturePayload | null, problem?: string) => void,
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
    const track = this.stream.getVideoTracks()[0];
    const caps = (track.getCapabilities?.() ?? {}) as TorchConstraintSet;
    try {
      if (caps.torch) {
        await track.applyConstraints({ advanced: [{ torch: true } as TorchConstraintSet] });
        this.torchOn = true;
      }
    } catch {
      /* torch not controllable on this device/browser */
    }
    this.loop();
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
    // A finger pressed over a lit lens reads bright and near-saturated; a
    // simplified stand-in for /ppg's fuller finger-on heuristic.
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
      this.onDone(null, `Sinyal terlalu sedikit (${this.samples.length} frame) — ulangi pemindaian.`);
      return;
    }
    const diffs: number[] = [];
    for (let i = 1; i < this.samples.length; i++) diffs.push(this.samples[i].t - this.samples[i - 1].t);
    diffs.sort((a, b) => a - b);
    const fs = Math.round((1 / diffs[Math.floor(diffs.length / 2)]) * 100) / 100;
    const t0 = this.samples[0].t, t1 = this.samples[this.samples.length - 1].t;
    const n = Math.floor((t1 - t0) * fs) + 1;
    const key = this.brightestChannel();
    const ts = new Array<number>(n), vals = new Array<number>(n);
    let j = 0;
    for (let i = 0; i < n; i++) {
      const t = t0 + i / fs;
      while (j < this.samples.length - 2 && this.samples[j + 1].t < t) j++;
      const a = this.samples[j], b = this.samples[j + 1] ?? a;
      const w = b.t > a.t ? (t - a.t) / (b.t - a.t) : 0;
      const c = Math.min(1, Math.max(0, w));
      ts[i] = t;
      vals[i] = a[key] + (b[key] - a[key]) * c;
    }
    this.onDone({ samples: vals, timestamps_seconds: ts, sampling_rate_hz: fs });
  }

  dispose(): void {
    cancelAnimationFrame(this.rafId);
    window.clearTimeout(this.timeoutId);
    this.recording = false;
    this.stream?.getTracks().forEach((t) => t.stop());
    this.video.srcObject = null;
  }
}
