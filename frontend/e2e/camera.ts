import type { Page } from "@playwright/test";

export interface CameraOptions {
  /** Canvas size of the fake rear camera; portrait stills use height > width. */
  width: number;
  height: number;
  /** "works": the torch lights the frame. "absent": the track reports no torch. Omitted: the browser's own track. */
  torch?: "works" | "absent";
  /** Every getUserMedia call rejects with this DOMException name. */
  fail?: "denied" | "notfound";
  /** No camera API at all (`navigator.mediaDevices` is undefined). */
  unsupported?: boolean;
  /**
   * Finger-scan frames: a dark frame that turns bright saturated red with a 1.2 Hz pulse once the torch is on
   * (30 fps). Off: a static scene repainted at 5 fps, for photo intake.
   */
  pulse?: boolean;
}

// Replaces getUserMedia for the page. Each call returns a fresh stream whose frames follow the options above.
export async function stubCamera(page: Page, options: CameraOptions) {
  await page.addInitScript((o: CameraOptions) => {
    if (o.unsupported) {
      Object.defineProperty(navigator, "mediaDevices", { configurable: true, get: () => undefined });
      return;
    }

    const newStream = () => {
      const canvas = document.createElement("canvas");
      canvas.width = o.width;
      canvas.height = o.height;
      canvas.style.cssText = "position:fixed;left:-99999px;top:0";
      document.body.appendChild(canvas);
      const ctx = canvas.getContext("2d")!;
      const started = performance.now();
      let lit = false;

      const paint = () => {
        if (!o.pulse) {
          ctx.fillStyle = "#7a9";
          ctx.fillRect(0, 0, o.width, o.height);
          ctx.fillStyle = "#234";
          ctx.fillRect(o.width * 0.2, o.height * 0.3, o.width * 0.6, o.height * 0.4);
          return;
        }
        if (!lit) {
          ctx.fillStyle = "#000";
          ctx.fillRect(0, 0, o.width, o.height);
          return;
        }
        const s = Math.sin((2 * Math.PI * 1.2 * (performance.now() - started)) / 1000);
        ctx.fillStyle = `rgb(230, ${40 + 6 * s}, ${40 + 3 * s})`;
        ctx.fillRect(0, 0, o.width, o.height);
      };
      paint();
      window.setInterval(paint, o.pulse ? 33 : 200);

      const stream = canvas.captureStream(o.pulse ? 30 : 5);
      const track = stream.getVideoTracks()[0];
      // Pinned: the app reads the same track object back from the stream.
      const tracks = [track];
      Object.assign(stream, { getVideoTracks: () => tracks });
      if (o.torch) {
        Object.assign(track, {
          getCapabilities: () => (o.torch === "works" ? { torch: true } : {}),
          applyConstraints: async (c?: { advanced?: { torch?: boolean }[] }) => {
            const torch = c?.advanced?.[0]?.torch;
            if (typeof torch === "boolean") lit = torch;
          },
        });
      }
      return stream;
    };

    const getUserMedia = async () => {
      if (o.fail === "denied") throw new DOMException("Permission denied", "NotAllowedError");
      if (o.fail === "notfound") throw new DOMException("Requested device not found", "NotFoundError");
      return newStream();
    };
    Object.defineProperty(navigator, "mediaDevices", { configurable: true, value: { getUserMedia } });
  }, options);
}
