import { UserError } from "./api";

export function frameToDataUrl(video: HTMLVideoElement): string | null {
  if (!video.videoWidth || !video.videoHeight) return null;
  const canvas = document.createElement("canvas");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const ctx = canvas.getContext("2d");
  if (!ctx) return null;
  ctx.drawImage(video, 0, 0);
  return canvas.toDataURL("image/jpeg", 0.85);
}

const MAX_EDGE = 1280;

/** Gallery photos over MAX_EDGE on the long edge are re-encoded as JPEG 0.85 at that size; smaller ones pass through unchanged. */
export async function photoToDataUrl(file: File): Promise<string> {
  const original = await readDataUrl(file);
  const img = await loadImage(original);
  const long = Math.max(img.naturalWidth, img.naturalHeight);
  if (long <= MAX_EDGE) return original;
  const scale = MAX_EDGE / long;
  const canvas = document.createElement("canvas");
  canvas.width = Math.round(img.naturalWidth * scale);
  canvas.height = Math.round(img.naturalHeight * scale);
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new UserError("Gagal membaca foto.");
  ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
  return canvas.toDataURL("image/jpeg", 0.85);
}

function readDataUrl(file: File): Promise<string> {
  const { promise, resolve, reject } = Promise.withResolvers<string>();
  const reader = new FileReader();
  reader.onload = () => resolve(reader.result as string);
  reader.onerror = () => reject(new UserError("Gagal membaca foto."));
  reader.readAsDataURL(file);
  return promise;
}

function loadImage(src: string): Promise<HTMLImageElement> {
  const { promise, resolve, reject } = Promise.withResolvers<HTMLImageElement>();
  const img = new Image();
  img.onload = () => resolve(img);
  img.onerror = () => reject(new UserError("Gagal membaca foto."));
  img.src = src;
  return promise;
}
