// State and helpers shared by the three steps (glucose, meal, forecast) and the app shell.
import { UserError, type DishRow, type G0Reading, type GlucoseRange, type PhotoItem } from "./api";
import { nutritionFor, sumNutrition, type Nutrition } from "./nutrition";
import type { Profile, SourceId } from "./storage";
import { minutesAgo, toInputValue } from "./time";
import type { Tone } from "./ui";

export const RANGE_LABEL: Record<GlucoseRange, string> = { normal: "Normal", borderline: "Waspada", high: "Tinggi" };
export const RANGE_TONE: Record<GlucoseRange, Tone> = { normal: "green", borderline: "amber", high: "red" };
export const SOURCE_LABEL: Record<SourceId, string> = { meter: "Glukometer", cgm: "CGM", ppg: "Scan Jari (PPG)" };
export const PORTION_LABELS = ["Sedikit", "Sedang", "Banyak"] as const;
/** "Sedang" is the row's standard serving; the others scale it. Same factors as the API's PORTION_SCALE. */
export const PORTION_SCALE = [0.5, 1, 1.5] as const;

/** The server refuses a glucose reading older than this (stale pre-meal value). */
export const FRESH_WITHIN_MIN = 60;
export const GLUCOSE_MIN = 40;
export const GLUCOSE_MAX = 400;

export type Portion = 0 | 1 | 2;

/** One food on the plate. `row` is null while a photo guess still needs the user's pick. */
export interface MealEntry {
  key: number;
  row: DishRow | null;
  source: "photo" | "manual";
  /** What the photo called it; set for entries that came from the photo. */
  vlmName?: string;
  suggestions: DishRow[];
  /** Photo-match scores, kept only for entries the photo accepted. */
  similarity?: number;
  margin?: number;
  mode: "portion" | "gram";
  portion: Portion;
  grams: string;
}

export interface Draft {
  source: SourceId | "";
  meterValue: string; meterTime: string;
  cgmValue: string; cgmTime: string;
  ppg: G0Reading | null;
  photo: string | null;
  vlmModelVersion: string | null;
  items: MealEntry[];
  foodMode: "photo" | "manual";
}

export const newDraft = (profile: Profile): Draft => ({
  source: profile.defaultSource === "ppg" ? "" : profile.defaultSource,
  meterValue: "", meterTime: toInputValue(minutesAgo(20)),
  cgmValue: "", cgmTime: toInputValue(minutesAgo(5)),
  ppg: null, photo: null, vlmModelVersion: null, items: [], foodMode: "photo",
});

let keyCounter = 1;

export function entryFor(row: DishRow | null, source: MealEntry["source"], extra: Partial<MealEntry> = {}): MealEntry {
  return {
    key: keyCounter++, row, source, suggestions: [], mode: row?.serving_g != null || !row ? "portion" : "gram",
    portion: 1, grams: "", ...extra,
  };
}

/** Photo items the matcher accepted become resolved entries; the rest wait for the user's pick. */
export function entryFromPhoto(item: PhotoItem): MealEntry {
  return item.accepted && item.row
    ? entryFor(item.row, "photo", { vlmName: item.vlm_name, suggestions: item.suggestions, similarity: item.similarity, margin: item.margin })
    : entryFor(null, "photo", { vlmName: item.vlm_name, suggestions: item.suggestions });
}

export const num = (s: string): number | null => {
  const v = parseFloat(s.replace(",", "."));
  return Number.isFinite(v) ? v : null;
};

export const fmt = (v: number): string => v.toFixed(1).replace(".", ",");

/** The user chose `row` for an unresolved entry (or replaced it): from now on it is a manual pick. */
export const pickRow = (e: MealEntry, row: DishRow): MealEntry => ({
  ...e, row, source: "manual", similarity: undefined, margin: undefined,
  mode: row.serving_g === null ? "gram" : "portion",
});

/** Grams the entry stands for: a portion of the row's standard serving, or the typed weight. */
export function entryGrams(e: MealEntry): number | null {
  if (!e.row) return null;
  if (e.mode === "gram") return num(e.grams);
  return e.row.serving_g === null ? null : Math.round(e.row.serving_g * PORTION_SCALE[e.portion] * 10) / 10;
}

export const isResolved = (e: MealEntry): boolean => (entryGrams(e) ?? 0) > 0;

export function mealNutrition(items: MealEntry[]): { total: Nutrition; counted: number } {
  const parts = items.flatMap((e) => {
    const g = entryGrams(e);
    return e.row && g && g > 0 ? [nutritionFor(e.row, g)] : [];
  });
  return { total: sumNutrition(parts), counted: parts.length };
}

export function glucoseOf(d: Draft): { value: number | null; time: string | null } {
  if (d.source === "meter") return { value: num(d.meterValue), time: d.meterTime };
  if (d.source === "cgm") return { value: num(d.cgmValue), time: d.cgmTime };
  if (d.source === "ppg") return { value: d.ppg ? Math.round(d.ppg.value_mgdl) : null, time: null };
  return { value: null, time: null };
}

export function validGlucose(value: number | null, time: string | null): string | null {
  if (value === null) return "Masukkan nilai gula darah.";
  if (value < GLUCOSE_MIN || value > GLUCOSE_MAX) return `Nilai harus antara ${GLUCOSE_MIN} dan ${GLUCOSE_MAX} mg/dL.`;
  if (time) {
    const t = new Date(time).getTime();
    if (t > Date.now()) return "Waktu pengukuran tidak boleh di masa depan.";
    if (Date.now() - t > FRESH_WITHIN_MIN * 60_000) return `Pengukuran lebih dari ${FRESH_WITHIN_MIN} menit yang lalu. Ukur ulang.`;
  }
  return null;
}

/** The range a PPG estimate stands for: value ± the model's 90% held-out error. */
export function ppgRange(r: G0Reading): { lo: number; hi: number } | null {
  if (r.error_range_mgdl === null) return null;
  return { lo: Math.round(r.value_mgdl - r.error_range_mgdl), hi: Math.round(r.value_mgdl + r.error_range_mgdl) };
}

// Only messages written for users are shown; anything raw (HTTP/contract
// errors, stack traces) goes to the console for developers instead.
export const errorText = (e: unknown): string => {
  if (e instanceof UserError) return e.message;
  console.error(e);
  return "Terjadi kendala saat memproses. Periksa data Anda lalu coba lagi.";
};
