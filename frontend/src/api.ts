import type { HistoryRecord, Profile } from "./storage";

// Mirrors integration/api.py: /health, /dishes, /glucose/ppg, /meal/photo and /predict (schema 2.0).

export class UserError extends Error {}

export type GlucoseRange = "normal" | "borderline" | "high";

export interface Health {
  status: string;
  schema_version: string;
  served: Record<string, unknown>;
  dish_table_build: string | null;
}

/** One row of the dish table; nutrition is per 100 g edible portion. */
export interface DishRow {
  id: string;
  name: string;
  kind: "dish" | "ingredient";
  serving_g: number | null;
  energy_kcal: number;
  carbs_g: number;
  protein_g: number;
  fat_g: number;
  fibre_g: number | null;
  confidence: "exact" | "approximate" | "computed";
}

export interface PpgCapture {
  samples: number[];
  fs: number;
  timestamps?: number[];
  torch_verified: boolean;
  clip_fraction: number;
}

export interface G0Reading {
  value_mgdl: number;
  source: "meter" | "cgm" | "ppg";
  timestamp: string;
  age_minutes: number;
  error_range_mgdl: number | null;
  heart_rate_bpm: number | null;
  capture_valid: boolean | null;
  ppg_model_version: string | null;
}

export interface PhotoItem {
  vlm_name: string;
  accepted: boolean;
  row: DishRow | null;
  suggestions: DishRow[];
  similarity: number;
  margin: number;
}

export interface MealPhotoResponse {
  items: PhotoItem[];
  vlm_model_version: string | null;
  dish_table_build: string | null;
}

export interface PredictItem {
  row_id: string;
  grams?: number;
  portion?: 0 | 1 | 2;
  source: "photo" | "manual";
  similarity?: number;
  margin?: number;
}

export type PredictGlucose =
  | { source: "meter" | "cgm"; value_mgdl: number; timestamp: string }
  | { source: "ppg"; reading: G0Reading };

export interface PredictProfile {
  age_years: number | null;
  sex: "M" | "F" | null;
  height_cm: number | null;
  weight_kg: number | null;
  diabetes_type: string;
}

export interface PredictBody {
  t0_timestamp: string;
  glucose: PredictGlucose;
  meal: { items: PredictItem[]; vlm_model_version: string | null };
  profile: PredictProfile;
}

export interface Horizon {
  glucose_mgdl: number;
  range: GlucoseRange;
  error_band_mgdl: number;
  model_version: string;
}

export interface PredictResponse {
  schema_version: string;
  g0: G0Reading;
  meal: {
    items: {
      row: DishRow; grams: number; source: "photo" | "manual";
      carbs_g: number; protein_g: number; fat_g: number; fibre_g: number | null; kcal: number;
    }[];
    totals: { carbs_g: number; protein_g: number; fat_g: number; fibre_g: number | null; kcal: number };
    dish_table_build: string;
    vlm_model_version: string | null;
  };
  forecast: { t60: Horizon; t120: Horizon };
  outside_training: boolean;
  disclaimer: string;
}

export async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(path, init);
  } catch {
    throw new UserError("Tidak dapat terhubung ke server.");
  }
  if (!res.ok) {
    let detail = `Permintaan gagal (${res.status}).`;
    try {
      const body = await res.json();
      if (body?.detail) detail = String(body.detail);
    } catch {
      /* non-JSON error body */
    }
    throw new UserError(detail);
  }
  if (res.status === 204) return undefined as T;
  return res.json() as Promise<T>;
}

const json = (body: unknown, method = "POST"): RequestInit => ({
  method,
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

export const getHealth = () => request<Health>("/health");

export const searchDishes = (q: string, limit = 10, signal?: AbortSignal) =>
  request<{ items: DishRow[] }>(`/dishes?q=${encodeURIComponent(q)}&limit=${limit}`, { signal }).then((r) => r.items);

export const scanGlucosePpg = (capture: PpgCapture) => request<G0Reading>("/glucose/ppg", json({ capture }));

export const identifyMealPhoto = (imageDataUrl: string) =>
  request<MealPhotoResponse>("/meal/photo", json({ image_data_url: imageDataUrl }));

export const predict = (body: PredictBody) => request<PredictResponse>("/predict", json(body));

// Accounts live in the `glucosight` database; the session is an HttpOnly cookie
// set by /auth/*, so none of these calls handle a token.
export interface AuthUser {
  participant_id: string;
  name: string;
  email: string;
}

export const registerAccount = (name: string, email: string, password: string) =>
  request<{ user: AuthUser }>("/auth/register", json({ name, email, password })).then((r) => r.user);

export const loginAccount = (email: string, password: string) =>
  request<{ user: AuthUser }>("/auth/login", json({ email, password })).then((r) => r.user);

export const logoutAccount = () => request<void>("/auth/logout", { method: "POST" });

/** The signed-in user, or null when there is no valid session (or the server is unreachable). */
export const getMe = () =>
  request<{ user: AuthUser }>("/auth/me").then((r) => r.user).catch(() => null);

// Per-account profile and prediction history (integration/user_data.py).
export const fetchProfile = () => request<{ profile: Profile; saved: boolean }>("/user/profile");

export const saveProfileRemote = (p: Profile) =>
  request<{ profile: Profile }>("/user/profile", json(p, "PUT")).then((r) => r.profile);

export const fetchHistory = () => request<{ history: HistoryRecord[] }>("/user/history").then((r) => r.history);

export const saveHistoryRemote = (record: HistoryRecord) =>
  request<unknown>("/user/history", json(record));
