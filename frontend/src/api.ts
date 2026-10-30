// Mirrors integration/api.py's PredictRequest / /predict / /health response shapes.

export class UserError extends Error {}

export type GlucoseRange = "normal" | "borderline" | "high";
export type PredictionMode = "baseline_demo" | "nlp_dish_identity_forecast" | "experimental_fusion";

export interface Health {
  status: string;
  prediction_mode: string;
  schema_version: string;
  model_version: string;
  is_trained_model: boolean;
  serving: string;
  g0_sources: Record<string, string>;
  nlp_dish_identity_mode: string;
  experimental_fusion_mode: string;
  disclaimer: string;
}

/** experimental_fusion_mode reads "available as prediction_mode=... — <version>" once an artifact is registered. */
export function fusionAvailable(health: Health | null): boolean {
  return !!health && health.experimental_fusion_mode.startsWith("available");
}

export interface PpgCapturePayload {
  samples: number[];
  timestamps_seconds: number[];
  sampling_rate_hz: number;
}

export interface GlucoseHistoryEntry {
  timestamp: string;
  value: number;
}

export interface PredictPayload {
  participant_id: string;
  t0_timestamp: string;
  delta_t_minutes: 60 | 120;
  prediction_mode: PredictionMode;
  meal_id?: string;
  source_dataset: string;
  meal_image_data_url?: string | null;
  note_text?: string | null;
  ppg_capture?: PpgCapturePayload | null;
  portion_reported?: 0 | 1 | 2;
  glucose_history?: GlucoseHistoryEntry[];
  fingerstick_mgdl?: number | null;
  fingerstick_timestamp?: string | null;
  allow_ppg_g0_estimate?: boolean;
  time_since_last_meal_hours?: number | null;
}

export interface ModalityStatus {
  status: string;
  detail: string | null;
  [extra: string]: unknown;
}

export interface PredictResponse {
  meal_id: string;
  prediction_mode: string;
  forecast: {
    glucose_t60: number;
    glucose_t120: number;
    glucose_range_t60: GlucoseRange;
    model_version: string;
    basis: string;
    modality_weights: { cv: number; nlp: number; ppg: number };
    g0_mgdl?: number;
    g0_source?: string;
    g0_is_direct_measurement?: boolean;
    g0_trust?: string;
    g0_detail?: string;
    [extra: string]: unknown;
  };
  forecast_input_usage: {
    used_for_glucose_value: string[];
    not_used_for_glucose_value: string[];
    message: string;
  };
  modality_processing: { cv: ModalityStatus; nlp: ModalityStatus; ppg: ModalityStatus };
  modalities: {
    cv: {
      present: number; confidence: number; food_class: string | null;
      carbs_g: number; protein_g: number; fat_g: number; fiber_g: number;
      gi_category: string; carbs_source: string; weight: number;
    };
    nlp: {
      present: number; confidence: number; is_fried_cooking: boolean; is_large_portion: boolean;
      feature_source: string; calibrated: boolean | null; calibration_status: string; weight: number;
    };
    ppg: {
      present: number; signal_quality: number; pulse_rate_bpm: number; hrv_rmssd: number;
      perfusion_index: number; clip_fraction: number; n_windows: number; weight: number;
    };
  };
  glucose_history_used: {
    n_causal_readings: number; n_dropped_as_noncausal: number;
    g0: number | null; g0_age_minutes: number | null; g0_rejected_as_stale: boolean; max_g0_age_minutes: number;
  };
  glucose_source: {
    value_mgdl: number; source: string; is_direct_measurement: boolean; trust: string;
    detail: string; rejected_candidates: unknown[];
  };
  schema_version: string;
  disclaimer: string;
  generated_at: string;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
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
  return res.json() as Promise<T>;
}

export function getHealth(): Promise<Health> {
  return request<Health>("/health");
}

export function predict(payload: PredictPayload): Promise<PredictResponse> {
  return request<PredictResponse>("/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}
