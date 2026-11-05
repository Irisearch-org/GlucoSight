import type { GlucoseRange } from "./api";

export type SourceId = "meter" | "cgm" | "ppg";

export interface Profile {
  participantId: string;
  name: string;
  email: string;
  age: string;
  gender: string;
  heightCm: string;
  weightKg: string;
  diabetesType: string;
  defaultSource: SourceId;
}

export interface HistoryRecord {
  id: string;
  createdAt: string;
  food: string;
  portion: string;
  t60: number;
  t120: number;
  range: GlucoseRange;
  g0: number;
  sourceId: SourceId;
  basis: string;
  usageMessage: string;
  modelVersion: string;
}

// Profile and history live in the database, per account (see api.ts). These keys are
// what older versions of the app kept in this browser; they are read once, uploaded
// to the signed-in account, and then removed.
const KEY_PROFILE = "glucosight.v1.profile";
const KEY_HISTORY = "glucosight.v1.history";

export const EMPTY_PROFILE: Profile = {
  participantId: "", name: "", email: "", age: "", gender: "Pria",
  heightCm: "", weightKg: "", diabetesType: "Tipe 2", defaultSource: "meter",
};

function read<T>(key: string, fallback: T): T {
  try {
    const raw = localStorage.getItem(key);
    return raw ? (JSON.parse(raw) as T) : fallback;
  } catch {
    return fallback;
  }
}

export function peekLegacy(): { profile: Profile | null; history: HistoryRecord[] } {
  return { profile: read<Profile | null>(KEY_PROFILE, null), history: read<HistoryRecord[]>(KEY_HISTORY, []) };
}

export function dropLegacy(): void {
  try {
    localStorage.removeItem(KEY_PROFILE);
    localStorage.removeItem(KEY_HISTORY);
  } catch {
    /* storage blocked */
  }
}
