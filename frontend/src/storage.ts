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

const KEY_PROFILE = "glucosight.v1.profile";
const KEY_HISTORY = "glucosight.v1.history";
const KEY_SIGNED_IN = "glucosight.v1.signedIn";

function uuid(): string {
  return crypto.randomUUID ? crypto.randomUUID() : `p-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function read<T>(key: string, fallback: T): T {
  try {
    const raw = localStorage.getItem(key);
    return raw ? (JSON.parse(raw) as T) : fallback;
  } catch {
    return fallback;
  }
}

function write(key: string, value: unknown): void {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    /* storage blocked (private mode, quota) */
  }
}

export function loadProfile(): Profile {
  const saved = read<Profile | null>(KEY_PROFILE, null);
  if (saved) return saved;
  const fresh: Profile = {
    participantId: uuid(), name: "", email: "", age: "", gender: "Pria",
    heightCm: "", weightKg: "", diabetesType: "Tipe 2", defaultSource: "meter",
  };
  write(KEY_PROFILE, fresh);
  return fresh;
}

export function saveProfile(p: Profile): void {
  write(KEY_PROFILE, p);
}

export function loadHistory(): HistoryRecord[] {
  return read<HistoryRecord[]>(KEY_HISTORY, []);
}

export function addHistory(record: HistoryRecord): HistoryRecord[] {
  const next = [record, ...loadHistory()];
  write(KEY_HISTORY, next);
  return next;
}

export function isSignedIn(): boolean {
  return read<boolean>(KEY_SIGNED_IN, false);
}

export function setSignedIn(v: boolean): void {
  write(KEY_SIGNED_IN, v);
}
