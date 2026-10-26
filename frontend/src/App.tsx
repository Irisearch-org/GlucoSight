import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import {
  UserError, fusionAvailable, getHealth, predict,
  type GlucoseRange, type Health, type ModalityStatus, type PpgCapturePayload, type PredictPayload, type PredictResponse,
} from "./api";
import { DISH_CATALOG, findDish, nutritionFor, sumNutrition, type Dish, type Nutrition } from "./foodCatalog";
import Landing from "./Landing";
import { FoodIcon } from "./foodIcons";
import { frameToDataUrl, photoToDataUrl } from "./image";
import { cameraSupported, PPG_SECONDS, PpgRecorder, type FrameInfo } from "./ppg";
import {
  addHistory, isSignedIn, loadHistory, loadProfile, saveProfile, setSignedIn,
  type HistoryRecord, type Profile, type SourceId,
} from "./storage";
import { ageMinutes, describeAge, formatWhen, minutesAgo, toApiTimestamp, toInputValue } from "./time";
import {
  AppBar, Badge, BottomNav, Brand, Button, ErrorBox, Field, Icon, SelectField, WizardLayout, initials,
  type IconName, type Screen, type Tone,
} from "./ui";

// ---------------------------------------------------------------- constants

const RANGE_LABEL: Record<GlucoseRange, string> = { normal: "Normal", borderline: "Waspada", high: "Tinggi" };
const RANGE_TONE: Record<GlucoseRange, Tone> = { normal: "green", borderline: "amber", high: "red" };
const PORTION_LABELS = ["Sedikit", "Sedang", "Banyak"] as const;
const SOURCE_LABEL: Record<SourceId, string> = { meter: "Glukometer", cgm: "CGM", ppg: "Scan Jari (PPG)" };

/** The CV model's own gate is 0.60; the design asks users to retake photos below 75%. */
const CV_CONFIDENCE_GATE = 75;
/** The API treats a pre-meal reading older than this as stale and falls back to the population value. */
const STALE_AFTER_MIN = 360;
const FRESH_WITHIN_MIN = 60;

const sourceOptions: { id: SourceId; icon: IconName; title: string; sub: string; badge?: string; tone?: Tone }[] = [
  { id: "meter", icon: "meter", title: "Glukometer", sub: "Masukkan hasil fingerstick", badge: "Terukur", tone: "green" },
  { id: "cgm", icon: "pulse", title: "CGM", sub: "Masukkan pembacaan terakhir dari aplikasi CGM", badge: "Terukur", tone: "green" },
  { id: "ppg", icon: "camera", title: "Scan Jari (PPG)", sub: "Estimasi lewat kamera ponsel" },
];

// Standard serving of the recognised dish = "Sedang"; the others scale it.
const PORTION_SCALE = [0.5, 1, 1.5];
const portionCards = [
  { grams: "±100 g", title: "½ piring", plate: "small" },
  { grams: "±200 g", title: "1 piring standar", plate: "medium" },
  { grams: ">350 g", title: "Piring munjung", plate: "large" },
];

// ------------------------------------------------------------------- types

interface PpgResult { glucose: number; bpm: number; quality: number; capture: PpgCapturePayload }
interface CvResult {
  recognized: boolean; status: ModalityStatus; detail: string | null; foodClass: string | null;
  confidence: number; macros: { carbs: number; protein: number; fat: number; fiber: number };
}
interface ManualItem { id: number; name: string; mode: "estimate" | "gram"; portion: 0 | 1 | 2; grams: string }
type FoodMode = "photo" | "manual";

interface Draft {
  source: SourceId | "";
  meterValue: string; meterTime: string;
  cgmValue: string; cgmTime: string;
  ppg: PpgResult | null;
  photo: string | null;
  cv: CvResult | null;
  portion: 0 | 1 | 2 | null;
  items: ManualItem[];
  foodMode: FoodMode;
}

type ResultState =
  | { status: "loading" }
  | { status: "error"; message: string }
  | { status: "ready"; record: HistoryRecord; fromHistory: boolean };

const newDraft = (profile: Profile): Draft => ({
  source: profile.defaultSource === "ppg" ? "" : profile.defaultSource,
  meterValue: "", meterTime: toInputValue(minutesAgo(20)),
  cgmValue: "", cgmTime: toInputValue(minutesAgo(5)),
  ppg: null, photo: null, cv: null, portion: null, items: [], foodMode: "photo",
});

// ---------------------------------------------------------------- helpers

const num = (s: string): number | null => {
  const v = parseFloat(s.replace(",", "."));
  return Number.isFinite(v) ? v : null;
};

function glucoseOf(d: Draft): { value: number | null; time: string | null } {
  if (d.source === "meter") return { value: num(d.meterValue), time: d.meterTime };
  if (d.source === "cgm") return { value: num(d.cgmValue), time: d.cgmTime };
  if (d.source === "ppg") return { value: d.ppg ? Math.round(d.ppg.glucose) : null, time: null };
  return { value: null, time: null };
}

function validGlucose(value: number | null, time: string | null): string | null {
  if (value === null) return "Masukkan nilai gula darah.";
  if (value < 20 || value > 600) return "Nilai harus antara 20 dan 600 mg/dL.";
  if (time && new Date(time).getTime() > Date.now()) return "Waktu pengukuran tidak boleh di masa depan.";
  return null;
}

const dishName = (foodClass: string | null): string =>
  (foodClass ?? "").split("_").filter(Boolean).map((w) => w[0].toUpperCase() + w.slice(1)).join(" ");

const fmt = (v: number): string => v.toFixed(1).replace(".", ",");

function manualPortionIndex(items: ManualItem[]): 0 | 1 | 2 {
  // Gram entries are bucketed so one overall in-app portion answer exists
  // for the API (portion_reported is a single 0/1/2 for the whole meal).
  const idx = items.map((i) => {
    if (i.mode === "estimate") return i.portion;
    const g = num(i.grams) ?? 0;
    return g < 150 ? 0 : g < 300 ? 1 : 2;
  });
  if (!idx.length) return 1;
  return Math.max(0, Math.min(2, Math.round(idx.reduce<number>((a, b) => a + b, 0) / idx.length))) as 0 | 1 | 2;
}

function noteFromItems(items: ManualItem[]): string {
  const size = ["kecil", "sedang", "besar"];
  return items
    .map((i) => (i.mode === "gram" ? `${i.name} ${num(i.grams) ?? 0} gram` : `${i.name} porsi ${size[i.portion]}`))
    .join(", ");
}

function foodLabel(d: Draft): string {
  if (d.foodMode === "photo" && d.cv?.recognized) return dishName(d.cv.foodClass);
  const names = d.items.map((i) => i.name);
  return names.length ? names.join(", ") : "Makanan";
}

function buildPayload(d: Draft, profile: Profile, mode: PredictPayload["prediction_mode"]): PredictPayload {
  const payload: PredictPayload = {
    participant_id: profile.participantId,
    t0_timestamp: toApiTimestamp(new Date()),
    delta_t_minutes: 60,
    prediction_mode: mode,
    source_dataset: "live_capture",
  };
  if (d.source === "meter") {
    payload.fingerstick_mgdl = num(d.meterValue);
    payload.fingerstick_timestamp = toApiTimestamp(new Date(d.meterTime));
  } else if (d.source === "cgm") {
    payload.glucose_history = [{ timestamp: toApiTimestamp(new Date(d.cgmTime)), value: num(d.cgmValue) ?? 0 }];
  } else if (d.source === "ppg" && d.ppg) {
    payload.ppg_capture = d.ppg.capture;
  }
  if (d.foodMode === "photo" && d.cv?.recognized && d.photo) {
    payload.meal_image_data_url = d.photo;
    payload.portion_reported = d.portion ?? 1;
  } else if (d.foodMode === "manual") {
    payload.note_text = noteFromItems(d.items);
    payload.portion_reported = manualPortionIndex(d.items);
  }
  return payload;
}

function toRecord(res: PredictResponse, d: Draft): HistoryRecord {
  const portion = d.foodMode === "photo" ? d.portion ?? 1 : manualPortionIndex(d.items);
  return {
    id: res.meal_id,
    createdAt: new Date().toISOString(),
    food: foodLabel(d),
    portion: PORTION_LABELS[portion],
    t60: res.forecast.glucose_t60,
    t120: res.forecast.glucose_t120,
    range: res.forecast.glucose_range_t60,
    g0: Math.round(res.glucose_source.value_mgdl),
    sourceId: d.source === "" ? "meter" : d.source,
    basis: res.forecast.basis,
    usageMessage: res.forecast_input_usage.message,
    modelVersion: res.forecast.model_version,
  };
}

/** Marker position on the 3-band bar: bands are equal thirds, split at 140 and 200 mg/dL. */
function markerPercent(v: number): number {
  if (v < 140) return Math.max(2, ((v - 70) / 70) * 33.3);
  if (v < 200) return 33.3 + ((v - 140) / 60) * 33.3;
  return Math.min(98, 66.6 + ((v - 200) / 60) * 33.3);
}

// Only messages written for users are shown; anything raw (HTTP/contract
// errors, stack traces) goes to the console for developers instead.
const errorText = (e: unknown): string => {
  if (e instanceof UserError) return e.message;
  console.error(e);
  return "Terjadi kendala saat memproses. Periksa data Anda lalu coba lagi.";
};

// ------------------------------------------------------------ small pieces

function SourceCard({ option, selected, onClick }: { option: (typeof sourceOptions)[number]; selected: boolean; onClick: () => void }) {
  return <button className={`option-card ${selected ? "is-selected" : ""}`} onClick={onClick}>
    <span className="option-card__icon"><Icon name={option.icon}/></span>
    <span className="option-card__copy"><span><strong>{option.title}</strong>{option.badge && <Badge tone={option.tone}>{option.badge}</Badge>}</span><small>{option.sub}</small></span>
    <span className="radio">{selected && <i/>}</span>
  </button>;
}

function Wave({ data }: { data: number[] }) {
  if (data.length < 2) return <svg className="wave" viewBox="0 0 300 70" preserveAspectRatio="none"><path d="M0 35h300"/></svg>;
  const min = Math.min(...data), max = Math.max(...data), span = max - min || 1;
  const points = data.map((v, i) => `${((i / (data.length - 1)) * 300).toFixed(1)},${(64 - ((v - min) / span) * 58).toFixed(1)}`).join(" ");
  return <svg className="wave" viewBox="0 0 300 70" preserveAspectRatio="none"><polyline points={points}/></svg>;
}

const qualityLabel = (q: number): { text: string; tone: Tone } =>
  q >= 0.7 ? { text: "Baik", tone: "green" } : q >= 0.4 ? { text: "Cukup", tone: "amber" } : { text: "Kurang", tone: "red" };

// -------------------------------------------------------------------- auth

function Auth({ register, profile, go, onSignIn }: {
  register?: boolean; profile: Profile; go: (s: Screen) => void; onSignIn: (name: string, email: string) => void;
}) {
  const [name, setName] = useState("");
  const [email, setEmail] = useState(profile.email);
  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");

  const emailOk = /\S+@\S+\.\S+/.test(email);
  const valid = register
    ? name.trim().length > 1 && emailOk && password.length >= 8 && password === confirm
    : emailOk && password.length > 0;
  const hint = register && password && password.length < 8 ? "Kata sandi minimal 8 karakter."
    : register && confirm && password !== confirm ? "Konfirmasi kata sandi belum sama." : "";

  return <div className="auth-screen">
    <div className="auth-glow"/>
    <main className="auth-card">
      <button className="auth-back" type="button" onClick={() => go("landing")}><Icon name="arrow" size={16}/> Halaman utama</button>
      <Brand/>
      <div className="auth-copy">
        <h1>{register ? "Buat akun Anda" : "Selamat datang kembali"}</h1>
        <p>{register ? "Mulai memahami respons gula darah Anda setelah makan." : "Masuk untuk melanjutkan perjalanan kesehatan Anda."}</p>
      </div>
      <form className="form-stack" onSubmit={(e) => { e.preventDefault(); if (valid) onSignIn(name.trim(), email.trim()); }}>
        {register && <Field label="Nama lengkap" value={name} onChange={setName}/>}
        <Field label="Email" type="email" value={email} onChange={setEmail}/>
        <Field label="Kata sandi" type="password" placeholder={register ? "Minimal 8 karakter" : "Kata sandi"} value={password} onChange={setPassword}/>
        {register && <Field label="Konfirmasi kata sandi" type="password" placeholder="Ulangi kata sandi" value={confirm} onChange={setConfirm}/>}
        {hint && <p className="field-hint">{hint}</p>}
        <button className="button button--primary" type="submit" disabled={!valid}><span>{register ? "Daftar" : "Masuk"}</span></button>
      </form>
      <p className="auth-switch">{register ? "Sudah punya akun?" : "Belum punya akun?"} <button onClick={() => go(register ? "login" : "register")}>{register ? "Masuk" : "Daftar"}</button></p>
    </main>
  </div>;
}

// -------------------------------------------------------------------- home

function Home({ profile, history, apiState, nav, go, onOpen }: {
  profile: Profile; history: HistoryRecord[]; apiState: "loading" | "ok" | "down";
  nav: (s: Screen) => void; go: (s: Screen) => void; onOpen: (r: HistoryRecord) => void;
}) {
  const [insightOpen, setInsightOpen] = useState(false);
  const dateLabel = new Intl.DateTimeFormat("id-ID", { weekday: "long", day: "numeric", month: "long", year: "numeric" }).format(new Date());
  const last = history[0];
  const week = useMemo(() => {
    const since = Date.now() - 7 * 24 * 3600_000;
    const rows = history.filter((h) => new Date(h.createdAt).getTime() >= since);
    const count = (r: GlucoseRange) => rows.filter((h) => h.range === r).length;
    return { n: rows.length, normal: count("normal"), borderline: count("borderline"), high: count("high") };
  }, [history]);
  const pct = (n: number) => (week.n ? Math.round((n / week.n) * 100) : 0);
  const first = profile.name.trim().split(/\s+/)[0] || "Anda";

  return <div className="shell with-nav shell--home">
    <AppBar title="" action={<button className="avatar" onClick={() => go("profile")} aria-label="Profil">{initials(profile.name)}</button>}/>
    <main className="page home-page">
      <section className="greeting"><div><p>{dateLabel}</p><h1>Halo, {first}</h1></div></section>
      {apiState === "down" && <ErrorBox title="Server prediksi tidak terhubung">Prediksi belum dapat dibuat saat ini. Kami mencoba menyambung kembali otomatis.</ErrorBox>}
      <button className="prediction-cta" onClick={() => nav("source")}>
        <div className="prediction-cta__art"><div className="plate"><Icon name="sparkle" size={25}/></div><i className="line line-a"/><i className="line line-b"/></div>
        <div className="prediction-cta__copy"><Badge tone="teal">Prediksi baru</Badge><h2>Prediksi gula<br/>setelah makan</h2><p>Kenali respons tubuh Anda dalam 60 menit.</p><span>Mulai prediksi <Icon name="chevron" size={17}/></span></div>
      </button>
      <div className="section-heading"><div><span>Terakhir</span><h2>Prediksi terbaru</h2></div>{history.length > 0 && <button onClick={() => go("history")}>Lihat riwayat</button>}</div>
      {last ? <button className="last-card last-card--button" onClick={() => onOpen(last)}>
        <div className="last-card__top"><FoodIcon name={last.food} size={43}/><div><h3>{last.food}</h3><p>{formatWhen(last.createdAt)}</p></div><Badge tone={RANGE_TONE[last.range]}>{RANGE_LABEL[last.range]}</Badge></div>
        <div className="last-card__stats"><div><span>Prediksi T+60</span><strong>{Math.round(last.t60)} <small>mg/dL</small></strong></div><div><span>Sebelum makan</span><strong>{last.g0} <small>mg/dL</small></strong></div></div>
      </button> : <div className="empty-card"><Icon name="chart" size={28}/><strong>Belum ada prediksi</strong><p>Hasil prediksi Anda akan tersimpan di sini.</p></div>}
      {week.n > 0 && <>
        <button className={`insight-card ${insightOpen ? "is-open" : ""}`} onClick={() => setInsightOpen(!insightOpen)} aria-expanded={insightOpen}>
          <div className="insight-card__icon"><Icon name="sparkle"/></div><div><strong>Wawasan 7 hari terakhir</strong><p>{week.normal} dari {week.n} prediksi berada di rentang normal.</p></div><Icon name="chevron" size={18}/>
        </button>
        {insightOpen && <section className="insight-detail">
          <div className="insight-detail__head"><div><span>7 hari terakhir</span><h3>Ringkasan</h3></div><Badge tone="green">{pct(week.normal)}% normal</Badge></div>
          <div className="insight-bars">
            <div><span>Normal</span><i><b className="normal" style={{ width: `${pct(week.normal)}%` }}/></i><strong>{week.normal}</strong></div>
            <div><span>Waspada</span><i><b className="warning" style={{ width: `${pct(week.borderline)}%` }}/></i><strong>{week.borderline}</strong></div>
            <div><span>Tinggi</span><i><b className="high" style={{ width: `${pct(week.high)}%` }}/></i><strong>{week.high}</strong></div>
          </div>
          <p className="insight-caption">Dihitung dari {week.n} prediksi Anda.</p>
        </section>}
      </>}
    </main>
    <BottomNav active="home" go={nav}/>
  </div>;
}

// ----------------------------------------------------------------- history

function History({ history, nav, go, onOpen }: { history: HistoryRecord[]; nav: (s: Screen) => void; go: (s: Screen) => void; onOpen: (r: HistoryRecord) => void }) {
  const [filter, setFilter] = useState<"Semua" | GlucoseRange>("Semua");
  const visible = history.filter((h) => filter === "Semua" || h.range === filter);
  const normalPct = history.length ? Math.round((history.filter((h) => h.range === "normal").length / history.length) * 100) : 0;
  const oldest = history.length ? new Intl.DateTimeFormat("id-ID", { month: "long", year: "numeric" }).format(new Date(history[history.length - 1].createdAt)) : "";
  const filters: { id: "Semua" | GlucoseRange; label: string }[] = [
    { id: "Semua", label: "Semua" }, { id: "normal", label: "Normal" }, { id: "borderline", label: "Waspada" }, { id: "high", label: "Tinggi" },
  ];
  return <div className="shell with-nav">
    <AppBar title="Riwayat Prediksi" onBack={() => go("home")}/>
    <main className="page history-page">
      <section className="history-overview">
        <div><span>Seluruh prediksi</span><strong>{history.length}</strong><small>{oldest ? `Tersimpan sejak ${oldest}` : "Belum ada data"}</small></div>
        <div className="history-donut" style={{ background: `conic-gradient(#9ce2d3 0 ${normalPct}%, rgba(255,255,255,.18) ${normalPct}%)` }}><span>{normalPct}%<small>Normal</small></span></div>
      </section>
      <div className="history-filters">{filters.map((f) => <button className={filter === f.id ? "active" : ""} onClick={() => setFilter(f.id)} key={f.id}>{f.label}</button>)}</div>
      <div className="history-heading"><div><h1>Prediksi tersimpan</h1><span>{visible.length} hasil</span></div></div>
      <div className="history-list">{visible.map((item) => <button className="history-item" key={item.id} onClick={() => onOpen(item)}>
        <FoodIcon name={item.food} size={48}/>
        <span className="history-item__copy"><strong>{item.food}</strong><small>{formatWhen(item.createdAt)}</small><span>Awal {item.g0} mg/dL</span></span>
        <span className="history-item__result"><Badge tone={RANGE_TONE[item.range]}>{RANGE_LABEL[item.range]}</Badge><strong>{Math.round(item.t60)}<small> mg/dL</small></strong><Icon name="chevron" size={15}/></span>
      </button>)}</div>
      {!visible.length && <div className="history-empty"><Icon name="chart" size={34}/><strong>Belum ada hasil</strong><p>{history.length ? "Tidak ada prediksi dengan status ini." : "Prediksi yang Anda buat akan muncul di sini."}</p></div>}
    </main>
    <BottomNav active="history" go={nav}/>
  </div>;
}

// ------------------------------------------------------------ step 1: source

function Source({ draft, patch, go }: { draft: Draft; patch: (p: Partial<Draft>) => void; go: (s: Screen) => void }) {
  const g = glucoseOf(draft);
  const problem = draft.source === "ppg" ? (draft.ppg ? null : "Lakukan scan jari terlebih dahulu.") : validGlucose(g.value, g.time);
  const touched = (draft.source === "meter" && draft.meterValue !== "") || (draft.source === "cgm" && draft.cgmValue !== "");
  return <WizardLayout current={0} title="Prediksi Baru" back={() => go("home")}>
    <div className="title-block"><span>Langkah 1 dari 4</span><h1>Dari mana data gula darah Anda?</h1><p>Pilih sumber pengukuran sebelum makan.</p></div>
    <div className="option-list">{sourceOptions.map((o) => <SourceCard key={o.id} option={o} selected={draft.source === o.id}
      onClick={() => { patch({ source: o.id }); if (o.id === "ppg") go("scan"); }}/>)}</div>
    {draft.source === "meter" && <section className="reveal-card">
      <div className="reveal-card__head"><Icon name="drop"/><div><strong>Hasil pengukuran</strong><span>Masukkan data terbaru</span></div></div>
      <Field label="Gula darah" type="number" value={draft.meterValue} onChange={(v) => patch({ meterValue: v })} suffix="mg/dL" min={20} max={600}/>
      <Field label="Waktu tes" type="datetime-local" value={draft.meterTime} onChange={(v) => patch({ meterTime: v })}/>
      <p className="helper"><Icon name="clock" size={15}/> Sebaiknya diukur &lt; 60 menit sebelum makan</p>
      {touched && problem && <p className="field-hint">{problem}</p>}
    </section>}
    {draft.source === "cgm" && <section className="reveal-card">
      <div className="reveal-card__head"><Icon name="pulse"/><div><strong>Pembacaan CGM terakhir</strong><span>Salin dari aplikasi CGM Anda</span></div></div>
      <Field label="Gula darah" type="number" value={draft.cgmValue} onChange={(v) => patch({ cgmValue: v })} suffix="mg/dL" min={20} max={600}/>
      <Field label="Waktu pembacaan" type="datetime-local" value={draft.cgmTime} onChange={(v) => patch({ cgmTime: v })}/>
      <p className="helper"><Icon name="info" size={15}/> Sambungan otomatis ke perangkat CGM belum tersedia.</p>
      {touched && problem && <p className="field-hint">{problem}</p>}
    </section>}
    {draft.source === "ppg" && draft.ppg && <section className="reveal-card">
      <div className="reveal-card__head"><Icon name="camera"/><div><strong>Estimasi dari scan jari</strong><span>{Math.round(draft.ppg.glucose)} mg/dL · {Math.round(draft.ppg.bpm)} BPM</span></div></div>
      <button className="inline-action" onClick={() => go("scan")}><Icon name="camera" size={16}/> Pindai ulang</button>
    </section>}
    <div className="sticky-actions"><Button disabled={!draft.source || !!problem} onClick={() => go("summary")}>Lanjut</Button></div>
  </WizardLayout>;
}

// ----------------------------------------------------------------- PPG scan

type ScanPhase = "ready" | "starting" | "aim" | "recording" | "processing" | "done" | "error";

function Scan({ mode, participantId, go, onUse }: {
  mode: PredictPayload["prediction_mode"]; participantId: string; go: (s: Screen) => void; onUse: (r: PpgResult) => void;
}) {
  const [phase, setPhase] = useState<ScanPhase>("ready");
  const [frame, setFrame] = useState<FrameInfo | null>(null);
  const [error, setError] = useState("");
  const [outcome, setOutcome] = useState<PpgResult | null>(null);
  const [torch, setTorch] = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null);
  const recRef = useRef<PpgRecorder | null>(null);

  useEffect(() => () => recRef.current?.dispose(), []);

  const process = async (capture: PpgCapturePayload) => {
    setPhase("processing");
    try {
      const res = await predict({
        participant_id: participantId, t0_timestamp: toApiTimestamp(new Date()), delta_t_minutes: 60,
        prediction_mode: mode, ppg_capture: capture, source_dataset: "live_capture",
      });
      const status = res.modality_processing.ppg;
      if (status.status !== "processed") {
        console.error("PPG not processed:", status);
        throw new UserError("Sinyal jari belum cukup baik. Pastikan jari menutup lensa dan flash, lalu ulangi pemindaian.");
      }
      if (res.glucose_source.source !== "ppg_estimate") {
        throw new UserError("Estimasi gula dari scan jari belum tersedia saat ini. Gunakan glukometer atau CGM.");
      }
      setOutcome({
        glucose: res.glucose_source.value_mgdl, bpm: res.modalities.ppg.pulse_rate_bpm,
        quality: res.modalities.ppg.signal_quality, capture,
      });
      setPhase("done");
    } catch (e) {
      setError(errorText(e));
      setPhase("error");
    }
  };

  const openCamera = async () => {
    if (!videoRef.current) return;
    setPhase("starting");
    setError("");
    const rec: PpgRecorder = new PpgRecorder(videoRef.current, setFrame, (capture, problem) => {
      rec.dispose();
      if (!capture) {
        setError(problem ?? "Sinyal belum cukup. Ulangi pemindaian.");
        setPhase("error");
        return;
      }
      void process(capture);
    });
    recRef.current = rec;
    try {
      await rec.start();
      setTorch(rec.torchOn);
      setPhase("aim");
    } catch (e) {
      rec.dispose();
      setError(e instanceof DOMException && e.name === "NotAllowedError"
        ? "Izin kamera ditolak. Izinkan akses kamera di pengaturan browser."
        : "Kamera tidak dapat dibuka di perangkat ini.");
      setPhase("error");
    }
  };

  const cancel = () => { recRef.current?.dispose(); setFrame(null); setPhase("ready"); };
  const remaining = Math.max(0, Math.ceil(PPG_SECONDS - (frame?.elapsed ?? 0)));
  const live = phase === "aim" || phase === "recording";
  const q = outcome ? qualityLabel(outcome.quality) : null;

  const heading = phase === "done" ? "Pemindaian selesai"
    : phase === "recording" ? "Tetap tahan jari Anda"
    : phase === "processing" ? "Memproses sinyal…"
    : phase === "aim" ? "Tutup lensa dengan jari" : "Posisikan ujung jari";
  const copy = phase === "done" ? "Kami berhasil menangkap sinyal Anda."
    : phase === "recording" ? "Jangan bergerak agar sinyal tetap stabil."
    : phase === "aim" ? "Tutup kamera belakang dan lampu flash sepenuhnya, lalu mulai merekam."
    : phase === "processing" ? "Menghitung denyut nadi dan kualitas sinyal."
    : "Tutup kamera belakang dan lampu flash sepenuhnya dengan ujung jari.";

  return <WizardLayout current={0} title="Scan Jari" back={() => { recRef.current?.dispose(); go("source"); }}>
    <div className="scan-heading"><h1>{heading}</h1><p>{copy}</p></div>
    {phase !== "done" ? <div className="scan-view">
      <div className={`finger-guide ${phase === "recording" ? "is-recording" : ""}`}>
        <video ref={videoRef} className={`finger-guide__video ${live ? "is-live" : ""}`} playsInline muted/>
        {!live && <Icon name="camera" size={33}/>}
        <span>{phase === "recording" ? remaining : PPG_SECONDS}<small>detik</small></span>
      </div>
      <div className="signal"><span>Sinyal jari</span>
        <Badge tone={frame?.fingerOn ? "green" : "gray"}>{phase === "recording" ? (frame?.fingerOn ? "Baik" : "Jari bergeser") : frame?.fingerOn ? "Jari terdeteksi" : "Menunggu"}</Badge></div>
      <Wave data={live ? frame?.wave ?? [] : []}/>
      <div className="flash-note"><span className="flash-icon">ϟ</span> {live ? (torch ? "Flash menyala" : "Flash tidak dapat dinyalakan otomatis — gunakan cahaya terang") : "Flash dinyalakan otomatis saat kamera dibuka"}</div>
    </div> : outcome && q && <div className="scan-result">
      <div className="success-ring"><Icon name="check" size={31}/></div><p>Estimasi gula darah</p><strong>{Math.round(outcome.glucose)} <small>mg/dL</small></strong>
      <div className="scan-result__meta">
        <div><Icon name="pulse"/><span>Denyut nadi<strong>{Math.round(outcome.bpm)} BPM</strong></span></div>
        <div><Icon name="chart"/><span>Kualitas sinyal<strong>{q.text}</strong></span></div>
      </div>    </div>}
    {phase === "error" && <ErrorBox title="Pemindaian belum berhasil">{error}</ErrorBox>}
    {phase === "ready" && !cameraSupported() && <p className="field-hint">Kamera belum dapat dipakai di halaman ini.</p>}
    <div className="sticky-actions">
      {phase === "ready" && <Button icon="camera" disabled={!cameraSupported()} onClick={openCamera}>Mulai pindai</Button>}
      {phase === "starting" && <Button disabled>Membuka kamera…</Button>}
      {phase === "aim" && <Button disabled={!frame?.fingerOn} onClick={() => { recRef.current?.record(); setPhase("recording"); }}>Mulai merekam</Button>}
      {phase === "recording" && <><Button disabled>Merekam… {remaining} detik</Button><Button variant="text" onClick={cancel}>Batalkan</Button></>}
      {phase === "processing" && <Button disabled>Memproses…</Button>}
      {phase === "done" && outcome && <><Button onClick={() => onUse(outcome)}>Gunakan hasil</Button><Button variant="text" onClick={() => { setOutcome(null); setFrame(null); setPhase("ready"); }}>Ulangi pemindaian</Button></>}
      {phase === "error" && <Button onClick={() => { setFrame(null); setPhase("ready"); }}>Ulangi</Button>}    </div>
  </WizardLayout>;
}

// --------------------------------------------------------- step 2: summary

function Summary({ draft, go }: { draft: Draft; go: (s: Screen) => void }) {
  const g = glucoseOf(draft);
  const source = (draft.source || "meter") as SourceId;
  const fromScan = source === "ppg";
  const age = g.time ? ageMinutes(g.time) : 0;
  const tip = age <= FRESH_WITHIN_MIN
    ? { cls: "", icon: "check" as IconName, title: "Data masih cukup baru", text: "Pengukuran dilakukan kurang dari 60 menit sebelum makan." }
    : age <= STALE_AFTER_MIN
      ? { cls: "tip-card--warn", icon: "clock" as IconName, title: "Data sudah agak lama", text: "Pertimbangkan mengukur ulang agar prediksi lebih tepat." }
      : { cls: "tip-card--warn", icon: "clock" as IconName, title: "Data terlalu lama", text: "Data ini tidak akan dipakai; prediksi akan memakai nilai populasi. Ukur ulang." };
  return <WizardLayout current={1} title="Ringkasan Data" back={() => go("source")}>
    <div className="title-block"><span>Langkah 2 dari 4</span><h1>Pastikan data sudah benar</h1><p>Data ini menjadi dasar prediksi Anda.</p></div>
    <article className="summary-card">
      <div className="summary-card__top"><div className="drop-box"><Icon name={source === "ppg" ? "camera" : source === "cgm" ? "pulse" : "drop"}/></div><div><span>Gula sebelum makan</span><strong>{g.value} <small>mg/dL</small></strong></div>{!fromScan && <Badge tone="green">Terukur</Badge>}</div>
      <div className="summary-card__rows">
        <div><span>Sumber data</span><strong>{SOURCE_LABEL[source]}</strong></div>
        <div><span>Waktu pengukuran</span><strong>{g.time ? describeAge(age) : "baru saja"}</strong></div>
        <div><span>Tanggal &amp; waktu</span><strong>{formatWhen(g.time ? new Date(g.time).toISOString() : new Date().toISOString())}</strong></div>
        {fromScan && draft.ppg && <>
          <div><span>Denyut nadi</span><strong>{Math.round(draft.ppg.bpm)} BPM</strong></div>
          <div><span>Kualitas sinyal</span><strong>{qualityLabel(draft.ppg.quality).text}</strong></div>
        </>}
      </div>
    </article>
    <div className={`tip-card ${tip.cls}`}><Icon name={tip.icon}/><div><strong>{tip.title}</strong><p>{tip.text}</p></div></div>
    <div className="sticky-actions split"><Button variant="secondary" onClick={() => go("source")} icon="edit">Ubah</Button><Button onClick={() => go("camera")}>Lanjut</Button></div>
  </WizardLayout>;
}

// ------------------------------------------------------------ step 3: food

type CamState = "starting" | "live" | "unavailable";

function Camera({ patch, participantId, go }: {
  patch: (p: Partial<Draft>) => void; participantId: string; go: (s: Screen) => void;
}) {
  const [shot, setShot] = useState<string | null>(null);
  const [cam, setCam] = useState<CamState>("starting");
  const [camProblem, setCamProblem] = useState("");  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const videoRef = useRef<HTMLVideoElement>(null);
  const gallery = useRef<HTMLInputElement>(null);
  const shoot = useRef<HTMLInputElement>(null);

  // The live preview runs whenever no photo has been taken yet, so coming back
  // here (retake, "Ganti foto") always starts from the camera, never a stale photo.
  useEffect(() => {
    if (shot) return;
    let stream: MediaStream | null = null;
    let cancelled = false;
    setCam("starting");
    (async () => {
      if (!cameraSupported()) {
        setCamProblem("Kamera belum dapat dipakai di halaman ini.");
        setCam("unavailable");
        return;
      }
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: { ideal: "environment" }, width: { ideal: 1280 }, height: { ideal: 960 } },
          audio: false,
        });
        if (cancelled) { stream.getTracks().forEach((t) => t.stop()); return; }
        const v = videoRef.current;
        if (v) { v.srcObject = stream; await v.play().catch(() => undefined); }
        setCam("live");
      } catch (e) {
        setCamProblem(e instanceof DOMException && e.name === "NotAllowedError"
          ? "Izin kamera ditolak. Izinkan akses kamera di pengaturan browser."
          : "Kamera tidak dapat dibuka di perangkat ini.");
        setCam("unavailable");
      }
    })();
    return () => { cancelled = true; stream?.getTracks().forEach((t) => t.stop()); };
  }, [shot]);

  // A photo goes straight to recognition; there is no confirm-the-photo step.
  const submit = (photo: string) => { setShot(photo); void recognize(photo); };

  const capture = () => {
    const video = videoRef.current;
    const frame = video && cam === "live" ? frameToDataUrl(video) : null;
    if (frame) { submit(frame); return; }
    shoot.current?.click(); // no live preview: hand off to the device's own camera app
  };

  const pick = async (file: File | undefined) => {
    if (!file) return;
    try { setError(""); submit(await photoToDataUrl(file)); } catch (e) { setError(errorText(e)); }
  };

  const recognize = async (photo: string) => {
    setError("");
    setLoading(true);
    try {
      patch({ photo, cv: null, portion: null, foodMode: "photo" });
      // baseline_demo is always available and CV runs regardless of mode, so
      // this call is only used to read what the CV track recognised.
      const res = await predict({
        participant_id: participantId, t0_timestamp: toApiTimestamp(new Date()), delta_t_minutes: 60,
        prediction_mode: "baseline_demo", meal_image_data_url: photo, source_dataset: "live_capture",
      });
      const st = res.modality_processing.cv;
      const cv = res.modalities.cv;
      const recognized = (st.status === "ok" || st.status === "low_confidence") && cv.present === 1 && !!cv.food_class;
      patch({
        cv: {
          recognized, status: st.status, detail: st.detail, foodClass: cv.food_class,
          confidence: Math.round(cv.confidence * 100),
          macros: { carbs: cv.carbs_g, protein: cv.protein_g, fat: cv.fat_g, fiber: cv.fiber_g },
        },
      });
      go("confirm");
    } catch (e) {
      setError(errorText(e));
      setShot(null); // back to the camera so the user can simply shoot again
    } finally {
      setLoading(false);
    }
  };

  return <WizardLayout current={2} title="Foto Makanan" back={() => go("summary")}>
    <div className="camera-copy"><span>Langkah 3 dari 4</span><h1>Foto makanan Anda</h1><p>Pastikan seluruh makanan terlihat jelas.</p></div>
    <div className="camera-view">
      {shot ? <img className="camera-photo" src={shot} alt="Foto makanan"/> : <>
        <video ref={videoRef} className={`camera-photo ${cam === "live" ? "" : "is-hidden"}`} playsInline muted autoPlay/>
        {cam !== "live" && <div className="camera-status">
          {cam === "starting"
            ? <><span className="loader"/><strong>Membuka kamera…</strong></>
            : <><Icon name="camera" size={30}/><strong>Kamera tidak tersedia</strong><small>{camProblem} Gunakan Galeri untuk memilih foto.</small></>}
        </div>}
      </>}
      {(shot || cam === "live") && <><div className="frame-corner c1"/><div className="frame-corner c2"/><div className="frame-corner c3"/><div className="frame-corner c4"/></>}
      {loading && <div className="recognizing"><span className="loader"/><strong>Mengenali makanan…</strong><small>Menganalisis isi piring Anda</small></div>}
    </div>
    {error && <ErrorBox title="Foto belum dapat diproses">{error}</ErrorBox>}
    {!shot && <div className="photo-tip"><Icon name="info"/><span><strong>Tips foto yang baik</strong>Foto dari atas, seluruh piring terlihat.</span></div>}
    <input ref={shoot} type="file" accept="image/*" capture="environment" hidden onChange={(e) => { void pick(e.target.files?.[0]); e.target.value = ""; }}/>
    <input ref={gallery} type="file" accept="image/jpeg,image/png,image/webp" hidden onChange={(e) => { void pick(e.target.files?.[0]); e.target.value = ""; }}/>
    {!shot && <div className="camera-actions">
      <button className="gallery-button" onClick={() => gallery.current?.click()}><Icon name="gallery"/><span>Galeri</span></button>
      <button className="shutter" disabled={cam === "starting"} onClick={capture} aria-label="Ambil foto"><span/></button>
      <div className="action-spacer"/>
    </div>}
    <Button variant="text" onClick={() => { patch({ foodMode: "manual" }); go("manual"); }}>Masukkan makanan secara manual</Button>
  </WizardLayout>;
}

function Confirm({ draft, patch, go, onPredict }: { draft: Draft; patch: (p: Partial<Draft>) => void; go: (s: Screen) => void; onPredict: () => void }) {
  const cv = draft.cv;
  const portion = draft.portion;
  const passed = !!cv?.recognized && cv.confidence >= CV_CONFIDENCE_GATE;
  const manual = () => { patch({ foodMode: "manual" }); go("manual"); };
  const scale = portion === null ? 1 : PORTION_SCALE[portion];
  const macros = cv ? [
    ["Karbohidrat", cv.macros.carbs * scale, 120], ["Protein", cv.macros.protein * scale, 40],
    ["Lemak", cv.macros.fat * scale, 40], ["Serat", cv.macros.fiber * scale, 12],
  ] as const : [];

  return <WizardLayout current={2} title="Konfirmasi Makanan" back={() => go("camera")}>
    <div className="food-hero">{draft.photo ? <img className="camera-photo" src={draft.photo} alt="Foto makanan"/> : null}
      <button onClick={() => go("camera")}><Icon name="edit" size={17}/> Ganti foto</button></div>
    {!cv?.recognized ? <>
      <ErrorBox title="Makanan belum dikenali">Ambil ulang foto dari atas dengan cahaya cukup, atau tambahkan makanan secara manual.</ErrorBox>
      <div className="retry-actions"><Button onClick={() => go("camera")} icon="camera">Foto ulang</Button><Button variant="secondary" icon="edit" onClick={manual}>Input manual</Button></div>
    </> : <>
      <section className={`confidence-card ${passed ? "confidence-card--pass" : "confidence-card--low"}`}>
        <div className="confidence-card__top"><div><span>Tingkat keyakinan</span><strong>{cv.confidence}%</strong></div><Badge tone={passed ? "green" : "amber"}>{passed ? "Foto dapat digunakan" : `Di bawah batas ${CV_CONFIDENCE_GATE}%`}</Badge></div>
        <div className="confidence-track"><i style={{ width: `${cv.confidence}%` }}/><b/></div>
        {!passed && <div className="low-confidence"><Icon name="info"/><div><strong>Foto belum cukup jelas</strong><p>Ambil ulang dari atas dengan pencahayaan lebih baik, atau tambahkan secara manual.</p></div></div>}
      </section>
      {!passed ? <div className="retry-actions"><Button onClick={() => go("camera")} icon="camera">Foto ulang</Button><Button variant="secondary" icon="edit" onClick={manual}>Input manual</Button></div> : <>
        <section className="detected-card"><div className="detected-card__head"><div><span>Makanan terdeteksi</span><h1>{dishName(cv.foodClass)}</h1></div><Badge tone="teal">Yakin {cv.confidence}%</Badge></div></section>
        <section className="portion"><div><strong>Seberapa besar porsinya?</strong><span>Bandingkan dengan porsi sehari-hari</span></div>
          <div className="portion-cards">{portionCards.map((item, i) => <button className={portion === i ? "active" : ""} onClick={() => patch({ portion: i as 0 | 1 | 2 })} key={item.title}>
            <span className={`portion-plate portion-plate--${item.plate}`}><i/></span><strong>{PORTION_LABELS[i]}</strong><small>{item.title}</small><b>{item.grams}</b>
          </button>)}</div>
        </section>
        {portion !== null && <section className="nutrition-reveal">
          <div className="nutrition-reveal__head"><div><span>Estimasi porsi {PORTION_LABELS[portion].toLowerCase()}</span><strong>Rincian nutrisi</strong></div><Badge tone="gray">Perkiraan</Badge></div>
          <div className="macro-list">{macros.map(([name, value, ref]) => <div className="macro" key={name}><span>{name}</span><div><i style={{ width: `${Math.min(100, (value / ref) * 100)}%` }}/></div><strong>{fmt(value)} g</strong></div>)}</div>
        </section>}
        <div className="sticky-actions"><Button disabled={portion === null} onClick={onPredict} icon="check">Gunakan makanan ini</Button><Button variant="text" onClick={manual}>Masih kurang sesuai? Input manual</Button></div>
      </>}
    </>}
  </WizardLayout>;
}

/** Grams a manual item stands for: a portion of the dish's standard serving, or the typed weight. */
function itemGrams(item: ManualItem, dish: Dish | undefined): number | null {
  if (item.mode === "gram") return num(item.grams);
  return dish ? dish.servingG * PORTION_SCALE[item.portion] : null;
}

function NutritionRows({ n }: { n: Nutrition }) {
  const rows = [["Karbohidrat", n.carbs, 120], ["Protein", n.protein, 40], ["Lemak", n.fat, 40], ["Serat", n.fiber, 12]] as const;
  return <div className="macro-list">{rows.map(([name, value, ref]) => <div className="macro" key={name}>
    <span>{name}</span><div><i style={{ width: `${Math.min(100, (value / ref) * 100)}%` }}/></div><strong>{fmt(value)} g</strong>
  </div>)}</div>;
}

function Manual({ draft, patch, go, onPredict }: { draft: Draft; patch: (p: Partial<Draft>) => void; go: (s: Screen) => void; onPredict: () => void }) {
  const [query, setQuery] = useState("");
  const [focused, setFocused] = useState(false);
  const nextId = useRef(1);
  const items = draft.items;
  const q = query.trim().toLowerCase();
  const matches = DISH_CATALOG.filter((f) => !q || f.toLowerCase().includes(q));
  const addFree = !!q && !matches.some((m) => m.toLowerCase() === q);
  const add = (name: string) => {
    patch({ foodMode: "manual", items: [...items, { id: nextId.current++, name, mode: "estimate", portion: 1, grams: "" }] });
    setQuery("");
  };
  const update = (id: number, change: Partial<ManualItem>) => patch({ items: items.map((i) => (i.id === id ? { ...i, ...change } : i)) });
  const ready = items.length > 0 && items.every((i) => i.mode === "estimate" || (num(i.grams) ?? 0) > 0);
  const known = items.flatMap((i) => {
    const dish = findDish(i.name);
    const g = itemGrams(i, dish);
    return dish && g && g > 0 ? [nutritionFor(dish, g)] : [];
  });
  const total = sumNutrition(known);

  return <WizardLayout current={2} title="Input Manual" back={() => go("camera")}>
    <div className="title-block"><span>Langkah 3 dari 4</span><h1>Tambahkan makanan</h1><p>Cari makanan Anda, lalu pilih porsinya.</p></div>
    <div className="search-wrap"><Icon name="search"/><input value={query} onChange={(e) => setQuery(e.target.value)} onFocus={() => setFocused(true)} onBlur={() => setFocused(false)} placeholder="Cari atau ketik nama makanan…"/>
      {focused && (matches.length > 0 || addFree) && <div className="suggestions" onMouseDown={(e) => e.preventDefault()}>
        {matches.length > 0 && <span className="suggestions__label">Pilihan makanan</span>}
        {matches.map((x) => <button onClick={() => add(x)} key={x}><span className="suggestions__item"><FoodIcon name={x} size={30}/>{x}</span><Icon name="plus" size={17}/></button>)}
        {addFree && <button onClick={() => add(query.trim())}><span className="suggestions__item"><FoodIcon name={query.trim()} size={30}/>Tambah “{query.trim()}”</span><Icon name="plus" size={17}/></button>}
      </div>}
    </div>
    <div className="manual-label"><strong>Makanan ditambahkan</strong><Badge tone="gray">{items.length} item</Badge></div>
    <div className="food-items">{items.map((item) => {
      const dish = findDish(item.name);
      const grams = itemGrams(item, dish);
      const n = dish && grams && grams > 0 ? nutritionFor(dish, grams) : null;
      return <div className="food-row" key={item.id}>
        <FoodIcon name={item.name} size={40}/>
        <div>
          <strong>{item.name}</strong>
          <div className="mini-segments"><button className={item.mode === "estimate" ? "active" : ""} onClick={() => update(item.id, { mode: "estimate" })}>Perkiraan</button><button className={item.mode === "gram" ? "active" : ""} onClick={() => update(item.id, { mode: "gram" })}>Gram</button></div>
          {item.mode === "estimate"
            ? <div className="mini-portions">{PORTION_LABELS.map((label, i) => <button key={label} className={item.portion === i ? "active" : ""} onClick={() => update(item.id, { portion: i as 0 | 1 | 2 })}>
                {label}{dish && <small>≈{Math.round(dish.servingG * PORTION_SCALE[i])} g</small>}
              </button>)}</div>
            : <div className="mini-gram"><input type="number" min={1} inputMode="decimal" value={item.grams} onChange={(e) => update(item.id, { grams: e.target.value })}/><b>gram</b></div>}
          {n ? <div className="food-nutri">
            <div className="food-nutri__head"><span>Rincian nutrisi</span><b>≈ {Math.round(grams ?? 0)} g · {Math.round(n.kcal)} kkal</b></div>
            <NutritionRows n={n}/>
          </div> : dish ? null : <p className="food-nutri__na">Rincian nutrisi belum tersedia untuk makanan ini.</p>}
        </div>
        <button onClick={() => patch({ items: items.filter((i) => i.id !== item.id) })} aria-label={`Hapus ${item.name}`}><Icon name="trash" size={19}/></button>
      </div>;
    })}</div>
    {!items.length && <p className="helper">Belum ada makanan. Ketik nama makanan di kolom pencarian.</p>}
    {known.length > 0 && <section className="nutrition-reveal">
      <div className="nutrition-reveal__head"><div><span>{known.length < items.length ? `Berdasarkan ${known.length} dari ${items.length} makanan` : `Berdasarkan ${known.length} makanan`}</span><strong>Total nutrisi</strong></div><Badge tone="teal">{Math.round(total.kcal)} kkal</Badge></div>
      <NutritionRows n={total}/>
    </section>}
    <div className="sticky-actions"><Button disabled={!ready} onClick={onPredict}>Lihat prediksi</Button></div>
  </WizardLayout>;
}

// ------------------------------------------------------------- step 4: result

const rangeOf = (v: number): GlucoseRange => (v < 140 ? "normal" : v < 200 ? "borderline" : "high");

/** Counts from the pre-meal glucose up to the predicted value; the range marker and colour follow along. */
function ResultGauge({ from, to }: { from: number; to: number }) {
  const reduce = typeof window !== "undefined" && !!window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
  const [v, setV] = useState(reduce ? to : from);
  useEffect(() => {
    if (reduce) return;
    let raf = 0;
    const t0 = performance.now();
    const tick = (now: number) => {
      const p = Math.min(1, (now - t0) / 1500);
      setV(from + (to - from) * (1 - Math.pow(1 - p, 3)));
      if (p < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [from, to, reduce]);
  return <>
    <div className={`result-number result-number--${rangeOf(v)}`}>{Math.round(v)} <small>mg/dL</small></div>
    <p>Perkiraan 60 menit setelah suapan pertama</p>
    <div className="range"><div><i style={{ left: `${markerPercent(v)}%` }}/></div><span><b>Normal</b><b>Waspada</b><b>Tinggi</b></span></div>
  </>;
}

/** Shown while the prediction is computed: a glucose curve drawn over and over, with a dot riding its head. */
function LoadingCurve() {
  const d = "M6 66C40 64 56 58 84 36S128 10 152 16S186 42 194 44";
  return <svg className="result-curve" viewBox="0 0 200 80" aria-hidden="true">
    <path d={d} fill="none" stroke="#d6ecea" strokeWidth="3" strokeLinecap="round"/>
    <path id="rc-path" className="result-curve__line" d={d} pathLength="1" fill="none" stroke="#12a9a3" strokeWidth="3" strokeLinecap="round"/>
    <circle r="5.5" fill="#fff" stroke="#0FA0A0" strokeWidth="3">
      <animateMotion dur="2.4s" repeatCount="indefinite" calcMode="spline" keyTimes="0;1" keySplines=".5 .1 .3 1"><mpath href="#rc-path"/></animateMotion>
    </circle>
  </svg>;
}

function Result({ state, nav, go, onRetry, backTo }: {
  state: ResultState; nav: (s: Screen) => void; go: (s: Screen) => void; onRetry: () => void; backTo: Screen;
}) {
  return <WizardLayout current={3} title="Hasil Prediksi" back={() => go(backTo)}>
    {state.status === "loading" && <div className="result-loading"><LoadingCurve/><h1>Menghitung prediksi…</h1><p>Kami menggabungkan data gula darah dan nutrisi makanan Anda.</p>
      <div className="loading-steps"><span className="done">Data gula darah <Icon name="check"/></span><span className="done">Data makanan <Icon name="check"/></span><span>Model prediksi <i className="loader"/></span></div></div>}
    {state.status === "error" && <>
      <ErrorBox title="Prediksi belum berhasil">{state.message}</ErrorBox>
      <div className="sticky-actions"><Button onClick={onRetry}>Coba lagi</Button><Button variant="secondary" onClick={() => go("source")}>Periksa data</Button></div>
    </>}
    {state.status === "ready" && (() => {
      const r = state.record;
      return <>
        <div className="result-intro"><Badge tone="teal"><Icon name="sparkle" size={13}/> {state.fromHistory ? "Prediksi tersimpan" : "Prediksi selesai"}</Badge><h1>Respons gula darah Anda</h1></div>
        <article className="result-card">
          <div className="result-card__top"><span>Prediksi T+60</span><Badge tone={RANGE_TONE[r.range]}>{RANGE_LABEL[r.range]}</Badge></div>
          <ResultGauge from={r.g0} to={r.t60}/>
        </article>
        <div className="baseline"><span className="baseline__icon"><Icon name="drop"/></span><div><span>Gula sebelum makan</span><strong>{r.g0} mg/dL <small>· {SOURCE_LABEL[r.sourceId]}</small></strong></div>{r.sourceId !== "ppg" && <Badge tone="green">Terukur</Badge>}</div>
        <section className="inputs-used"><strong>Data yang digunakan</strong>
          <div><span><Icon name="drop"/> Gula darah</span><b>{r.g0} mg/dL</b></div>
          <div><span><Icon name="gallery"/> Makanan</span><b>{r.food} · {r.portion}</b></div></section>        <div className="sticky-actions"><Button onClick={() => nav("home")}>Selesai</Button><Button variant="secondary" onClick={() => nav("source")}>Prediksi lagi</Button></div>
      </>;
    })()}
  </WizardLayout>;
}

// ------------------------------------------------------------------ profile

function bmiOf(h: string, w: string): { value: string; label: string } | null {
  const height = num(h), weight = num(w);
  if (!height || !weight || height < 80) return null;
  const v = weight / Math.pow(height / 100, 2);
  // Kemenkes RI cut-offs for adults.
  const label = v < 18.5 ? "Kurus" : v < 25 ? "Normal" : v < 27 ? "Gemuk" : "Obesitas";
  return { value: v.toFixed(1).replace(".", ","), label };
}

function ProfileScreen({ profile, historyCount, nav, go, onSave, onSignOut }: {
  profile: Profile; historyCount: number; nav: (s: Screen) => void; go: (s: Screen) => void;
  onSave: (p: Profile) => void; onSignOut: () => void;
}) {
  const [form, setForm] = useState(profile);
  const [saved, setSaved] = useState(false);
  const set = (change: Partial<Profile>) => { setForm({ ...form, ...change }); setSaved(false); };
  const bmi = bmiOf(form.heightCm, form.weightKg);
  return <div className="shell with-nav"><AppBar title="Profil &amp; Pengaturan"/><main className="page profile-page">
    <div className="profile-hero"><div className="profile-avatar">{initials(form.name)}</div><div><h1>{form.name || "Pengguna"}</h1><p>{form.email || "Belum ada email"}</p></div></div>
    <section className="settings-section"><div className="section-title"><h2>Data pribadi</h2><span>Lengkapi data diri Anda</span></div><div className="settings-card">
      <Field label="Nama lengkap" value={form.name} onChange={(v) => set({ name: v })}/>
      <div className="field-grid"><Field label="Usia" type="number" value={form.age} onChange={(v) => set({ age: v })} suffix="tahun"/><SelectField label="Jenis kelamin" value={form.gender} options={["Pria", "Wanita"]} onChange={(v) => set({ gender: v })}/></div>
      <div className="field-grid"><Field label="Tinggi badan" type="number" value={form.heightCm} onChange={(v) => set({ heightCm: v })} suffix="cm"/><Field label="Berat badan" type="number" value={form.weightKg} onChange={(v) => set({ weightKg: v })} suffix="kg"/></div>
      <div className="bmi-row"><div><span>Indeks massa tubuh</span><strong>{bmi ? bmi.value : "—"} {bmi && <small>{bmi.label}</small>}</strong></div><span>Dihitung otomatis</span></div>
      <SelectField label="Jenis diabetes" value={form.diabetesType} options={["Tipe 2", "Tipe 1", "Prediabetes", "Lainnya"]} onChange={(v) => set({ diabetesType: v })}/>
    </div></section>
    <section className="settings-section"><div className="section-title"><h2>Aktivitas</h2><span>Data prediksi yang sudah tersimpan</span></div>
      <button className="profile-history-row" onClick={() => go("history")}><span className="device-logo"><Icon name="chart"/></span><span><strong>Riwayat prediksi</strong><small>{historyCount} prediksi tersimpan</small></span><Icon name="chevron" size={18}/></button>
    </section>
    <section className="settings-section"><div className="section-title"><h2>Sumber gula default</h2><span>Dipilih otomatis saat memulai prediksi</span></div>
      <div className="option-list">{sourceOptions.filter((o) => o.id !== "ppg").map((o) => <SourceCard key={o.id} option={o} selected={form.defaultSource === o.id} onClick={() => set({ defaultSource: o.id })}/>)}</div>
      <p className="helper" style={{ marginTop: 10 }}><Icon name="info" size={15}/> Scan Jari dipilih langsung saat memulai prediksi.</p>
    </section>
    <Button onClick={() => { onSave(form); setSaved(true); }}>{saved ? "Tersimpan ✓" : "Simpan perubahan"}</Button>
    <Button variant="text" onClick={onSignOut}>Keluar</Button>
  </main><BottomNav active="profile" go={nav}/></div>;
}

// ---------------------------------------------------------------------- app

const WIZARD: Screen[] = ["source", "scan", "summary", "camera", "confirm", "manual", "result"];

export default function App() {
  const [profile, setProfile] = useState<Profile>(loadProfile);
  // Every visit starts on the landing page; signed-in users go on via "Masuk".
  const [screen, setScreen] = useState<Screen>("landing");
  const [draft, setDraft] = useState<Draft>(() => newDraft(loadProfile()));
  const [history, setHistory] = useState<HistoryRecord[]>(loadHistory);
  const [result, setResult] = useState<ResultState>({ status: "loading" });
  const [resultBack, setResultBack] = useState<Screen>("home");
  const [health, setHealth] = useState<Health | null>(null);
  const [apiState, setApiState] = useState<"loading" | "ok" | "down">("loading");

  // Re-check every few seconds while the API is unreachable, so a server that
  // starts after the page loaded is picked up without a manual reload.
  useEffect(() => {
    if (apiState === "ok") return;
    let cancelled = false;
    const check = () => getHealth()
      .then((h) => { if (!cancelled) { setHealth(h); setApiState("ok"); } })
      .catch(() => { if (!cancelled) setApiState("down"); });
    void check();
    const timer = window.setInterval(check, 4000);
    return () => { cancelled = true; window.clearInterval(timer); };
  }, [apiState]);

  const mode = fusionAvailable(health) ? "experimental_fusion" : "baseline_demo";
  const patch = (p: Partial<Draft>) => setDraft((d) => ({ ...d, ...p }));
  const go = (next: Screen) => { setScreen(next); window.scrollTo({ top: 0 }); };
  // Entering the wizard from outside it always starts a clean draft.
  const nav = (next: Screen) => {
    if (next === "source") setDraft(newDraft(profile));
    go(next);
  };

  const runPrediction = async () => {
    setResult({ status: "loading" });
    setResultBack(draft.foodMode === "photo" ? "confirm" : "manual");
    go("result");
    try {
      const res = await predict(buildPayload(draft, profile, mode));
      const record = toRecord(res, draft);
      setHistory(addHistory(record));
      setResult({ status: "ready", record, fromHistory: false });
    } catch (e) {
      setResult({ status: "error", message: errorText(e) });
    }
  };

  const openRecord = (record: HistoryRecord) => {
    setResult({ status: "ready", record, fromHistory: true });
    setResultBack(screen === "home" ? "home" : "history");
    go("result");
  };

  const signIn = (name: string, email: string) => {
    const local = email.split("@")[0];
    const next = { ...profile, email, name: name || profile.name || local.charAt(0).toUpperCase() + local.slice(1) };
    setProfile(next);
    saveProfile(next);
    setSignedIn(true);
    go("home");
  };

  const signOut = () => { setSignedIn(false); go("landing"); };

  if (screen === "landing") {
    return <Landing onLogin={() => go(isSignedIn() ? "home" : "login")} onRegister={() => go(isSignedIn() ? "home" : "register")}/>;
  }
  if (!WIZARD.includes(screen) && ["login", "register"].includes(screen)) {
    return <Auth register={screen === "register"} profile={profile} go={go} onSignIn={signIn}/>;
  }

  switch (screen) {
    case "home": return <Home profile={profile} history={history} apiState={apiState} nav={nav} go={go} onOpen={openRecord}/>;
    case "history": return <History history={history} nav={nav} go={go} onOpen={openRecord}/>;
    case "profile": return <ProfileScreen profile={profile} historyCount={history.length} nav={nav} go={go} onSave={(p) => { setProfile(p); saveProfile(p); }} onSignOut={signOut}/>;
    case "source": return <Source draft={draft} patch={patch} go={go}/>;
    case "scan": return <Scan mode={mode} participantId={profile.participantId} go={go}
      onUse={(r) => { patch({ source: "ppg", ppg: r }); go("summary"); }}/>;
    case "summary": return <Summary draft={draft} go={go}/>;
    case "camera": return <Camera patch={patch} participantId={profile.participantId} go={go}/>;
    case "confirm": return <Confirm draft={draft} patch={patch} go={go} onPredict={runPrediction}/>;
    case "manual": return <Manual draft={draft} patch={patch} go={go} onPredict={runPrediction}/>;
    case "result": return <Result state={result} nav={nav} go={go} onRetry={runPrediction} backTo={resultBack}/>;
    default: return <Auth profile={profile} go={go} onSignIn={signIn}/>;
  }
}
