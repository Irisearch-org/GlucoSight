// Step 1: current glucose. Source choice, glucometer/CGM entry, and the finger-camera (PPG) scan.
import { useEffect, useRef, useState } from "react";
import { scanGlucosePpg, type G0Reading, type PpgCapture } from "../api";
import {
  FRESH_WITHIN_MIN, GLUCOSE_MAX, GLUCOSE_MIN, SOURCE_LABEL, errorText, glucoseOf, ppgRange, validGlucose, type Draft,
} from "../flow";
import { cameraSupported, PPG_SECONDS, PpgRecorder, type FrameInfo } from "../ppg";
import type { SourceId } from "../storage";
import { ageMinutes, describeAge, formatWhen } from "../time";
import { Badge, Button, ErrorBox, Field, Icon, WizardLayout, type IconName, type Screen, type Tone } from "../ui";

export const sourceOptions: { id: SourceId; icon: IconName; title: string; sub: string; badge?: string; tone?: Tone }[] = [
  { id: "meter", icon: "meter", title: "Glukometer", sub: "Masukkan hasil fingerstick", badge: "Terukur", tone: "green" },
  { id: "cgm", icon: "pulse", title: "CGM", sub: "Masukkan pembacaan terakhir dari aplikasi CGM", badge: "Terukur", tone: "green" },
  { id: "ppg", icon: "camera", title: "Scan Jari (PPG)", sub: "Estimasi kasar lewat kamera ponsel · butuh Chrome di Android" },
];

export function SourceCard({ option, selected, onClick }: { option: (typeof sourceOptions)[number]; selected: boolean; onClick: () => void }) {
  return <button className={`option-card ${selected ? "is-selected" : ""}`} onClick={onClick}>
    <span className="option-card__icon"><Icon name={option.icon}/></span>
    <span className="option-card__copy"><span><strong>{option.title}</strong>{option.badge && <Badge tone={option.tone}>{option.badge}</Badge>}</span><small>{option.sub}</small></span>
    <span className="radio">{selected && <i/>}</span>
  </button>;
}

/** "112 mg/dL (87–137)": the scan value with the model's error range. */
const ppgText = (r: G0Reading): string => {
  const range = ppgRange(r);
  return `${Math.round(r.value_mgdl)} mg/dL${range ? ` (${range.lo}–${range.hi})` : ""}`;
};

// ------------------------------------------------------------------ source

export function Source({ draft, patch, go }: { draft: Draft; patch: (p: Partial<Draft>) => void; go: (s: Screen) => void }) {
  const g = glucoseOf(draft);
  const problem = draft.source === "ppg" ? (draft.ppg ? null : "Lakukan scan jari terlebih dahulu.") : validGlucose(g.value, g.time);
  const touched = (draft.source === "meter" && draft.meterValue !== "") || (draft.source === "cgm" && draft.cgmValue !== "");
  return <WizardLayout current={0} title="Prediksi Baru" back={() => go("home")}>
    <div className="title-block"><span>Langkah 1 dari 4</span><h1>Dari mana data gula darah Anda?</h1><p>Pilih sumber pengukuran sebelum makan.</p></div>
    <div className="option-list">{sourceOptions.map((o) => <SourceCard key={o.id} option={o} selected={draft.source === o.id}
      onClick={() => { patch({ source: o.id }); if (o.id === "ppg") go("scan"); }}/>)}</div>
    {draft.source === "meter" && <section className="reveal-card">
      <div className="reveal-card__head"><Icon name="drop"/><div><strong>Hasil pengukuran</strong><span>Masukkan data terbaru</span></div></div>
      <Field label="Gula darah" type="number" inputMode="decimal" value={draft.meterValue} onChange={(v) => patch({ meterValue: v })} suffix="mg/dL" min={GLUCOSE_MIN} max={GLUCOSE_MAX}/>
      <Field label="Waktu tes" type="datetime-local" value={draft.meterTime} onChange={(v) => patch({ meterTime: v })}/>
      <p className="helper"><Icon name="clock" size={15}/> Harus diukur &lt; {FRESH_WITHIN_MIN} menit sebelum makan</p>
      {touched && problem && <p className="field-hint">{problem}</p>}
    </section>}
    {draft.source === "cgm" && <section className="reveal-card">
      <div className="reveal-card__head"><Icon name="pulse"/><div><strong>Pembacaan CGM terakhir</strong><span>Salin dari aplikasi CGM Anda</span></div></div>
      <Field label="Gula darah" type="number" inputMode="decimal" value={draft.cgmValue} onChange={(v) => patch({ cgmValue: v })} suffix="mg/dL" min={GLUCOSE_MIN} max={GLUCOSE_MAX}/>
      <Field label="Waktu pembacaan" type="datetime-local" value={draft.cgmTime} onChange={(v) => patch({ cgmTime: v })}/>
      <p className="helper"><Icon name="info" size={15}/> Sambungan otomatis ke perangkat CGM belum tersedia.</p>
      {touched && problem && <p className="field-hint">{problem}</p>}
    </section>}
    {draft.source === "ppg" && draft.ppg && <section className="reveal-card">
      <div className="reveal-card__head"><Icon name="camera"/><div><strong>Estimasi kasar dari scan jari</strong><span>≈ {ppgText(draft.ppg)}</span></div></div>
      <button className="inline-action" onClick={() => go("scan")}><Icon name="camera" size={16}/> Pindai ulang</button>
    </section>}
    <div className="sticky-actions"><Button disabled={!draft.source || !!problem} onClick={() => go("summary")}>Lanjut</Button></div>
  </WizardLayout>;
}

// ----------------------------------------------------------------- PPG scan

type ScanPhase = "ready" | "starting" | "aim" | "recording" | "processing" | "done" | "error";

function Wave({ data }: { data: number[] }) {
  if (data.length < 2) return <svg className="wave" viewBox="0 0 300 70" preserveAspectRatio="none"><path d="M0 35h300"/></svg>;
  const min = Math.min(...data), max = Math.max(...data), span = max - min || 1;
  const points = data.map((v, i) => `${((i / (data.length - 1)) * 300).toFixed(1)},${(64 - ((v - min) / span) * 58).toFixed(1)}`).join(" ");
  return <svg className="wave" viewBox="0 0 300 70" preserveAspectRatio="none"><polyline points={points}/></svg>;
}

const MANUAL_FALLBACK = "Masukkan hasil glukometer/CGM";

export function Scan({ go, onUse, onManual }: { go: (s: Screen) => void; onUse: (r: G0Reading) => void; onManual: () => void }) {
  const [phase, setPhase] = useState<ScanPhase>("ready");
  const [frame, setFrame] = useState<FrameInfo | null>(null);
  const [error, setError] = useState("");
  const [outcome, setOutcome] = useState<G0Reading | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const recRef = useRef<PpgRecorder | null>(null);

  const [fallbackFirst, setFallbackFirst] = useState(false);
  useEffect(() => () => recRef.current?.dispose(), []);
  // The camera is closed on every error, so the last frame (and its "Jari terdeteksi") is stale.
  useEffect(() => { if (phase === "error") setFrame(null); }, [phase]);

  const process = async (capture: PpgCapture) => {
    setPhase("processing");
    try {
      setOutcome(await scanGlucosePpg(capture));
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
    setFallbackFirst(false);
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
      if (!rec.torchVerified) {
        rec.dispose();
        setError("Flash tidak menyala di perangkat atau browser ini, sehingga jari tidak dapat dipindai. Coba Chrome di Android, atau masukkan hasil glukometer/CGM.");
        setFallbackFirst(true);
        setPhase("error");
        return;
      }
      setPhase("aim");
    } catch (e) {
      rec.dispose();
      setError(e instanceof DOMException && e.name === "NotAllowedError"
        ? "Izin kamera ditolak. Izinkan akses kamera di pengaturan browser."
        : "Kamera tidak dapat dibuka di perangkat ini.");
      setFallbackFirst(true);
      setPhase("error");
    }
  };

  const retry = () => { setFrame(null); setPhase("ready"); };

  const cancel = () => { recRef.current?.dispose(); setFrame(null); setPhase("ready"); };
  const remaining = Math.max(0, Math.ceil(PPG_SECONDS - (frame?.elapsed ?? 0)));
  const live = phase === "aim" || phase === "recording";
  const range = outcome ? ppgRange(outcome) : null;

  const heading = phase === "done" ? "Pemindaian selesai"
    : phase === "recording" ? "Tetap tahan jari Anda"
    : phase === "processing" ? "Memproses sinyal…"
    : phase === "starting" ? "Memeriksa flash…"
    : phase === "aim" ? "Tutup lensa dengan jari" : "Posisikan ujung jari";
  const copy = phase === "done" ? "Kami berhasil menangkap sinyal Anda."
    : phase === "recording" ? "Jangan bergerak agar sinyal tetap stabil."
    : phase === "aim" ? "Tutup kamera belakang dan lampu flash sepenuhnya, lalu mulai merekam."
    : phase === "starting" ? "Jangan tutup lensa dulu: kami memastikan flash benar-benar menyala."
    : phase === "processing" ? "Menghitung estimasi gula darah dari sinyal jari."
    : "Buka kamera lebih dulu, lalu tutup kamera belakang dan lampu flash dengan ujung jari.";

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
      <div className="flash-note"><span className="flash-icon">ϟ</span> {live ? "Flash menyala" : "Flash dinyalakan dan diperiksa otomatis saat kamera dibuka"}</div>
    </div> : outcome && <div className="scan-result">
      <div className="success-ring"><Icon name="check" size={31}/></div>
      <p>Estimasi kasar dari kamera</p>
      <strong>≈ {Math.round(outcome.value_mgdl)} <small>mg/dL{range ? ` (${range.lo}–${range.hi})` : ""}</small></strong>
      <p className="scan-result__note"><Icon name="info" size={15}/> Glukometer lebih akurat; pakai bila tersedia.</p>
      {outcome.heart_rate_bpm !== null && <p className="scan-result__secondary">Detak jantung {Math.round(outcome.heart_rate_bpm)} bpm</p>}
    </div>}
    {phase === "error" && <ErrorBox title="Pemindaian belum berhasil">{error}</ErrorBox>}
    {phase === "ready" && !cameraSupported() && <p className="field-hint">Kamera belum dapat dipakai di halaman ini.</p>}
    <div className="sticky-actions">
      {phase === "ready" && <Button icon="camera" variant={cameraSupported() ? "primary" : "secondary"} disabled={!cameraSupported()} onClick={openCamera}>Mulai pindai</Button>}
      {phase === "ready" && !cameraSupported() && <Button onClick={onManual}>{MANUAL_FALLBACK}</Button>}
      {phase === "starting" && <Button disabled>Memeriksa flash…</Button>}
      {phase === "aim" && <Button disabled={!frame?.fingerOn} onClick={() => { recRef.current?.record(); setPhase("recording"); }}>Mulai merekam</Button>}
      {phase === "recording" && <><Button disabled>Merekam… {remaining} detik</Button><Button variant="text" onClick={cancel}>Batalkan</Button></>}
      {phase === "processing" && <Button disabled>Memproses…</Button>}
      {phase === "done" && outcome && <><Button onClick={() => onUse(outcome)}>Gunakan hasil</Button><Button variant="text" onClick={() => { setOutcome(null); setFrame(null); setPhase("ready"); }}>Ulangi scan</Button></>}
      {phase === "error" && fallbackFirst && <><Button onClick={onManual}>{MANUAL_FALLBACK}</Button><Button variant="secondary" onClick={retry}>Ulangi scan</Button></>}
      {phase === "error" && !fallbackFirst && <><Button onClick={retry}>Ulangi scan</Button><Button variant="text" onClick={onManual}>{MANUAL_FALLBACK}</Button></>}
    </div>
  </WizardLayout>;
}

// ----------------------------------------------------------------- summary

export function Summary({ draft, go }: { draft: Draft; go: (s: Screen) => void }) {
  const g = glucoseOf(draft);
  const source = (draft.source || "meter") as SourceId;
  const fromScan = source === "ppg";
  const age = g.time ? ageMinutes(g.time) : 0;
  const tip = fromScan
    ? { cls: "tip-card--warn", icon: "info" as IconName, title: "Estimasi kasar", text: "Hasil scan jari bisa meleset ±25 mg/dL. Gunakan glukometer bila tersedia." }
    : { cls: "", icon: "check" as IconName, title: "Data masih cukup baru", text: `Pengukuran dilakukan kurang dari ${FRESH_WITHIN_MIN} menit sebelum makan.` };
  const range = draft.ppg ? ppgRange(draft.ppg) : null;
  return <WizardLayout current={1} title="Ringkasan Data" back={() => go("source")}>
    <div className="title-block"><span>Langkah 2 dari 4</span><h1>Pastikan data sudah benar</h1><p>Data ini menjadi dasar prediksi Anda.</p></div>
    <article className="summary-card">
      <div className="summary-card__top"><div className="drop-box"><Icon name={source === "ppg" ? "camera" : source === "cgm" ? "pulse" : "drop"}/></div>
        <div><span>Gula sebelum makan</span><strong>{fromScan ? "≈ " : ""}{g.value} <small>mg/dL{range ? ` (${range.lo}–${range.hi})` : ""}</small></strong></div>{!fromScan && <Badge tone="green">Terukur</Badge>}</div>
      <div className="summary-card__rows">
        <div><span>Sumber data</span><strong>{SOURCE_LABEL[source]}</strong></div>
        <div><span>Waktu pengukuran</span><strong>{g.time ? describeAge(age) : "baru saja"}</strong></div>
        <div><span>Tanggal &amp; waktu</span><strong>{formatWhen(g.time ? new Date(g.time).toISOString() : new Date().toISOString())}</strong></div>
        {fromScan && draft.ppg?.heart_rate_bpm != null && <div><span>Detak jantung</span><strong>{Math.round(draft.ppg.heart_rate_bpm)} bpm</strong></div>}
      </div>
    </article>
    <div className={`tip-card ${tip.cls}`}><Icon name={tip.icon}/><div><strong>{tip.title}</strong><p>{tip.text}</p></div></div>
    <div className="sticky-actions split"><Button variant="secondary" onClick={() => go("source")} icon="edit">Ubah</Button><Button onClick={() => go("camera")}>Lanjut</Button></div>
  </WizardLayout>;
}
