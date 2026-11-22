// Step 3: the forecast. Builds the /predict request from the draft and shows both horizons.
import { useEffect, useState } from "react";
import type { GlucoseRange, PredictBody, PredictGlucose, PredictResponse } from "../api";
import {
  RANGE_LABEL, RANGE_TONE, SOURCE_LABEL, entryGrams, glucoseOf, num, type Draft,
} from "../flow";
import type { HistoryRecord, Profile } from "../storage";
import { Badge, Button, Disclaimer, ErrorBox, Icon, WizardLayout, type Screen } from "../ui";

/** What only the live response knows; a record reopened from history has none of it. */
export interface ForecastDetail {
  band60: number;
  band120: number;
  outsideTraining: boolean;
}

export type ResultState =
  | { status: "loading" }
  | { status: "error"; message: string }
  | { status: "ready"; record: HistoryRecord; detail: ForecastDetail | null; fromHistory: boolean };

export function buildPredictBody(d: Draft, profile: Profile): PredictBody {
  const g = glucoseOf(d);
  const glucose: PredictGlucose = d.source === "ppg" && d.ppg
    ? { source: "ppg", reading: d.ppg }
    : { source: d.source === "cgm" ? "cgm" : "meter", value_mgdl: g.value ?? 0, timestamp: new Date(g.time ?? Date.now()).toISOString() };
  const items = d.items.flatMap((e) => {
    const grams = entryGrams(e);
    if (!e.row || !grams) return [];
    return [{
      row_id: e.row.id, grams,
      ...(e.mode === "portion" ? { portion: e.portion } : {}),
      source: e.source,
      ...(e.source === "photo" && e.similarity !== undefined ? { similarity: e.similarity, margin: e.margin } : {}),
    }];
  });
  return {
    t0_timestamp: new Date().toISOString(),
    glucose,
    meal: { items, vlm_model_version: items.some((i) => i.source === "photo") ? d.vlmModelVersion : null },
    profile: {
      age_years: num(profile.age),
      sex: profile.gender === "Pria" ? "M" : profile.gender === "Wanita" ? "F" : null,
      height_cm: num(profile.heightCm),
      weight_kg: num(profile.weightKg),
      diabetes_type: profile.diabetesType,
    },
  };
}

export function toRecord(res: PredictResponse, d: Draft): HistoryRecord {
  const source = d.source === "" ? "meter" : d.source;
  return {
    id: `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`,
    createdAt: new Date().toISOString(),
    food: res.meal.items.map((i) => i.row.name).join(", ").slice(0, 255),
    portion: `${res.meal.items.length} item · ${Math.round(res.meal.totals.carbs_g)} g karbohidrat`,
    t60: res.forecast.t60.glucose_mgdl,
    t120: res.forecast.t120.glucose_mgdl,
    range: res.forecast.t60.range,
    g0: Math.round(res.g0.value_mgdl),
    sourceId: source,
    basis: `Prakiraan LightGBM T+60 dan T+120; gula awal dari ${SOURCE_LABEL[source]}`,
    usageMessage: res.disclaimer,
    modelVersion: res.forecast.t60.model_version.slice(0, 100),
  };
}

export const toDetail = (res: PredictResponse): ForecastDetail => ({
  band60: res.forecast.t60.error_band_mgdl,
  band120: res.forecast.t120.error_band_mgdl,
  outsideTraining: res.outside_training,
});

const rangeOf = (v: number): GlucoseRange => (v < 140 ? "normal" : v < 200 ? "borderline" : "high");

/** Marker position on the 3-band bar: bands are equal thirds, split at 140 and 200 mg/dL. */
function markerPercent(v: number): number {
  if (v < 140) return Math.max(2, ((v - 70) / 70) * 33.3);
  if (v < 200) return 33.3 + ((v - 140) / 60) * 33.3;
  return Math.min(98, 66.6 + ((v - 200) / 60) * 33.3);
}

/** Counts from the pre-meal glucose up to the predicted value; the range marker and colour follow along. */
function ResultGauge({ from, to, band, label }: { from: number; to: number; band: number | null; label: string }) {
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
    <div className="result-reading">
      <div className={`result-number result-number--${rangeOf(v)}`}>{Math.round(v)} <small>mg/dL</small></div>
      {band !== null && <p className="result-band">± {Math.round(band)} mg/dL</p>}
    </div>
    <p>{label}</p>
    <div className="range"><div><i style={{ left: `${markerPercent(v)}%` }}/></div><span><b>Normal</b><b>Waspada</b><b>Tinggi</b></span></div>
  </>;
}

/** Shown while the prediction is computed: a glucose curve drawn over and over, with a dot riding its head. Reduced motion: a static, fully drawn curve. */
function LoadingCurve() {
  const d = "M6 66C40 64 56 58 84 36S128 10 152 16S186 42 194 44";
  const reduce = typeof window !== "undefined" && !!window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
  return <svg className="result-curve" viewBox="0 0 200 80" aria-hidden="true">
    <path d={d} fill="none" stroke="#d6ecea" strokeWidth="3" strokeLinecap="round"/>
    <path id="rc-path" className={reduce ? undefined : "result-curve__line"} d={d} pathLength="1" fill="none" stroke="#12a9a3" strokeWidth="3" strokeLinecap="round"/>
    {reduce
      ? <circle cx="194" cy="44" r="5.5" fill="#fff" stroke="#0FA0A0" strokeWidth="3"/>
      : <circle r="5.5" fill="#fff" stroke="#0FA0A0" strokeWidth="3">
        <animateMotion dur="2.4s" repeatCount="indefinite" calcMode="spline" keyTimes="0;1" keySplines=".5 .1 .3 1"><mpath href="#rc-path"/></animateMotion>
      </circle>}
  </svg>;
}

export function Result({ state, nav, go, onRetry, backTo }: {
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
      const { record: r, detail } = state;
      return <>
        <div className="result-intro"><Badge tone="teal"><Icon name="sparkle" size={13}/> {state.fromHistory ? "Prediksi tersimpan" : "Prediksi selesai"}</Badge><h1>Respons gula darah Anda</h1></div>
        {r.sourceId === "ppg" && <div className="tip-card tip-card--warn"><Icon name="info"/><div><strong>Prediksi dimulai dari estimasi kasar</strong><p>Prediksi ini dimulai dari estimasi kasar kamera dan kurang andal. Gunakan glukometer bila tersedia.</p></div></div>}
        {state.fromHistory && <div className="tip-card"><Icon name="info"/><p>Rentang galat hanya tersedia pada prediksi baru.</p></div>}
        {detail?.outsideTraining && <div className="tip-card tip-card--warn"><Icon name="info"/><div><strong>Model belum dilatih untuk jenis diabetes ini</strong><p>Prediksi tetap ditampilkan, tetapi bisa kurang tepat untuk Anda.</p></div></div>}
        <article className="result-card">
          <div className="result-card__top"><span>Prediksi T+60</span><Badge tone={RANGE_TONE[r.range]}>{RANGE_LABEL[r.range]}</Badge></div>
          <ResultGauge from={r.g0} to={r.t60} band={detail?.band60 ?? null} label="Perkiraan 60 menit setelah suapan pertama"/>
        </article>
        <article className="result-card">
          <div className="result-card__top"><span>Prediksi T+120</span><Badge tone={RANGE_TONE[rangeOf(r.t120)]}>{RANGE_LABEL[rangeOf(r.t120)]}</Badge></div>
          <ResultGauge from={r.g0} to={r.t120} band={detail?.band120 ?? null} label="Perkiraan 120 menit setelah suapan pertama"/>
        </article>
        <div className="baseline"><span className="baseline__icon"><Icon name="drop"/></span><div><span>Gula sebelum makan</span><strong>{r.g0} mg/dL <small>· {SOURCE_LABEL[r.sourceId]}</small></strong></div>{r.sourceId !== "ppg" && <Badge tone="green">Terukur</Badge>}</div>
        <section className="inputs-used"><strong>Data yang digunakan</strong>
          <div><span><Icon name="drop"/> Gula darah</span><b>{r.g0} mg/dL</b></div>
          <div><span><Icon name="gallery"/> Makanan</span><b>{r.food} · {r.portion}</b></div></section>
        <Disclaimer detail={r.usageMessage}/>
        <div className="sticky-actions"><Button onClick={() => nav("home")}>Selesai</Button><Button variant="secondary" onClick={() => nav("source")}>Prediksi lagi</Button></div>
      </>;
    })()}
  </WizardLayout>;
}
