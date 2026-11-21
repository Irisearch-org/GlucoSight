// Step 2: the meal. Photo (camera → per-item confirm) or manual item-by-item entry.
// Every food is a row of the dish table (row_id) with grams or a portion of its standard serving.
import { useEffect, useRef, useState } from "react";
import { identifyMealPhoto, searchDishes, type DishRow } from "../api";
import {
  PORTION_LABELS, PORTION_SCALE, entryFor, entryFromPhoto, entryGrams, errorText, fmt, isResolved, mealNutrition, pickRow,
  type Draft, type MealEntry, type Portion,
} from "../flow";
import { FoodIcon } from "../foodIcons";
import { frameToDataUrl, photoToDataUrl } from "../image";
import { nutritionFor, type Nutrition } from "../nutrition";
import { cameraSupported } from "../ppg";
import { Badge, Button, ErrorBox, Icon, WizardLayout, type Screen } from "../ui";

type Patch = (p: Partial<Draft>) => void;

/** Switch to manual entry, carrying over the foods that already have a row; guesses still unresolved are dropped. */
function toManual(draft: Draft, patch: Patch, go: (s: Screen) => void) {
  patch({ foodMode: "manual", items: draft.items.filter((e) => e.row) });
  go("manual");
}

// ------------------------------------------------------------------ pieces

/** Autocomplete over the dish table (GET /dishes), debounced. Picking a result calls `onPick` and clears the box. */
function DishSearch({ onPick, autoFocus = false }: { onPick: (row: DishRow) => void; autoFocus?: boolean }) {
  const [query, setQuery] = useState("");
  const [focused, setFocused] = useState(false);
  const [results, setResults] = useState<DishRow[] | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    const q = query.trim();
    if (!q) { setResults(null); setError(""); return; }
    const ctl = new AbortController();
    const timer = window.setTimeout(() => {
      searchDishes(q, 10, ctl.signal)
        .then((rows) => { setResults(rows); setError(""); })
        .catch((e) => { if (!ctl.signal.aborted) { setResults(null); setError(errorText(e)); } });
    }, 250);
    return () => { window.clearTimeout(timer); ctl.abort(); };
  }, [query]);

  const pick = (row: DishRow) => { setQuery(""); setResults(null); onPick(row); };
  const open = focused && query.trim() !== "" && (results !== null || error !== "");

  return <div className="search-wrap"><Icon name="search"/>
    <input value={query} autoFocus={autoFocus} onChange={(e) => setQuery(e.target.value)} onFocus={() => setFocused(true)} onBlur={() => setFocused(false)} placeholder="Cari atau ketik nama makanan…"/>
    {open && <div className="suggestions" onMouseDown={(e) => e.preventDefault()}>
      {error && <span className="suggestions__label">{error}</span>}
      {results?.length === 0 && <span className="suggestions__label">Makanan tidak ditemukan. Coba kata lain.</span>}
      {results?.map((r) => <button onClick={() => pick(r)} key={r.id}>
        <span className="suggestions__item"><FoodIcon size={30}/>{r.name}{r.serving_g !== null && <small> · ≈{Math.round(r.serving_g)} g</small>}</span><Icon name="plus" size={17}/>
      </button>)}
    </div>}
  </div>;
}

function NutritionRows({ n }: { n: Nutrition }) {
  const rows = [["Karbohidrat", n.carbs, 120], ["Protein", n.protein, 40], ["Lemak", n.fat, 40], ["Serat", n.fibre, 12]] as const;
  return <div className="macro-list">{rows.map(([name, value, ref]) => <div className="macro" key={name}>
    <span>{name}</span><div><i style={{ width: `${value === null ? 0 : Math.min(100, (value / ref) * 100)}%` }}/></div><strong>{value === null ? "—" : `${fmt(value)} g`}</strong>
  </div>)}</div>;
}

function Totals({ items }: { items: MealEntry[] }) {
  const { total, counted } = mealNutrition(items);
  if (!counted) return null;
  return <section className="nutrition-reveal">
    <div className="nutrition-reveal__head"><div><span>{counted < items.length ? `Berdasarkan ${counted} dari ${items.length} makanan` : `Berdasarkan ${counted} makanan`}</span><strong>Total nutrisi</strong></div><Badge tone="teal">{Math.round(total.kcal)} kkal</Badge></div>
    <NutritionRows n={total}/>
  </section>;
}

/** A food with a row: portion or gram input, and what that weight is worth. */
function EntryRow({ entry, detail, onChange, onRemove }: {
  entry: MealEntry; detail: boolean; onChange: (c: Partial<MealEntry>) => void; onRemove: () => void;
}) {
  const row = entry.row as DishRow;
  const grams = entryGrams(entry);
  const n = grams && grams > 0 ? nutritionFor(row, grams) : null;
  return <div className="food-row">
    <FoodIcon size={40}/>
    <div>
      <strong>{row.name} <Badge tone={entry.source === "photo" ? "green" : "teal"}>{entry.source === "photo" ? "Dari foto" : "Manual"}</Badge></strong>
      {row.serving_g !== null && <div className="mini-segments"><button className={entry.mode === "portion" ? "active" : ""} onClick={() => onChange({ mode: "portion" })}>Perkiraan</button><button className={entry.mode === "gram" ? "active" : ""} onClick={() => onChange({ mode: "gram" })}>Gram</button></div>}
      {entry.mode === "portion" && row.serving_g !== null
        ? <div className="mini-portions">{PORTION_LABELS.map((label, i) => <button key={label} className={entry.portion === i ? "active" : ""} onClick={() => onChange({ portion: i as Portion })}>
            {label}<small>≈{Math.round(row.serving_g! * PORTION_SCALE[i])} g</small>
          </button>)}</div>
        : <div className="mini-gram"><input type="number" min={1} inputMode="decimal" value={entry.grams} onChange={(e) => onChange({ grams: e.target.value })}/><b>gram</b></div>}
      {n && (detail
        ? <div className="food-nutri">
            <div className="food-nutri__head"><span>Rincian nutrisi</span><b>≈ {Math.round(grams ?? 0)} g · {Math.round(n.kcal)} kkal</b></div>
            <NutritionRows n={n}/>
          </div>
        : <p className="food-nutri__line">≈ {Math.round(grams ?? 0)} g · {Math.round(n.kcal)} kkal · {fmt(n.carbs)} g karbohidrat</p>)}
    </div>
    <button onClick={onRemove} aria-label={`Hapus ${row.name}`}><Icon name="trash" size={19}/></button>
  </div>;
}

/** A photo guess the matcher was not sure about: the user picks the right dish. */
function PendingRow({ entry, searching, onSearch, onPick, onRemove }: {
  entry: MealEntry; searching: boolean; onSearch: () => void; onPick: (row: DishRow) => void; onRemove: () => void;
}) {
  return <div className="food-row food-row--pending">
    <FoodIcon size={40}/>
    <div>
      <strong>Foto menebak “{entry.vlmName}”. Mana yang benar?</strong>
      <div className="dish-chips">
        {entry.suggestions.map((s) => <button className="chip" key={s.id} onClick={() => onPick(s)}>{s.name}</button>)}
        <button className="chip chip--dashed" onClick={onSearch}>Cari lainnya…</button>
      </div>
      {searching && <DishSearch autoFocus onPick={onPick}/>}
    </div>
    <button onClick={onRemove} aria-label={`Hapus ${entry.vlmName}`}><Icon name="trash" size={19}/></button>
  </div>;
}

// ------------------------------------------------------------------ camera

type CamState = "starting" | "live" | "unavailable";

export function Camera({ draft, patch, go }: { draft: Draft; patch: Patch; go: (s: Screen) => void }) {
  const [shot, setShot] = useState<string | null>(null);
  const [cam, setCam] = useState<CamState>("starting");
  const [camProblem, setCamProblem] = useState("");
  const [loading, setLoading] = useState(false);
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
      const res = await identifyMealPhoto(photo);
      patch({ photo, foodMode: "photo", vlmModelVersion: res.vlm_model_version, items: res.items.map(entryFromPhoto) });
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
    <Button variant="text" onClick={() => toManual(draft, patch, go)}>Masukkan makanan secara manual</Button>
  </WizardLayout>;
}

// ----------------------------------------------------------------- confirm

/** The photo's items, one by one: the dishes the matcher is sure of are in, every other one needs the user's pick. */
export function Confirm({ draft, patch, go, onPredict }: { draft: Draft; patch: Patch; go: (s: Screen) => void; onPredict: () => void }) {
  const items = draft.items;
  // A key = the entry whose "Cari lainnya…" search is open; "add" = the search for a missed food.
  const [searching, setSearching] = useState<number | "add" | null>(null);
  const update = (key: number, change: Partial<MealEntry>) => patch({ items: items.map((e) => (e.key === key ? { ...e, ...change } : e)) });
  const remove = (key: number) => patch({ items: items.filter((e) => e.key !== key) });
  const open = items.filter((e) => !e.row).length;
  const ready = items.length > 0 && items.every(isResolved);

  return <WizardLayout current={2} title="Konfirmasi Makanan" back={() => go("camera")}>
    <div className="food-hero">{draft.photo ? <img className="camera-photo" src={draft.photo} alt="Foto makanan"/> : null}
      <button onClick={() => go("camera")}><Icon name="edit" size={17}/> Ganti foto</button></div>
    {items.length === 0
      ? <>
        <ErrorBox title="Makanan belum dikenali">Ambil ulang foto dari atas dengan cahaya cukup, atau tambahkan makanan secara manual.</ErrorBox>
        <div className="retry-actions"><Button onClick={() => go("camera")} icon="camera">Foto ulang</Button><Button variant="secondary" icon="edit" onClick={() => toManual(draft, patch, go)}>Input manual</Button></div>
      </>
      : <>
        <div className="manual-label"><strong>Makanan di piring</strong>{open ? <Badge tone="amber">{open} perlu dipilih</Badge> : <Badge tone="green">Siap</Badge>}</div>
        <div className="food-items">{items.map((e) => e.row
          ? <EntryRow key={e.key} entry={e} detail={false} onChange={(c) => update(e.key, c)} onRemove={() => remove(e.key)}/>
          : <PendingRow key={e.key} entry={e} searching={searching === e.key}
              onSearch={() => setSearching(e.key)}
              onPick={(row) => { patch({ items: items.map((x) => (x.key === e.key ? pickRow(x, row) : x)) }); setSearching(null); }}
              onRemove={() => { remove(e.key); setSearching(null); }}/>)}</div>
      </>}
    {searching === "add" && <DishSearch autoFocus onPick={(row) => { patch({ items: [...items, entryFor(row, "manual")] }); setSearching(null); }}/>}
    <Button variant="text" icon="plus" onClick={() => setSearching(searching === "add" ? null : "add")}>Tambah makanan yang terlewat</Button>
    <Totals items={items}/>
    <div className="sticky-actions"><Button disabled={!ready} onClick={onPredict} icon="check">Lihat prediksi</Button><Button variant="text" onClick={() => toManual(draft, patch, go)}>Masih kurang sesuai? Input manual</Button></div>
  </WizardLayout>;
}

// ------------------------------------------------------------------ manual

export function Manual({ draft, patch, go, onPredict }: { draft: Draft; patch: Patch; go: (s: Screen) => void; onPredict: () => void }) {
  const items = draft.items;
  const update = (key: number, change: Partial<MealEntry>) => patch({ items: items.map((e) => (e.key === key ? { ...e, ...change } : e)) });
  const ready = items.length > 0 && items.every(isResolved);

  return <WizardLayout current={2} title="Input Manual" back={() => go("camera")}>
    <div className="title-block"><span>Langkah 3 dari 4</span><h1>Tambahkan makanan</h1><p>Cari makanan atau bahan Anda, lalu pilih porsi atau berat.</p></div>
    <DishSearch onPick={(row) => patch({ foodMode: "manual", items: [...items, entryFor(row, "manual")] })}/>
    <div className="manual-label"><strong>Makanan ditambahkan</strong><Badge tone="gray">{items.length} item</Badge></div>
    <div className="food-items">{items.map((e) => <EntryRow key={e.key} entry={e} detail onChange={(c) => update(e.key, c)} onRemove={() => patch({ items: items.filter((x) => x.key !== e.key) })}/>)}</div>
    {!items.length && <p className="helper">Belum ada makanan. Ketik nama makanan di kolom pencarian.</p>}
    <Totals items={items}/>
    <div className="sticky-actions"><Button disabled={!ready} onClick={onPredict}>Lihat prediksi</Button></div>
  </WizardLayout>;
}
