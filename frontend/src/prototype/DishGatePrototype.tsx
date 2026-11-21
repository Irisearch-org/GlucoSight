// PROTOTYPE — throwaway (wayfinder #19: "What single confidence rule decides between
// accepting photo dishes and asking for manual entry?").
// Three variants of the meal-confirm step (flow step 2), switchable via ?variant=A|B|C,
// with stubbed VLM + embedding-matcher results via ?scenario=all|some|none.
// Open: http://localhost:5173/?prototype=dish-gate&variant=A&scenario=some
// Rule under test (measured on held-out photos, #18): accept an item when the embedding
// similarity >= 0.80 AND the top-1 minus top-2 margin >= 0.05; otherwise it needs manual entry.
import { useEffect, useState } from "react";
import { Badge, Button, ErrorBox, Icon, WizardLayout } from "../ui";
import { PrototypeSwitcher, useSearchParam } from "./PrototypeSwitcher";

const SIM_MIN = 0.8;
const MARGIN_MIN = 0.05;
const PORTION_LABELS = ["Sedikit", "Sedang", "Banyak"] as const;
const PORTION_SCALE = [0.5, 1, 1.5];

interface Row { id: string; name: string; servingG: number; carbs100g: number }
interface Detected { vlmName: string; sim: number; margin: number; top: Row[] }
interface Item { key: number; row: Row | null; vlmName?: string; suggestions: Row[]; portion: 0 | 1 | 2; fromPhoto: boolean }

// Stub rows; real values come from the locally built TKPI 2017 table.
const R = {
  nasi: { id: "AP0xx", name: "Nasi putih", servingG: 150, carbs100g: 40 },
  rendang: { id: "DP0xx", name: "Rendang daging sapi", servingG: 100, carbs100g: 4 },
  gulaiNangka: { id: "GS-012", name: "Gulai nangka", servingG: 100, carbs100g: 9 },
  sambalHijau: { id: "GS-021", name: "Sambal hijau (lado mudo)", servingG: 20, carbs100g: 8 },
  sambal: { id: "GS-020", name: "Sambal merah", servingG: 20, carbs100g: 10 },
  gulaiAyam: { id: "GS-014", name: "Gulai ayam", servingG: 150, carbs100g: 4 },
  gulaiKambing: { id: "GS-015", name: "Gulai kambing", servingG: 150, carbs100g: 3 },
  telurBalado: { id: "GS-030", name: "Telur balado", servingG: 60, carbs100g: 5 },
} satisfies Record<string, Row>;

const SCENARIOS: Record<string, { label: string; detected: Detected[] }> = {
  all: { label: "Semua cocok", detected: [
    { vlmName: "nasi putih", sim: 0.93, margin: 0.14, top: [R.nasi] },
    { vlmName: "rendang", sim: 0.89, margin: 0.11, top: [R.rendang] },
    { vlmName: "gulai nangka", sim: 0.85, margin: 0.07, top: [R.gulaiNangka] },
  ] },
  some: { label: "Sebagian cocok (nasi padang)", detected: [
    { vlmName: "nasi putih", sim: 0.93, margin: 0.14, top: [R.nasi] },
    { vlmName: "rendang", sim: 0.87, margin: 0.09, top: [R.rendang] },
    { vlmName: "sambal ijo", sim: 0.74, margin: 0.06, top: [R.sambalHijau, R.sambal] },
    { vlmName: "gulai", sim: 0.82, margin: 0.01, top: [R.gulaiAyam, R.gulaiKambing, R.gulaiNangka] },
  ] },
  none: { label: "Tidak dikenali", detected: [
    { vlmName: "lauk pauk", sim: 0.61, margin: 0.02, top: [R.telurBalado, R.gulaiAyam] },
  ] },
};

const passes = (d: Detected) => d.sim >= SIM_MIN && d.margin >= MARGIN_MIN;

function initialItems(detected: Detected[], keepFailing: boolean): Item[] {
  return detected
    .filter((d) => keepFailing || passes(d))
    .map((d, i) => ({ key: i, row: passes(d) ? d.top[0] : null, vlmName: d.vlmName, suggestions: d.top, portion: 1, fromPhoto: true }));
}

/** What the meal stage would send to the forecaster: the state this prototype exists to show. */
function StatePanel({ items, needsManual }: { items: Item[]; needsManual: boolean }) {
  const payload = {
    needs_manual_entry: needsManual,
    items: items.map((it) => ({ row_id: it.row?.id ?? null, name: it.row?.name ?? null, grams: it.row ? Math.round(it.row.servingG * PORTION_SCALE[it.portion]) : null, source: it.fromPhoto ? "photo" : "manual" })),
    carbs_g_total: Math.round(items.reduce((a, it) => a + (it.row ? (it.row.carbs100g * it.row.servingG * PORTION_SCALE[it.portion]) / 100 : 0), 0)),
  };
  return <details open style={{ marginTop: 18, fontSize: 12, background: "#f6f8f8", borderRadius: 12, padding: 10 }}>
    <summary><strong>PROTOTYPE — sent to the forecaster</strong></summary>
    <pre style={{ whiteSpace: "pre-wrap", margin: 0 }}>{JSON.stringify(payload, null, 2)}</pre>
  </details>;
}

function PortionPicker({ item, onPick }: { item: Item; onPick: (p: 0 | 1 | 2) => void }) {
  return <div className="mini-portions">{PORTION_LABELS.map((label, i) => <button key={label} className={item.portion === i ? "active" : ""} onClick={() => onPick(i as 0 | 1 | 2)}>
    {label}{item.row && <small>≈{Math.round(item.row.servingG * PORTION_SCALE[i])} g</small>}
  </button>)}</div>;
}

function ItemList({ items, setItems, unresolvedHint }: { items: Item[]; setItems: (i: Item[]) => void; unresolvedHint: (it: Item) => string }) {
  const update = (key: number, change: Partial<Item>) => setItems(items.map((i) => (i.key === key ? { ...i, ...change } : i)));
  return <div className="food-items">{items.map((it) => <div className="food-row" key={it.key} style={{ border: `1px solid ${it.row ? "#bfe8d6" : "#f6d9ba"}`, borderRadius: 14, padding: 10 }}>
    <div>
      <strong>{it.row ? it.row.name : unresolvedHint(it)}{" "}
        {it.row ? <Badge tone={it.fromPhoto ? "green" : "teal"}>{it.fromPhoto ? "Dari foto" : "Manual"}</Badge> : <Badge tone="amber">Perlu dipilih</Badge>}</strong>
      {!it.row && <div style={{ display: "flex", flexWrap: "wrap", gap: 6, margin: "6px 0" }}>
        {it.suggestions.map((s) => <button key={s.id} className="badge badge--gray" style={{ border: "1px solid #dde5e5", cursor: "pointer" }} onClick={() => update(it.key, { row: s, fromPhoto: false })}>{s.name}</button>)}
        <button className="badge badge--gray" style={{ border: "1px dashed #b8c4c4", cursor: "pointer" }} onClick={() => update(it.key, { row: { id: "GS-typed", name: "(ketik nama lain…)", servingG: 100, carbs100g: 10 }, fromPhoto: false })}>Cari lainnya…</button>
      </div>}
      {it.row && <PortionPicker item={it} onPick={(p) => update(it.key, { portion: p })}/>}
    </div>
    <button onClick={() => setItems(items.filter((i) => i.key !== it.key))} aria-label="Hapus"><Icon name="trash" size={19}/></button>
  </div>)}</div>;
}

/** A: one decision for the whole plate. Any failing item sends the whole plate to manual entry. */
function VariantA({ detected }: { detected: Detected[] }) {
  const allPass = detected.every(passes);
  const [manual, setManual] = useState(false);
  const [items, setItems] = useState<Item[]>(() => initialItems(detected, false));
  if (!allPass && !manual) {
    const ok = detected.filter(passes).length;
    return <>
      <ErrorBox title="Sebagian makanan belum dikenali">{ok} dari {detected.length} makanan dikenali dengan yakin. Masukkan makanan Anda secara manual; yang sudah dikenali sudah kami isi.</ErrorBox>
      <div className="retry-actions"><Button icon="camera">Foto ulang</Button><Button variant="secondary" icon="edit" onClick={() => setManual(true)}>Input manual</Button></div>
      <StatePanel items={[]} needsManual/>
    </>;
  }
  return <>
    <div className="manual-label"><strong>{manual ? "Input manual (terisi dari foto)" : "Makanan terdeteksi"}</strong><Badge tone="gray">{items.length} item</Badge></div>
    <ItemList items={items} setItems={setItems} unresolvedHint={() => ""}/>
    {manual && <Button variant="text" icon="plus">Tambah makanan lain</Button>}
    <div className="sticky-actions"><Button disabled={!items.length} icon="check">{manual ? "Lihat prediksi" : "Gunakan makanan ini"}</Button></div>
    <StatePanel items={items} needsManual={manual}/>
  </>;
}

/** B: per-item gate on the confirm screen. Confident items are accepted; each failing item is fixed inline. */
function VariantB({ detected }: { detected: Detected[] }) {
  const [items, setItems] = useState<Item[]>(() => initialItems(detected, true));
  const open = items.filter((i) => !i.row).length;
  return <>
    <div className="manual-label"><strong>Makanan di piring</strong>{open ? <Badge tone="amber">{open} perlu dipilih</Badge> : <Badge tone="green">Siap</Badge>}</div>
    <ItemList items={items} setItems={setItems} unresolvedHint={(it) => `Foto menebak “${it.vlmName}”. Mana yang benar?`}/>
    <Button variant="text" icon="plus">Tambah makanan yang terlewat</Button>
    <div className="sticky-actions"><Button disabled={!!open || !items.length} icon="check">Lihat prediksi</Button></div>
    <StatePanel items={items} needsManual={detected.some((d) => !passes(d))}/>
  </>;
}

/** C: no confirm screen. The photo pre-fills the manual-entry list; failing items become empty search slots. */
function VariantC({ detected }: { detected: Detected[] }) {
  const [items, setItems] = useState<Item[]>(() => initialItems(detected, true));
  const open = items.filter((i) => !i.row).length;
  return <>
    <div className="title-block"><span>Langkah 3 dari 4</span><h1>Periksa daftar makanan</h1><p>Kami mengisi daftar dari foto. Lengkapi yang kosong, ubah porsinya bila perlu.</p></div>
    <div className="search-wrap"><Icon name="search"/><input placeholder="Cari atau ketik nama makanan…"/></div>
    <ItemList items={items} setItems={setItems} unresolvedHint={() => "Belum diketahui. Ketik namanya"}/>
    <div className="sticky-actions"><Button disabled={!!open || !items.length}>Lihat prediksi</Button></div>
    <StatePanel items={items} needsManual={detected.some((d) => !passes(d))}/>
  </>;
}

const VARIANTS = [
  { key: "A", name: "Satu keputusan per piring" },
  { key: "B", name: "Per item di layar konfirmasi" },
  { key: "C", name: "Foto mengisi daftar manual" },
];

export default function DishGatePrototype() {
  const [, force] = useState(0);
  useEffect(() => {
    const on = () => force((n) => n + 1);
    window.addEventListener("prototype-nav", on);
    return () => window.removeEventListener("prototype-nav", on);
  }, []);
  const [variant, setVariant] = useSearchParam("variant", "A");
  const [scenario, setScenario] = useSearchParam("scenario", "some");
  const detected = SCENARIOS[scenario]?.detected ?? SCENARIOS.some.detected;
  const k = `${variant}-${scenario}`;
  return <>
    <WizardLayout current={2} title="Konfirmasi Makanan" back={() => undefined}>
      <div style={{ display: "flex", gap: 6, marginBottom: 12, flexWrap: "wrap" }}>
        {Object.entries(SCENARIOS).map(([id, s]) => <button key={id} className={`badge ${id === scenario ? "badge--teal" : "badge--gray"}`} style={{ border: "none", cursor: "pointer" }} onClick={() => setScenario(id)}>Skenario: {s.label}</button>)}
      </div>
      <div className="food-hero" style={{ height: 120, background: "#eef2f2", borderRadius: 16, display: "grid", placeItems: "center", marginBottom: 14, color: "#789" }}>📷 foto piring (stub)</div>
      {variant === "A" && <VariantA key={k} detected={detected}/>}
      {variant === "B" && <VariantB key={k} detected={detected}/>}
      {variant === "C" && <VariantC key={k} detected={detected}/>}
    </WizardLayout>
    <PrototypeSwitcher variants={VARIANTS} current={variant} onChange={setVariant}/>
  </>;
}
