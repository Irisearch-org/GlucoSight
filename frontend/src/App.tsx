import { useEffect, useMemo, useState } from "react";
import {
  fetchHistory, fetchProfile, getHealth, getMe, loginAccount, logoutAccount, predict, registerAccount,
  saveHistoryRemote, saveProfileRemote, type AuthUser, type GlucoseRange,
} from "./api";
import { RANGE_LABEL, RANGE_TONE, errorText, newDraft, num, type Draft } from "./flow";
import { FoodIcon } from "./foodIcons";
import Landing from "./Landing";
import { EMPTY_PROFILE, dropLegacy, peekLegacy, type HistoryRecord, type Profile } from "./storage";
import { formatWhen } from "./time";
import { Camera, Confirm, Manual } from "./steps/MealStep";
import { Result, buildPredictBody, toDetail, toRecord, type ResultState } from "./steps/ForecastStep";
import { Scan, Source, SourceCard, Summary, sourceOptions } from "./steps/GlucoseStep";
import {
  AppBar, Badge, PrimaryNav, Brand, Button, Disclaimer, ErrorBox, Field, Icon, SelectField, initials, type Screen,
} from "./ui";

// -------------------------------------------------------------------- auth

function Auth({ register, profile, go, onSubmit }: {
  register?: boolean; profile: Profile; go: (s: Screen) => void;
  /** Resolves to an error message to show, or null once signed in. */
  onSubmit: (register: boolean, name: string, email: string, password: string) => Promise<string | null>;
}) {
  const [name, setName] = useState("");
  const [email, setEmail] = useState(profile.email);
  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

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
      <form className="form-stack" onSubmit={async (e) => {
        e.preventDefault();
        if (!valid || busy) return;
        setBusy(true);
        setError("");
        const message = await onSubmit(!!register, name.trim(), email.trim(), password);
        if (message) { setError(message); setBusy(false); }
      }}>
        {register && <Field label="Nama lengkap" value={name} onChange={setName} autoComplete="name"/>}
        <Field label="Email" type="email" value={email} onChange={setEmail} autoComplete="email" inputMode="email"/>
        <Field label="Kata sandi" type="password" placeholder={register ? "Minimal 8 karakter" : "Kata sandi"} value={password} onChange={setPassword} autoComplete={register ? "new-password" : "current-password"}/>
        {register && <Field label="Konfirmasi kata sandi" type="password" placeholder="Ulangi kata sandi" value={confirm} onChange={setConfirm} autoComplete="new-password"/>}
        {(hint || error) && <p className="field-hint" role="alert">{error || hint}</p>}
        <button className="button button--primary" type="submit" disabled={!valid || busy}><span>{busy ? "Memproses…" : register ? "Daftar" : "Masuk"}</span></button>
      </form>
      <p className="auth-switch">{register ? "Sudah punya akun?" : "Belum punya akun?"} <button type="button" onClick={() => go(register ? "login" : "register")}>{register ? "Masuk" : "Daftar"}</button></p>
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
      <Disclaimer/>
      <div className="section-heading"><div><span>Terakhir</span><h2>Prediksi terbaru</h2></div>{history.length > 0 && <button onClick={() => go("history")}>Lihat riwayat</button>}</div>
      {last ? <button className="last-card last-card--button" onClick={() => onOpen(last)}>
        <div className="last-card__top"><FoodIcon size={43}/><div><h3>{last.food}</h3><p>{formatWhen(last.createdAt)}</p></div><Badge tone={RANGE_TONE[last.range]}>{RANGE_LABEL[last.range]}</Badge></div>
        <div className="last-card__stats"><div><span>Prediksi T+60</span><strong className={`tone--${last.range}`}>{Math.round(last.t60)} <small>mg/dL</small></strong></div><div><span>Sebelum makan</span><strong>{last.g0} <small>mg/dL</small></strong></div></div>
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
    <PrimaryNav active="home" go={nav}/>
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
        <div className="history-donut" role="img" aria-label={`${normalPct}% prediksi normal`} style={{ background: `conic-gradient(var(--green) 0 ${normalPct}%, var(--line) ${normalPct}%)` }}><span aria-hidden="true">{normalPct}%<small>Normal</small></span></div>
      </section>
      <div className="history-filters">{filters.map((f) => <button className={filter === f.id ? "active" : ""} onClick={() => setFilter(f.id)} key={f.id}>{f.label}</button>)}</div>
      <div className="history-heading"><div><h1>Prediksi tersimpan</h1><span>{visible.length} hasil</span></div></div>
      <div className="history-list">{visible.map((item) => <button className="history-item" key={item.id} onClick={() => onOpen(item)}>
        <FoodIcon size={48}/>
        <span className="history-item__copy"><strong>{item.food}</strong><small>{formatWhen(item.createdAt)}</small><span>Awal {item.g0} mg/dL</span></span>
        <span className="history-item__result"><Badge tone={RANGE_TONE[item.range]}>{RANGE_LABEL[item.range]}</Badge><strong>{Math.round(item.t60)}<small> mg/dL</small></strong><Icon name="chevron" size={15}/></span>
      </button>)}</div>
      {!visible.length && <div className="history-empty"><Icon name="chart" size={34}/><strong>Belum ada hasil</strong><p>{history.length ? "Tidak ada prediksi dengan status ini." : "Prediksi yang Anda buat akan muncul di sini."}</p></div>}
    </main>
    <PrimaryNav active="history" go={nav}/>
  </div>;
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
  onSave: (p: Profile) => Promise<void>; onSignOut: () => void;
}) {
  const [form, setForm] = useState(profile);
  const [saved, setSaved] = useState(false);
  const [saveError, setSaveError] = useState("");
  const [saving, setSaving] = useState(false);
  const set = (change: Partial<Profile>) => { setForm({ ...form, ...change }); setSaved(false); };
  const bmi = bmiOf(form.heightCm, form.weightKg);
  return <div className="shell with-nav"><AppBar title="Profil &amp; Pengaturan"/><main className="page profile-page">
    <div className="profile-hero"><div className="profile-avatar">{initials(form.name)}</div><div><h1>{form.name || "Pengguna"}</h1><p>{form.email || "Belum ada email"}</p><div className="profile-hero__chips"><span>{form.diabetesType}</span>{form.age && <span>{form.age} tahun</span>}</div></div></div>
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
      <p className="helper" style={{ marginTop: ".625rem" }}><Icon name="info" size={15}/> Scan Jari dipilih langsung saat memulai prediksi.</p>
    </section>
    {saveError && <p className="field-hint" role="alert">{saveError}</p>}
    <Button onClick={async () => {
      if (saving) return;
      setSaving(true);
      setSaveError("");
      try { await onSave(form); setSaved(true); } catch (e) { setSaveError(errorText(e)); }
      setSaving(false);
    }}>{saving ? "Menyimpan…" : saved ? "Tersimpan ✓" : "Simpan perubahan"}</Button>
    <Button variant="danger" onClick={onSignOut}>Keluar</Button>
  </main><PrimaryNav active="profile" go={nav}/></div>;
}

// ---------------------------------------------------------------------- app

const WIZARD: Screen[] = ["source", "scan", "summary", "camera", "confirm", "manual", "result"];

export default function App() {
  const [profile, setProfile] = useState<Profile>(EMPTY_PROFILE);
  // Every visit starts on the landing page; signed-in users go on via "Masuk".
  const [screen, setScreen] = useState<Screen>("landing");
  const [draft, setDraft] = useState<Draft>(() => newDraft(EMPTY_PROFILE));
  const [history, setHistory] = useState<HistoryRecord[]>([]);
  const [result, setResult] = useState<ResultState>({ status: "loading" });
  const [resultBack, setResultBack] = useState<Screen>("home");
  const [apiState, setApiState] = useState<"loading" | "ok" | "down">("loading");
  const [user, setUser] = useState<AuthUser | null>(null);

  // Restore the server session (HttpOnly cookie) on page load.
  useEffect(() => {
    let cancelled = false;
    getMe().then((u) => (!cancelled && u ? adoptUser(u) : undefined)).catch(() => { /* stay signed out */ });
    return () => { cancelled = true; };
  }, []);

  // Re-check every few seconds while the API is unreachable, so a server that
  // starts after the page loaded is picked up without a manual reload.
  useEffect(() => {
    if (apiState === "ok") return;
    let cancelled = false;
    const check = () => getHealth()
      .then(() => { if (!cancelled) setApiState("ok"); })
      .catch(() => { if (!cancelled) setApiState("down"); });
    void check();
    const timer = window.setInterval(check, 4000);
    return () => { cancelled = true; window.clearInterval(timer); };
  }, [apiState]);

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
      const res = await predict(buildPredictBody(draft, profile));
      const record = toRecord(res, draft);
      setHistory((h) => [record, ...h.filter((x) => x.id !== record.id)]);
      setResult({ status: "ready", record, detail: toDetail(res), fromHistory: false });
      saveHistoryRemote(record).catch(() => { /* shown on screen already; it will not be in history after a reload */ });
    } catch (e) {
      setResult({ status: "error", message: errorText(e) });
    }
  };

  const openRecord = (record: HistoryRecord) => {
    setResult({ status: "ready", record, detail: null, fromHistory: true });
    setResultBack(screen === "home" ? "home" : "history");
    go("result");
  };

  // Load the account's profile and history from the server. Anything an older
  // version of the app kept in this browser is uploaded once, then removed.
  async function adoptUser(u: AuthUser) {
    let { profile: loaded, saved } = await fetchProfile();
    let loadedHistory = await fetchHistory();
    const legacy = peekLegacy();
    const isMine = !legacy.profile?.email || legacy.profile.email.toLowerCase() === u.email.toLowerCase();
    if (isMine && (legacy.profile || legacy.history.length)) {
      if (!saved && legacy.profile) loaded = await saveProfileRemote({ ...legacy.profile, name: u.name, email: u.email, participantId: loaded.participantId });
      const known = new Set(loadedHistory.map((r) => r.id));
      await Promise.all(legacy.history.filter((r) => !known.has(r.id)).map(saveHistoryRemote));
      loadedHistory = await fetchHistory();
      dropLegacy();
    }
    setProfile(loaded);
    setHistory(loadedHistory);
    setDraft(newDraft(loaded));
    setUser(u);
  }

  const authenticate = async (register: boolean, name: string, email: string, password: string): Promise<string | null> => {
    try {
      adoptUser(register ? await registerAccount(name, email, password) : await loginAccount(email, password));
      go("home");
      return null;
    } catch (e) {
      return errorText(e);
    }
  };

  const signOut = async () => {
    try { await logoutAccount(); } catch { /* the server session expires on its own; still sign out here */ }
    setUser(null);
    setProfile(EMPTY_PROFILE);
    setHistory([]);
    setDraft(newDraft(EMPTY_PROFILE));
    go("landing");
  };

  if (screen === "landing") {
    return <Landing onLogin={() => go(user ? "home" : "login")} onRegister={() => go(user ? "home" : "register")}/>;
  }
  if (!WIZARD.includes(screen) && ["login", "register"].includes(screen)) {
    return <Auth register={screen === "register"} profile={profile} go={go} onSubmit={authenticate}/>;
  }
  // Everything past landing/login/register needs an account.
  if (!user) return <Auth profile={profile} go={go} onSubmit={authenticate}/>;

  switch (screen) {
    case "home": return <Home profile={profile} history={history} apiState={apiState} nav={nav} go={go} onOpen={openRecord}/>;
    case "history": return <History history={history} nav={nav} go={go} onOpen={openRecord}/>;
    case "profile": return <ProfileScreen profile={profile} historyCount={history.length} nav={nav} go={go} onSave={async (p) => { setProfile(await saveProfileRemote(p)); }} onSignOut={signOut}/>;
    case "source": return <Source draft={draft} patch={patch} go={go}/>;
    case "scan": return <Scan go={go} onUse={(r) => { patch({ source: "ppg", ppg: r }); go("summary"); }} onManual={() => { patch({ source: "meter" }); go("source"); }}/>;
    case "summary": return <Summary draft={draft} go={go}/>;
    case "camera": return <Camera draft={draft} patch={patch} go={go}/>;
    case "confirm": return <Confirm draft={draft} patch={patch} go={go} onPredict={runPrediction}/>;
    case "manual": return <Manual draft={draft} patch={patch} go={go} onPredict={runPrediction}/>;
    case "result": return <Result state={result} nav={nav} go={go} onRetry={runPrediction} backTo={resultBack}/>;
    default: return <Auth profile={profile} go={go} onSubmit={authenticate}/>;
  }
}
