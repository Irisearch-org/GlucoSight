import { useId, useLayoutEffect, useRef, useState, type ReactNode } from "react";

export type Screen =
  | "landing" | "login" | "register" | "home" | "history" | "profile"
  | "source" | "scan" | "summary" | "camera" | "confirm" | "manual" | "result";

export type IconName =
  | "arrow" | "eye" | "home" | "chart" | "user" | "drop" | "camera" | "pulse" | "meter"
  | "clock" | "info" | "chevron" | "check" | "sparkle" | "gallery" | "search" | "trash"
  | "plus" | "edit" | "shield" | "menu";

const paths: Record<IconName, ReactNode> = {
  arrow: <><path d="m15 18-6-6 6-6"/><path d="M9 12h10"/></>,
  eye: <><path d="M3 12s3.5-5 9-5 9 5 9 5-3.5 5-9 5-9-5-9-5Z"/><circle cx="12" cy="12" r="2.5"/></>,
  home: <><path d="m3 11 9-7 9 7"/><path d="M5.5 9.5V20h13V9.5"/><path d="M10 20v-6h4v6"/></>,
  chart: <><path d="M4 19V9"/><path d="M10 19V5"/><path d="M16 19v-7"/><path d="M22 19H2"/></>,
  user: <><circle cx="12" cy="8" r="4"/><path d="M4 21a8 8 0 0 1 16 0"/></>,
  drop: <path d="M12 3s6 6.7 6 11a6 6 0 1 1-12 0c0-4.3 6-11 6-11Z"/>,
  camera: <><path d="M4 8h3l1.5-2h7L17 8h3v11H4Z"/><circle cx="12" cy="13" r="3.5"/></>,
  pulse: <><path d="M3 12h4l2-5 4 10 2-5h6"/><path d="M12 21S4 16.5 4 9.5A4.5 4.5 0 0 1 12 7a4.5 4.5 0 0 1 8 2.5c0 7-8 11.5-8 11.5Z"/></>,
  meter: <><path d="M5 19a8 8 0 1 1 14 0"/><path d="m12 15 4-5"/><path d="M8 19h8"/></>,
  clock: <><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/></>,
  info: <><circle cx="12" cy="12" r="9"/><path d="M12 11v6"/><path d="M12 7.5v.5"/></>,
  chevron: <path d="m9 18 6-6-6-6"/>,
  check: <path d="m5 12 4 4L19 6"/>,
  sparkle: <><path d="m12 3 1.3 4.2L17 9l-3.7 1.8L12 15l-1.3-4.2L7 9l3.7-1.8Z"/><path d="m19 15 .7 2.3L22 18l-2.3.7L19 21l-.7-2.3L16 18l2.3-.7Z"/></>,
  gallery: <><rect x="3" y="4" width="18" height="16" rx="2"/><circle cx="9" cy="10" r="2"/><path d="m3 17 5-4 4 3 3-2 6 5"/></>,
  search: <><circle cx="11" cy="11" r="7"/><path d="m20 20-4-4"/></>,
  trash: <><path d="M5 7h14"/><path d="M9 7V4h6v3"/><path d="m7 7 1 13h8l1-13"/><path d="M10 11v5M14 11v5"/></>,
  plus: <path d="M12 5v14M5 12h14"/>,
  edit: <><path d="m4 20 4.5-1 10-10-3.5-3.5-10 10Z"/><path d="m13.5 7 3.5 3.5"/></>,
  shield: <><path d="M12 3 5 6v5c0 5 3 8.2 7 10 4-1.8 7-5 7-10V6Z"/><path d="m9 12 2 2 4-4"/></>,
  menu: <path d="M4 7h16M4 12h16M4 17h16"/>,
};

export function Icon({ name, size = 22, className = "" }: { name: IconName; size?: number; className?: string }) {
  return <svg className={className} width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">{paths[name]}</svg>;
}

/**
 * The GlucoSight mark: a blood drop (glucose) carrying a response curve that
 * ends in one point, the predicted value. No tile behind it, gradient fill.
 * `animated` draws the curve and pulses the prediction point.
 */
export function LogoMark({ size = 40, animated = false }: { size?: number; animated?: boolean }) {
  const id = useId().replace(/:/g, "");
  return <svg className={`logo-mark ${animated ? "logo-mark--animated" : ""}`} width={size} height={size} viewBox="0 0 48 48" aria-hidden="true">
    <defs>
      <linearGradient id={`${id}-fill`} x1="12" y1="4" x2="36" y2="46" gradientUnits="userSpaceOnUse">
        <stop offset="0" stopColor="#4FE0C4"/>
        <stop offset=".5" stopColor="#0FA0A0"/>
        <stop offset="1" stopColor="#0A5E70"/>
      </linearGradient>
    </defs>
    <path d="M24 3.5S8.5 20.4 8.5 30.2a15.5 15.5 0 0 0 31 0C39.5 20.4 24 3.5 24 3.5Z" fill={`url(#${id}-fill)`}/>
    <path d="M14.6 23.2c.8-3.1 2.6-6.4 4.8-9.4" fill="none" stroke="#fff" strokeOpacity=".5" strokeWidth="2.2" strokeLinecap="round"/>
    <path className="logo-mark__line" d="M13.6 31.5h4.4l2.9-6.6 4 11 3-7.4h5.6" fill="none" stroke="#fff" strokeWidth="2.6" strokeLinecap="round" strokeLinejoin="round"/>
    <circle className="logo-mark__dot" cx="34" cy="28.5" r="2.4" fill="#fff"/>
  </svg>;
}

export function Brand({ compact = false, animated = false }: { compact?: boolean; animated?: boolean }) {
  return <div className={`brand ${compact ? "brand--compact" : ""}`}>
    <LogoMark size={compact ? 36 : 48} animated={animated}/>
    <div><strong>Gluco<span>Sight</span></strong>{!compact && <small>Kenali respons tubuhmu</small>}</div>
  </div>;
}

export function Button({ children, variant = "primary", disabled, onClick, icon }: {
  children: ReactNode; variant?: "primary" | "secondary" | "text"; disabled?: boolean; onClick?: () => void; icon?: IconName;
}) {
  return <button className={`button button--${variant}`} disabled={disabled} onClick={onClick}>
    {icon && <Icon name={icon} size={19}/>}<span>{children}</span>
  </button>;
}

export type Tone = "gray" | "green" | "amber" | "red" | "teal";

export function Badge({ children, tone = "gray" }: { children: ReactNode; tone?: Tone }) {
  return <span className={`badge badge--${tone}`}>{tone === "green" && <Icon name="check" size={12}/>} {children}</span>;
}

export function AppBar({ title, onBack, action }: { title: string; onBack?: () => void; action?: ReactNode }) {
  return <header className="appbar">
    {onBack ? <button className="icon-button" onClick={onBack} aria-label="Kembali"><Icon name="arrow"/></button> : <Brand compact/>}
    <div className="appbar__title">{title}</div>
    <div className="appbar__action">{action}</div>
  </header>;
}

const steps = ["Gula Darah", "Ringkasan", "Makanan", "Hasil"];

export function Progress({ current }: { current: number }) {
  return <div className="progress-wrap">
    <div className="progress">
      {steps.map((step, index) => <div className={`progress__item ${index < current ? "is-done" : ""} ${index === current ? "is-active" : ""}`} key={step}>
        <div className="progress__dot">{index < current ? <Icon name="check" size={12}/> : index + 1}</div>
        <span>{step}</span>
      </div>)}
    </div>
  </div>;
}

export function Field({ label, placeholder, type = "text", suffix, value, onChange, min, max, step }: {
  label: string; placeholder?: string; type?: string; suffix?: string; value?: string;
  onChange?: (value: string) => void; min?: number; max?: number; step?: number;
}) {
  return <label className="field"><span>{label}</span>
    <div className="field__input">
      <input type={type} placeholder={placeholder} value={value} min={min} max={max} step={step}
        readOnly={onChange === undefined && value !== undefined} onChange={(e) => onChange?.(e.target.value)}/>
      {suffix && <b>{suffix}</b>}
    </div>
  </label>;
}

export function SelectField({ label, value, options, onChange }: { label: string; value: string; options: string[]; onChange: (value: string) => void }) {
  return <label className="field"><span>{label}</span>
    <div className="field__input field__select">
      <select value={value} onChange={(e) => onChange(e.target.value)}>{options.map((option) => <option key={option}>{option}</option>)}</select>
      <Icon name="chevron" size={16}/>
    </div>
  </label>;
}

const NAV_ORDER = ["home", "prediction", "history", "profile"] as const;
// Where the sidebar's highlight sat on the previous screen. Each screen mounts
// its own BottomNav, so this is what lets the highlight glide between menus.
let lastNavIndex = 0;

/** A bottom tab bar on phones; on desktop the same element becomes the left sidebar (see desktop.css). */
export function BottomNav({ active, go }: { active: "home" | "prediction" | "history" | "profile"; go: (s: Screen) => void }) {
  // Desktop sidebar: collapsed to an icon rail unless the user opened it; the choice is remembered.
  const [open, setOpen] = useState(() => {
    try { return localStorage.getItem("glucosight.v1.sidebar") === "open"; } catch { return false; }
  });
  const pill = useRef<HTMLSpanElement>(null);
  const index = NAV_ORDER.indexOf(active);
  useLayoutEffect(() => {
    const el = pill.current;
    if (!el) return;
    el.style.setProperty("--i", String(lastNavIndex)); // start where it was last time ...
    void el.offsetWidth;                               // ... commit that position ...
    el.style.setProperty("--i", String(index));        // ... then let the transition carry it over
    lastNavIndex = index;
  }, [index]);
  const toggle = () => {
    const next = !open;
    setOpen(next);
    try { localStorage.setItem("glucosight.v1.sidebar", next ? "open" : "closed"); } catch { /* storage blocked */ }
  };
  return <nav className={`bottom-nav ${open ? "bottom-nav--open" : ""}`} aria-label="Menu utama">
    <div className="bottom-nav__brand">
      <button className="bottom-nav__toggle" onClick={toggle} aria-expanded={open} aria-label={open ? "Tutup menu" : "Buka menu"} title={open ? "Tutup menu" : "Buka menu"}><Icon name="menu"/></button>
      {open && <Brand compact/>}
    </div>
    <span className="bottom-nav__pill" ref={pill} aria-hidden="true"/>
    <button title="Beranda" className={active === "home" ? "active" : ""} onClick={() => go("home")}><Icon name="home"/><span>Beranda</span></button>
    <button title="Prediksi" className={`bottom-nav__main ${active === "prediction" ? "active" : ""}`} onClick={() => go("source")}><span className="bottom-nav__main-icon"><Icon name="chart"/></span><span>Prediksi</span></button>
    <button title="Riwayat" className={`bottom-nav__history ${active === "history" ? "active" : ""}`} onClick={() => go("history")}><Icon name="clock"/><span>Riwayat</span></button>
    <button title="Profil" className={active === "profile" ? "active" : ""} onClick={() => go("profile")}><Icon name="user"/><span>Profil</span></button>
  </nav>;
}

export function WizardLayout({ current, title, back, children }: { current: number; title: string; back: () => void; children: ReactNode }) {
  return <div className="shell wizard"><AppBar title={title} onBack={back}/><Progress current={current}/><main className="page wizard-page">{children}</main></div>;
}

/** Compact research disclaimer with an expandable detail block, shown on Beranda and Hasil. */
export function Disclaimer({ detail }: { detail?: ReactNode }) {
  const [open, setOpen] = useState(false);
  return <div className={`disclaimer ${open ? "is-open" : ""}`}>
    <span className="disclaimer__icon"><Icon name="info" size={17}/></span>
    <div>
      <p><strong>Prototipe riset — bukan alat medis.</strong> Jangan gunakan untuk keputusan pengobatan.</p>
      {open && <p className="disclaimer__detail">{detail}</p>}
      {detail && <button className="inline-link" onClick={() => setOpen(!open)} aria-expanded={open}>
        {open ? "Sembunyikan detail" : "Lihat detail"} <Icon name="chevron" size={11}/>
      </button>}
    </div>
  </div>;
}

export function ErrorBox({ title, children }: { title: string; children?: ReactNode }) {
  return <div className="error-box" role="alert"><Icon name="info" size={19}/><div><strong>{title}</strong>{children && <p>{children}</p>}</div></div>;
}

export function initials(name: string): string {
  const parts = name.trim().split(/\s+/).filter(Boolean);
  if (!parts.length) return "G";
  return (parts[0][0] + (parts[1]?.[0] ?? "")).toUpperCase();
}
