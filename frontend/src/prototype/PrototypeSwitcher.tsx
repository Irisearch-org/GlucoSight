/// <reference types="vite/client" />
// PROTOTYPE — throwaway (wayfinder #19). Floating variant switcher; dev builds only.
import { useEffect } from "react";

export function useSearchParam(key: string, fallback: string): [string, (v: string) => void] {
  const params = new URLSearchParams(window.location.search);
  const value = params.get(key) ?? fallback;
  const set = (v: string) => {
    const p = new URLSearchParams(window.location.search);
    p.set(key, v);
    window.history.replaceState(null, "", `?${p.toString()}`);
    window.dispatchEvent(new Event("prototype-nav"));
  };
  return [value, set];
}

export function PrototypeSwitcher({ variants, current, onChange }: {
  variants: { key: string; name: string }[]; current: string; onChange: (key: string) => void;
}) {
  const i = Math.max(0, variants.findIndex((v) => v.key === current));
  const step = (d: number) => onChange(variants[(i + d + variants.length) % variants.length].key);
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement;
      if (t.closest("input, textarea, [contenteditable]")) return;
      if (e.key === "ArrowLeft") step(-1);
      if (e.key === "ArrowRight") step(1);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });
  if (!import.meta.env.DEV) return null;
  const pill: React.CSSProperties = {
    position: "fixed", bottom: 18, left: "50%", transform: "translateX(-50%)", zIndex: 1000,
    display: "flex", alignItems: "center", gap: 10, padding: "8px 12px", borderRadius: 999,
    background: "#111", color: "#fff", fontSize: 13, boxShadow: "0 6px 20px rgba(0,0,0,.3)",
  };
  const btn: React.CSSProperties = { background: "#333", color: "#fff", border: "none", borderRadius: 999, width: 28, height: 28, cursor: "pointer" };
  return <div style={pill}>
    <button style={btn} onClick={() => step(-1)} aria-label="Previous variant">←</button>
    <strong>{variants[i].key}</strong><span>({variants[i].name})</span>
    <button style={btn} onClick={() => step(1)} aria-label="Next variant">→</button>
  </div>;
}
