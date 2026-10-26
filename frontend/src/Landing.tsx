import { useEffect, useRef, type PointerEvent as RPointerEvent, type ReactNode, type RefObject } from "react";
import { Brand, Icon, type IconName } from "./ui";
import "./landing.css";

const steps: { icon: IconName; title: string; text: string }[] = [
  { icon: "drop", title: "Masukkan gula darah", text: "Pilih sumber gula darah sebelum makan: hasil glukometer, pembacaan CGM, atau scan jari lewat kamera." },
  { icon: "camera", title: "Foto makanan Anda", text: "Ambil foto piring. Sistem mengenali hidangan dan memperkirakan nutrisinya. Belum dikenali? Tambahkan makanan secara manual." },
  { icon: "chart", title: "Lihat prediksi T+60", text: "Dapatkan perkiraan gula darah 60 menit setelah suapan pertama, dengan status Normal, Waspada, atau Tinggi." },
];

const features: { icon: IconName; title: string; text: string }[] = [
  { icon: "meter", title: "Tanpa wajib CGM", text: "Cukup ponsel dan hasil glukometer. Alat pemantau glukosa kontinu (CGM) tidak wajib dimiliki." },
  { icon: "gallery", title: "Untuk makanan Indonesia", text: "Dirancang untuk hidangan sehari-hari seperti nasi goreng, rendang, soto, dan gado-gado." },
  { icon: "pulse", title: "Scan jari lewat kamera", text: "Tempelkan jari ke kamera belakang untuk membaca denyut nadi dan memperkirakan gula darah." },
  { icon: "chart", title: "Riwayat prediksi", text: "Lihat kembali prediksi sebelumnya dan pola 7 hari terakhir." },
  { icon: "check", title: "Status mudah dibaca", text: "Setiap hasil ditandai Normal, Waspada, atau Tinggi dengan warna yang jelas." },
  { icon: "user", title: "Ramah dipakai", text: "Seluruh tampilan berbahasa Indonesia, dengan langkah yang singkat dan jelas." },
];

function PhoneMock() {
  return <div className="lp-phone" aria-hidden="true">
    <div className="lp-phone__notch"/>
    <div className="lp-phone__screen">
      <div className="lp-phone__bar"><span/><b>Hasil Prediksi</b><span/></div>
      <div className="lp-phone__card">
        <div className="lp-phone__row"><small>Prediksi T+60</small><i className="lp-chip">Waspada</i></div>
        <svg className="lp-phone__curve" viewBox="0 0 200 70" preserveAspectRatio="none">
          <defs>
            <linearGradient id="lp-curve-fill" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0" stopColor="#e3a737" stopOpacity=".28"/>
              <stop offset="1" stopColor="#e3a737" stopOpacity="0"/>
            </linearGradient>
          </defs>
          <path className="lp-phone__area" d="M4 58 C40 56 58 50 80 34 S124 8 142 12 S176 30 196 30 V70 H4Z" fill="url(#lp-curve-fill)"/>
          <path className="lp-phone__path" d="M4 58 C40 56 58 50 80 34 S124 8 142 12 S176 30 196 30" fill="none" stroke="#c98a2b" strokeWidth="3" strokeLinecap="round"/>
          <circle className="lp-phone__point" cx="142" cy="12" r="5" fill="#fff" stroke="#c98a2b" strokeWidth="3"/>
        </svg>
        <div className="lp-phone__range"><i/></div>
        <div className="lp-phone__labels"><small>Normal</small><small>Waspada</small><small>Tinggi</small></div>
      </div>
      <div className="lp-phone__line"><span className="lp-dot"><Icon name="drop" size={14}/></span><div><em/><em className="short"/></div></div>
      <div className="lp-phone__line"><span className="lp-dot"><Icon name="gallery" size={14}/></span><div><em/><em className="short"/></div></div>
    </div>
  </div>;
}

/** Adds `is-visible` to every `.lp-reveal` inside `root` once it scrolls into view. */
function useReveal(root: RefObject<HTMLDivElement | null>) {
  useEffect(() => {
    const items = root.current?.querySelectorAll<HTMLElement>(".lp-reveal") ?? [];
    if (!("IntersectionObserver" in window)) {
      items.forEach((el) => el.classList.add("is-visible"));
      return;
    }
    const io = new IntersectionObserver((entries) => {
      entries.forEach((e) => {
        if (e.isIntersecting) {
          e.target.classList.add("is-visible");
          io.unobserve(e.target);
        }
      });
    // threshold 0: the hidden state is clipped, so a ratio-based threshold would never be reached.
    }, { threshold: 0, rootMargin: "0px 0px -8% 0px" });
    items.forEach((el) => io.observe(el));
    // Safety net: anything already on screen after a moment is shown, whatever the observer did.
    const safety = window.setTimeout(() => {
      items.forEach((el) => {
        const r = el.getBoundingClientRect();
        if (r.top < window.innerHeight && r.bottom > 0) el.classList.add("is-visible");
      });
    }, 2500);
    return () => { io.disconnect(); window.clearTimeout(safety); };
  }, [root]);
}

/** Headline split into words that rise out from behind a mask, one after another. */
function Words({ text, start = 0, step = 0.07 }: { text: string; start?: number; step?: number }) {
  return <>{text.split(" ").map((w, i) => <span key={i}><span className="lp-w"><span style={{ animationDelay: `${start + i * step}s` }}>{w}</span></span>{" "}</span>)}</>;
}

/** A button that leans toward the pointer. */
function Magnetic({ className, onClick, children }: { className: string; onClick: () => void; children: ReactNode }) {
  const ref = useRef<HTMLButtonElement>(null);
  const move = (e: RPointerEvent) => {
    const el = ref.current;
    if (!el || e.pointerType === "touch") return;
    const r = el.getBoundingClientRect();
    el.style.transform = `translate(${(e.clientX - (r.left + r.width / 2)) * 0.22}px, ${(e.clientY - (r.top + r.height / 2)) * 0.32}px)`;
  };
  return <button ref={ref} className={`${className} lp-btn--mag`} onClick={onClick} onPointerMove={move} onPointerLeave={() => { if (ref.current) ref.current.style.transform = ""; }}>{children}</button>;
}

/** Feeds the pointer position into `--mx/--my` of the `.lp-spot` card it is over, for the spotlight glow. */
function useSpotlight(root: RefObject<HTMLDivElement | null>) {
  useEffect(() => {
    const el = root.current;
    if (!el) return;
    const move = (e: PointerEvent) => {
      const card = (e.target as HTMLElement).closest<HTMLElement>(".lp-spot");
      if (!card) return;
      const r = card.getBoundingClientRect();
      card.style.setProperty("--mx", `${e.clientX - r.left}px`);
      card.style.setProperty("--my", `${e.clientY - r.top}px`);
    };
    el.addEventListener("pointermove", move);
    return () => el.removeEventListener("pointermove", move);
  }, [root]);
}

/** onLogin opens the app directly when this browser is already signed in. */
export default function Landing({ onLogin, onRegister }: { onLogin: () => void; onRegister: () => void }) {
  const root = useRef<HTMLDivElement>(null);
  useReveal(root);
  useSpotlight(root);

  // The phone tilts toward the pointer, with a sheen that follows it.
  const tilt = (e: RPointerEvent<HTMLDivElement>) => {
    const phone = e.currentTarget.querySelector<HTMLElement>(".lp-phone");
    if (!phone || e.pointerType === "touch") return;
    const r = e.currentTarget.getBoundingClientRect();
    const px = (e.clientX - r.left) / r.width - 0.5;
    const py = (e.clientY - r.top) / r.height - 0.5;
    phone.style.setProperty("--ry", `${px * 18}deg`);
    phone.style.setProperty("--rx", `${-py * 12}deg`);
    phone.style.setProperty("--gx", `${(px + 0.5) * 100}%`);
    phone.style.setProperty("--gy", `${(py + 0.5) * 100}%`);
    phone.style.setProperty("--glare", "1");
  };
  const untilt = (e: RPointerEvent<HTMLDivElement>) => {
    const phone = e.currentTarget.querySelector<HTMLElement>(".lp-phone");
    ["--rx", "--ry", "--glare"].forEach((v) => phone?.style.removeProperty(v));
  };

  return <div className="lp" id="top" ref={root}>
    <header className="lp-header">
      <div className="lp-wrap lp-header__in">
        <Brand compact animated/>
        <nav className="lp-nav" aria-label="Navigasi utama">
          <a href="#cara-kerja">Cara kerja</a><a href="#fitur">Fitur</a>
        </nav>
        <div className="lp-header__cta">
          <button className="lp-btn lp-btn--ghost" onClick={onLogin}>Masuk</button>
          <button className="lp-btn lp-btn--solid" onClick={onRegister}>Daftar</button>
        </div>
      </div>
    </header>

    <main>
      <section className="lp-hero">
        <div className="lp-wrap lp-hero__in">
          <div className="lp-hero__copy">
            <span className="lp-pill lp-enter" style={{ animationDelay: ".05s" }}><Icon name="sparkle" size={14}/> Untuk penderita diabetes tipe 2</span>
            <h1>
              <Words text="Perkirakan gula darah setelah makan," start={0.15}/>
              <em className="lp-em"><Words text="cukup dengan ponsel." start={0.6}/>
                <svg className="lp-squig" viewBox="0 0 240 14" preserveAspectRatio="none" aria-hidden="true"><path d="M3 9c10-9 20-9 30 0s20 9 30 0 20-9 30 0 20 9 30 0 20-9 30 0 20 9 30 0 20-9 30 0 20 9 24 0" pathLength="1" fill="none" stroke="currentColor" strokeWidth="3.4" strokeLinecap="round"/></svg>
              </em>
            </h1>
            <p className="lp-enter" style={{ animationDelay: ".95s" }}>GlucoSight menggabungkan gula darah sebelum makan, foto makanan, dan scan jari untuk memperkirakan gula darah Anda 60 menit setelah makan.</p>
            <div className="lp-hero__actions lp-enter" style={{ animationDelay: "1.1s" }}>
              <button className="lp-btn lp-btn--solid lp-btn--lg" onClick={onRegister}>Mulai prediksi <Icon name="chevron" size={18}/></button>
            </div>
            <ul className="lp-hero__facts lp-enter" style={{ animationDelay: "1.25s" }}>
              <li><Icon name="check" size={16}/> Bahasa Indonesia</li>
              <li><Icon name="check" size={16}/> Tanpa wajib CGM</li>
              <li><Icon name="check" size={16}/> Foto makanan Indonesia</li>
            </ul>
          </div>
          <div className="lp-hero__art lp-enter" style={{ animationDelay: ".4s" }} onPointerMove={tilt} onPointerLeave={untilt}><PhoneMock/></div>
        </div>
      </section>

      <section className="lp-section" id="cara-kerja">
        <div className="lp-wrap">
          <div className="lp-head lp-reveal"><span>Cara kerja</span><h2>Tiga langkah, tanpa ribet</h2><p>Dari gula darah sebelum makan sampai perkiraan 60 menit setelahnya.</p></div>
          <ol className="lp-steps">
            {steps.map((s, i) => <li className="lp-step lp-spot lp-reveal" style={{ transitionDelay: `${i * 0.14}s` }} key={s.title}>
              <span className="lp-step__num">{i + 1}</span>
              <span className="lp-step__icon lp-ic"><Icon name={s.icon} size={26}/></span>
              <h3>{s.title}</h3><p>{s.text}</p>
            </li>)}
          </ol>
        </div>
      </section>

      <section className="lp-section lp-section--tint" id="fitur">
        <div className="lp-wrap">
          <div className="lp-head lp-reveal"><span>Fitur</span><h2>Dibuat untuk kebiasaan makan Anda</h2><p>Sederhana dipakai, dirancang untuk kebutuhan sehari-hari.</p></div>
          <div className="lp-features">
            {features.map((f, i) => <article className="lp-feature lp-spot lp-reveal" style={{ transitionDelay: `${(i % 3) * 0.12}s` }} key={f.title}>
              <span className="lp-feature__icon lp-ic"><Icon name={f.icon} size={22}/></span>
              <h3>{f.title}</h3>
              <p>{f.text}</p>
            </article>)}
          </div>
        </div>
      </section>

      <section className="lp-cta">
        <div className="lp-wrap">
          <div className="lp-cta__in lp-reveal">
            <span className="lp-ripples" aria-hidden="true"><i/><i/><i/></span>
            <h2>Coba prediksi pertama Anda</h2>
            <p>Buat akun dan lihat perkiraan gula darah setelah makan berikutnya.</p>
            <button className="lp-btn lp-btn--light lp-btn--lg" onClick={onRegister}>Daftar dan mulai <Icon name="chevron" size={18}/></button>
          </div>
        </div>
      </section>
    </main>

    <footer className="lp-footer">
      <div className="lp-wrap lp-footer__in">
        <span>© 2026 GlucoSight</span>
        <a href="#top" className="lp-footer__top">Kembali ke atas <Icon name="chevron" size={14}/></a>
      </div>
    </footer>
  </div>;
}
