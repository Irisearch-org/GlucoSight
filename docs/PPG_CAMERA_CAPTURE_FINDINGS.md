# Contact PPG from a phone browser — capture findings

**Date:** 2026-10-18
**Device:** iPhone (iOS / WebKit), Chrome-on-iOS-equivalent — every iOS browser is
WebKit, so vendor is irrelevant here
**Transport:** HTTPS via a temporary tunnel; `http://<LAN-IP>` cannot work because
`getUserMedia` requires a secure context
**Page:** `integration/web/ppg-capture.html`, served at `GET /ppg`
**Question:** can the contracted contact-PPG capture be produced from a browser,
and are the numbers the pipeline returns from it trustworthy?
**Status:** Yes on the capture, with a channel fix. No on three of the four
returned PPG fields.

> **Scope.** This is about acquisition and about the fields the pipeline returns.
> It does not run the glucose model, and no result here is a claim about
> predictive accuracy. Raw captures were taken on a phone and are not retained,
> so findings marked *(observation)* have evidence recorded here but no
> in-repo reproduction.

---

## Why this exists

`INTERFACE_CONTRACT_v1.md` specifies the capture protocol as *"finger against the
lens with the flash on"*, plus camera lock (C7), and requires the producer to
report `ppg_signal_quality`, `ppg_method`, `ppg_clip_fraction` and
`ppg_n_windows`. `integration/README.md` records that the demo page could only
**simulate** a capture, and that a scalar PPG signal *"cannot measure per-channel
clipping or validate phone-camera acquisition."*

So the track's stated gap was: nobody had ever produced a real one. This closes
that, and records what the real one turned out to be — which is not what the
simulated one looked like.

The failures below were found in sequence, each one exposed by the diagnostic
added for the previous one. The order matters: every individual number looked
plausible at the time it was read.

---

## Findings

### P1 — A finger over the torch puts green at the quantisation floor; red carries the signal

*(reproducible — `integration/web/tests/ppg_signal_check.js`)*

With a fingertip clamped over the lens and the torch lit, the measured **green
DC was 0.0021** of full scale — about **half a count out of 255**. That is not a
dark frame; it is a frame with almost no *green* light, which is what
haemoglobin does. The picture is red.

The extractor's perfusion index is a ratio (`rppg/features/extractor.py:84`):

```python
ac_pp = ac_signal.max() - ac_signal.min()
return float(ac_pp / dc * 100.0)
```

With a DC at the floor, an AC of roughly a third of a count produces a quotient
of **0.53** — a "perfusion index" of 53%, where living tissue manages 1–3%.
Dividing one near-zero number by another is the whole mechanism.

The demo page was **hardcoded to green**. Same synthetic finger, only the channel
differs:

| channel | DC | AC/DC | verdict |
|---|---|---|---|
| green (previously hardcoded) | 0.0021 | 0.0404 | frame nearly black |
| red (auto-selected) | 0.5501 | **0.0240** | physiological |

**Fix.** The page's channel selector now defaults to `auto`, taking whichever of
R/G/B is brightest — the one with signal in it. It also reports the DC of each
channel, because the ratio is only comparable to a physiological band when the
denominator is sane.

**Confirmed by outcome, not just by argument:** after the fix the live capture
returned `perfusion_index` **0.0284** (P7). A real 53% pulse would not have
moved to a textbook value because the channel changed.

### P2 — The pipeline does not invent pulsation: the negative control

*(reproducible — weight arithmetic in `rppg/check_ppg_findings.py`)*

A folded sheet of paper over the lens, torch on, 34 s, same page:

| | perfusion index | signal quality | weight | reported bpm |
|---|---|---|---|---|
| paper | **0.000345** | 0.1309 | **0.493** | 52.5 |
| finger (P1 capture) | 0.5319 | 0.6646 | 1.0 | 60.0 |

A factor of **1540** in amplitude between paper and finger. The extractor has a
real noise floor and paper sits on it — so the large finger amplitudes were a
response to something physical, not an artefact of the estimator.

The quality gate also worked unprompted. `signal_quality` 0.1309 is below
`CONFIDENCE_FLOOR = 0.3` (`integration/contract.py:41`), so ppg was down-weighted
to **0.1 + 0.9 × (0.1309/0.3) = 0.493** (`contract.py:323`, H1
down-weight-never-exclude). Never dropped.

**But note what was *not* refused.** For a piece of paper it still returned
`pulse_rate_bpm: 52.5` and `hrv_rmssd: 127.53`. Between them, those two fields
are the reason five consecutive captures all looked plausible.

### P3 — `pulse_rate_bpm` is quantised to the FFT grid, and is returned even with no pulse

*(reproducible — `rppg/check_ppg_findings.py`)*

`estimate_pulse_rate_bpm` (`rppg/features/extractor.py:35`) takes the argmax of a
Welch PSD on a fixed frequency grid:

```python
freqs, psd = welch(s, fs=fs, nperseg=len(s), nfft=len(s) * 4, detrend='linear')
f0 = freqs[mask][np.argmax(psd[mask])]
return float(f0 * 60.0)
```

At the observed 30.3 Hz over a 10-second window that is 303 samples, `nfft` 1212,
so the bin spacing is `30.3 / 1212 = 0.025 Hz` — **1.50 bpm**. The reported rate
can only land on multiples of 1.5:

| true rate | reported |
|---|---|
| 58.0 | 58.5 |
| 59.0 | 58.5 |
| **60.0** | **60.000** |
| 61.0 | 61.5 |
| 62.0 | 61.5 |

A live capture returned `pulse_rate_bpm: 60.000000000000014`. That is not
precision — it is the 60.0 bin, and the trailing digits are float noise. True
rate could be anywhere in roughly 59.3–60.8.

There is no sub-bin interpolation, and **no confidence gate**: an argmax always
returns a bin, so a rate is reported for cardboard.

### P4 — `ppg_hrv_rmssd` is not measuring heart-rate variability

*(observation — five captures, raw signals not retained)*

Across every capture in this exercise:

| capture | signal quality | perfusion index | **hrv_rmssd** |
|---|---|---|---|
| green, finger | 0.640 | 0.2901 | **165.30** |
| green, finger | — | 0.4148 | **91.50** |
| green, finger, pressed | 0.665 | 0.5319 | **99.71** |
| **paper — no circulation at all** | 0.131 | 0.000345 | **127.53** |
| red, finger (P7) | 0.765 | 0.0284 | **112.30** |

Healthy resting RMSSD is 20–60 ms. At 55.5 bpm the beat interval is ~1081 ms, so
112 ms means successive intervals differ by over 10% of the interval; 3–6% is
typical. The column never once entered the human range — **not even on paper**,
at quality 0.131.

`compute_hrv_rmssd` (`extractor.py:71`) filters intervals to median × [0.7, 1.4]
before squaring differences, so it is not naive — but the filter cannot rescue
beat positions from a detector that is firing irregularly. The field is tracking
the beat detector's own jitter.

The contract declares it optional with default 30.0 (`contract.py:140`), so a
junk value does not break anything. It does, however, mean `ppg_hrv_rmssd` should
not be quoted.

**Corollary.** The rate and the HRV come out of the *same* detector —
`extract_features` calls `estimate_pulse_rate_bpm`, then `detect_beats`, then
`compute_hrv_rmssd` (`extractor.py:154–158`). A capture whose HRV is demonstrably
fictitious gives no independent reason to trust its rate. In the one capture
where the rate was checked against a manual count it did match (P7), but that is
one check, not a property of the pipeline.

### P5 — iOS cannot honour the contract's camera lock

*(observation — `getCapabilities()` as reported by the device)*

```
aspectRatio, backgroundBlur, deviceId, facingMode, focusDistance, frameRate,
groupId, height, powerEfficient, torch, whiteBalanceMode, width, zoom
```

`exposureMode` is **absent**, and `applyConstraints({advanced:[{exposureMode:
'manual'}]})` is refused. This is not a browser choice: **every browser on iOS is
WebKit**, so Safari, Chrome and Firefox on iOS expose the identical video
constraints. Chrome on Android is currently the only common browser that hands
over exposure control.

`torch` **is** offered and, on this device, the LED was confirmed lit by eye. The
page now verifies it by measurement rather than by assuming the constraint took —
it samples frame brightness with the torch on and off, and reports `refused` if
the picture does not actually get brighter, because `applyConstraints` can return
without error and do nothing. (An earlier revision recorded the torch as
`accepted` on the strength of "no exception thrown", which was not evidence.)

**Consequence for C7.** The contracted camera lock is unachievable on iOS. It
does not by itself destroy a capture — a still finger under a steady torch is
close to a static scene — but it is the reason the page reports and measures
exposure drift rather than asserting the protocol was met.

### P6 — `ppg_perfusion_index` units disagree between the extractor and the contract

*(reproducible — `rppg/check_ppg_findings.py`)*

| where | units |
|---|---|
| `extractor.compute_perfusion_index` | **percent** (`ac_pp / dc * 100`) |
| `integration/api.py:239` | divides by 100 → **ratio** |
| `integration/contract.py:142` | `Field(default=2.0, lo=0.0, hi=20.0)` — **percent** bounds |
| `integration/tests/test_api.py:93` | `assert 0.0 <= perfusion_index <= 20.0` |

The validator and its test still use the percent range while the service emits a
ratio, so the check it appears to perform is not the check it performs: it
permits a ratio of 20, i.e. an AC/DC of 2000%. **The impossible 0.5319 from P1
passed validation**, and so would anything else. The one field that could have
flagged a capture built from two near-zero numbers is the field whose range is
declared in the wrong units.

`schema.py` in the CV track is already scheduled for replacement by the shared
validator; this field is a concrete reason to do it.

### P7 — A successful capture, verified against a manual count

*(observation)*

After the P1 channel fix, using the auto-selected channel:

| field | value |
|---|---|
| `capture_duration_seconds` | 33.96 |
| `observed_sampling_rate_hz` | 30.30 |
| `timestamp_jitter_fraction` | 8.6e-14 |
| `n_valid_windows` | 3 |
| `ppg_signal_quality` | **0.765** |
| `ppg_perfusion_index` | **0.0284** (physiological, band 0.01–0.03) |
| `ppg_pulse_rate_bpm` | **55.5** |
| `ppg_hrv_rmssd` | 112.30 (see P4) |
| `ppg_present` | 1, weight 1.0 |

**The rate matched a 30-second manual pulse count.** That is the only figure in
this exercise that does not come out of the same code that produced it.

**Caveats.** One capture is not a method. 0.0284 sits at the *top* of the
physiological band, and with no exposure lock (P5) nothing here establishes that
the reading is **stable** across captures, fingers or time. The stability
question is the same one the integration audit already flags as unrun for the
`g0` decay curve.

---

## What this changes, and what it does not

**It does not change the forecast.** With the successful capture in hand,
`glucose_t60` came back **149.8** — identical to a request with no PPG at all.
`forecast_input_usage.not_used_for_glucose_value` lists `ppg`, and
`glucose_source.rejected_candidates` states it outright:

> "a PPG capture is available but the PPG g0 path is disabled (ring-fence E1.2 —
> see `integration/glucose_source.py`)"

The best version of this signal contributes **zero** to the glucose number. That
is E1 working as designed, not a bug — but it means a working capture is a
diagnostic result and not a modelling one.

**It does change three things.**

1. **The capture path is real.** A browser in a WebKit-only environment that
   refuses the exposure lock can still produce a contract-accepted
   `ppg_capture` with a correct rate. Requirement: select the channel that has
   light in it (P1).
2. **Three of the four returned PPG fields are not trustworthy as-is** — the rate
   is quantised and ungated (P3), the HRV never enters the human range (P4), and
   the perfusion index has no working range check (P6). Only `signal_quality` and
   the amplitude behaved correctly throughout.
3. **The negative control exists.** Paper → 0.000345 with quality 0.131 and
   correct down-weighting (P2). That is the discipline the integration audit
   found missing elsewhere — the constant predictor at Zone A 83.6%, and 88.1%
   for a model trained on pure noise.

---

## Relationship to review findings

| finding | relationship |
|---|---|
| **E1 / E1.2** | Ring-fence holds end to end; a good capture moves the forecast by zero (P7). |
| **H1** | Down-weight-never-exclude verified on live data: paper at quality 0.131 → weight 0.493 (P2). |
| **C7** | Camera lock is unachievable on iOS; the page measures drift instead of asserting it (P5). C7's own fix already names red as the channel to watch — *"keep red channel off saturation"* — which anticipates P1: red carries the signal, and the page flags clipping on it. |
| **F11** | Same class of defect as P6: three schemas and no validator that tests what it appears to (`ppg_perfusion_index` is declared in percent and emitted as a ratio). |

---

## Reproduce

```bash
# extractor- and contract-side findings (P2 arithmetic, P3, P6)
python -m rppg.check_ppg_findings

# page-side findings (P1 channel selection and DC guard, P2 amplitude)
node integration/web/tests/ppg_signal_check.js

# the capture itself
python -m uvicorn integration.api:app --host 0.0.0.0 --port 8010
#   secure context is required: localhost, or https. A LAN IP over http will
#   leave navigator.mediaDevices undefined.
#   then open /ppg, Start camera, cover the lens, Record 34 s.
```

The page is generated from `integration/web/ppg-capture.html`; `GET /ppg` is
registered in `integration/api.py`.

## What would change this answer

1. **A repeat capture set** — 3+ fingers, 3+ repetitions each, checking that
   `perfusion_index` stays in band and the rate agrees with a manual count every
   time. Until then P7 is one measurement.
2. **The same page on Chrome/Android**, where `exposureMode` exists. That would
   separate "browser capture works" from "browser capture works *without* an
   exposure lock", which is the question P5 raises and this data cannot answer.
3. **A fixed `hrv_rmssd`** — either a sub-bin beat-position estimator or honest
   `null` — with the field re-checked against a known-rate source (P4).
4. **The P6 range fix**, so an impossible perfusion index fails loudly instead of
   passing a check written in the wrong units.
