# Mobile UI manual device checklist

Closing gate for the mobile-first UI refinement (spec [#32](https://github.com/Irisearch-org/GlucoSight/issues/32)). The automated browser suite (`npm run test:ui`, Chromium and WebKit) covers layout only; it cannot light a torch, use a real camera, or show a real on-screen keyboard. Run this on **one iPhone (Safari)** and **one Android phone (Chrome)**, plus an iPad or large tablet for the last section.

## Setup

1. Start the backend: `python -m uvicorn integration.api:app --reload` (see `integration/README.md`).
2. Start the frontend over LAN HTTPS (phone cameras need a secure context): `cd frontend && npm run dev:phone`.
3. Open the printed `https://<LAN-IP>:5173` URL on the phone and accept the certificate warning.
4. Sign in with a test account that has a profile and at least one saved prediction.

Mark each item pass or fail, with the device and browser version.

## Primary nav and screen edges

- [ ] Four tabs (Beranda, Prediksi, Riwayat, Profil) are reachable with one thumb; the bar sits above the home indicator (iPhone) or gesture bar (Android), not under it.
- [ ] The app bar clears the status bar and notch on every screen.
- [ ] Browser toolbar showing and hiding while scrolling does not make any screen jump or leave a gap at the bottom.
- [ ] iPhone SE-size screen (or Chrome device mode at 320 px width): nothing is cut off on Home, History, Profile.
- [ ] Rotating to landscape and back does not break the layout.

## Keyboard against the pinned action row

- [ ] Source: focus the glucometer field. The field and the pinned "Lanjut" button stay visible above the keyboard (iPhone and Android).
- [ ] Manual entry: type in the dish search. The suggestion list is fully visible and scrollable above the keyboard (or flips above the field), and the pinned "Lihat prediksi" button is not covered.
- [ ] Sign-in and registration: with the keyboard open, the submit button can be reached.
- [ ] The page does not zoom when any field is focused.

## Finger scan (Android Chrome)

- [ ] Source shows the scan option with the "butuh Chrome di Android" note.
- [ ] Opening the camera lights the torch; the finger guide, countdown, signal badge and wave are readable at full brightness.
- [ ] A full scan ends with the rough estimate, its range, "Estimasi kasar dari kamera", the glucometer note and the heart rate.
- [ ] Denying camera permission shows "Masukkan hasil glukometer/CGM" as the primary button; "Ulangi scan" is secondary.

## Finger scan (iPhone Safari)

- [ ] Opening the camera fails the torch check and shows the error state with "Masukkan hasil glukometer/CGM" as the primary button.
- [ ] Tapping it lands on Source with Glukometer selected.

## Meal photo

- [ ] The live viewfinder is portrait and shows exactly the frame that is sent (take a photo of a plate with an object at the edge and check it is in the result).
- [ ] Take a portrait photo on the phone and confirm the on-screen viewfinder shows the whole frame (no hidden strip).
- [ ] The shutter, Galeri and manual entry are reachable; note whether scrolling to the shutter is acceptable at your screen size.
- [ ] Choose a large gallery photo (a recent photo straight from the camera roll, including a HEIC one on iPhone): recognition starts quickly and succeeds.
- [ ] Confirm: items the matcher was unsure about can be resolved with the suggestion chips; "Lihat prediksi" stays disabled until every item is resolved.

## Result and honesty elements

- [ ] A result from a meter or CGM reading shows both horizons with their ± range, no extra warning.
- [ ] A result started from a finger scan shows the "Prediksi dimulai dari estimasi kasar" card above the gauges.
- [ ] A Tipe 1 or Lainnya profile shows the outside-training warning.
- [ ] A result opened from Riwayat shows "Rentang galat hanya tersedia pada prediksi baru." and no ± value.
- [ ] The research disclaimer is visible on Home and Result.
- [ ] No confidence percentage, HRV, perfusion index or GI appears anywhere.

## Motion and touch feel

- [ ] Buttons show a pressed state; there is no grey tap flash and no text selection on long press.
- [ ] Step changes in the prediction flow are quick and do not block taps.
- [ ] With "Reduce Motion" on (iOS Accessibility) or "Remove animations" (Android), no screen animates.

## Tablet and desktop

- [ ] iPad portrait (768 to 820 px): intentional centred column, History in two columns, Primary nav unchanged.
- [ ] Desktop browser at 1280 px and wider: Primary nav is open on a first visit, items aligned, highlight on the active item; collapse and reopen works and is remembered.
