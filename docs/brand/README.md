# GlucoSight — Logo

Derived from `CLAUDE.md`. Not a signed-off brand system; it is a working
identity for slides, the repo, and the prototype UI.

## The mark

An open lens containing one unbroken stroke.

- **The ring** is the phone camera — the only sensor this project assumes.
  It is open at 3 o'clock because the system does not close the loop on a
  measurement; it hands off a forecast.
- **The stroke** starts as a contact-PPG pulse (sharp systolic spike, red)
  and resolves into a postprandial glucose response (broad swell, teal),
  then leaves the frame through the opening. Measurement and prediction are
  drawn as the same continuous line — that is the whole thesis of the
  project in one gesture: T+60 and T+120 are inferred *from* the pulse, not
  measured separately.
- **What it deliberately is not:** no blood drop, no lancet, no glucometer.
  The point of GlucoSight is that none of those are in the loop.

## Files

| File | Use |
|---|---|
| `glucosight-mark.svg` | Primary mark, full colour. App icon, favicon, avatar |
| `glucosight-mark-mono.svg` | One colour via `currentColor`. Papers, stamps, watermarks, any single-ink print |
| `glucosight-lockup.svg` | Mark + wordmark, horizontal. Slide titles, README header, poster |
| `preview.html` | Open in a browser to see every variant, size, and swatch |

## Palette

| Role | Hex | Where |
|---|---|---|
| Ink | `#0B1F3A` | Lens ring, wordmark, body text |
| Pulse | `#FF4D5E` | Start of the signal stroke — the optical/PPG end |
| Transit | `#FF6B4A` | Gradient midpoint |
| Forecast | `#00C2A8` | End of the signal stroke — the predicted end |

On dark grounds the ring goes `#FFFFFF` and the gradient lightens to
`#FF6B7A → #FF8A63 → #2AE0C6` to hold contrast. `preview.html` shows both.

## Usage

- **Clear space:** one quarter of the mark's height on all sides. The tail
  overshoots the ring on the right; do not crop it — the overshoot is the
  point.
- **Minimum size:** 22 px for the mark, 120 px wide for the lockup. Below
  22 px the second peak closes up; use the mono mark there.
- **Stroke weights** (`7` ring, `6.5` signal at a 96-unit viewBox) are tuned
  so the trough between the two peaks stays open. Do not rescale one path
  without the other.
- **Wordmark:** currently live `<text>` in a system sans stack. Outline it to
  paths before any external publication (paper submission, poster, press)
  so it cannot reflow on a machine without the font.

## Exporting raster

```bash
magick -background none docs/brand/glucosight-mark.svg -resize 512x512 mark-512.png
```
