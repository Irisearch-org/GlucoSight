# The PPG option serves a real model's glucose estimate, shown with its error

**Status:** accepted (2026-11-21). Amends ADR 0003 for the demo's starting-glucose path only.

The user flow is fixed: step 1 gets the current glucose from a glucometer entry, a CGM reading, or the finger PPG scan, and the forecaster needs that value. ADR 0003's answer (the scan gives no glucose, g0 comes from a direct measurement) would make PPG users type a meter value anyway, which changes the flow. Every stage must also serve a real ML model, never a mean. So the PPG option serves the best trained or public PPG→glucose model found by the PPG trial ticket (PaPaGei-S embeddings if they pass the ADR 0004 gate, otherwise ridge on the existing features, MAE 15.64 vs the mean's 14.54 on 23 subjects). Its output becomes g0, ring-fence E1.2 is lifted for this path, and the estimate is always shown with its measured error range.

ADR 0003 rejected shipping a *mean-equivalent* estimate labelled as PPG information. This differs in two ways: the served model's output varies with the signal, and its error is shown to the user and stated in the paper. The paper says plainly that on this dataset it does not beat the mean, and it reports forecast error separately for PPG-started and meter-started forecasts, so the cost of the scan's error is visible.

## Consequences

- ADR 0003's research conclusion stands. No claim that PPG estimates glucose well is made anywhere.
- The CGM option stays a single latest reading; no CGM-history input and no pretrained-forecaster comparison for CGM users.
- **Measured after acceptance (PPG trial, wayfinder #16 and #17):** the served model is nested ridge on 7 handcrafted features, MAE 14.69 vs the mean's 14.55. Its out-of-fold predictions vary (SD 4.2 mg/dL) but barely track glucose: pred ≈ 113.4 + 0.010·true, correlation 0.045. In practice the PPG g0 is close to mean-equivalent, the case ADR 0003 warned about. The safeguard is honesty, not accuracy: the scan always shows the ±25 mg/dL range and a "rough estimate" label, the forecaster knows `g0_source = ppg`, and the paper states the correlation.
- **Checked on phone-camera data (wayfinder #27, #28; BUT PPG v2, CC BY 4.0, 18 finger subjects):** no model beats the mean (served features + ridge MAE 22.15 vs 22.29, r = −0.18; PaPaGei-S 22.25), and calibrating on a glucometer reading is worse than assuming glucose is unchanged (20.9 vs 19.4 MAE). The served model is not retrained, and no calibration feature is built.
