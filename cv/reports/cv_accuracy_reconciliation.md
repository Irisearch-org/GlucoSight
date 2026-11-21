# CV accuracy reconciliation (wayfinder #11)

Question: which CV accuracy numbers are true for the food-10 MobileNetV2 and the
99-image VLM vs MobileNet comparison.

## Inputs

- Checkpoint: `cv/models/mobilenetv2_food10.pt` (main clone, gitignored), sha256
  `394c8869408025ad2cb29a3bd3e1a75d8bdc30598ec776dd75ec3db1d1bcd181`. Identical to
  `~/Downloads/cv-baseline-v0.1/models/mobilenetv2_food10.pt`.
- Dataset: Mendeley `vtjd68bmwt` v1 (`fetch_dataset.py` source), zip 110,332,093 bytes,
  extracted to `~/Downloads/glucosight-cv-data/extracted/Indonesian Food Image/Clean_Data/`
  (outside the repo). Test split: 1,193 images, per-class counts (bakso 130, bebek_betutu 97,
  gado_gado 113, gudeg 82, nasi_goreng 155, pempek 123, rawon 104, rendang 94, sate 154,
  soto 141) match the `cv/MODEL_CARD.md` support column exactly.
- Eval: `cv/scripts/eval_food10_reconcile.py`, run as
  `python cv/scripts/eval_food10_reconcile.py --checkpoint <abs ckpt> --test-root "<abs Clean_Data/test>" --out-json ... --out-predictions ...`
  (torch 2.7.0, torchvision 0.22.0, Python 3.12).
- Two inference paths on the same checkpoint and the same 1,193 files:
  - **app**: `cv_baseline.preprocessing.preprocess` + `MobileNetV2Classifier.predict_topk`
    (what `predict()` runs). Resize shorter side to 255 with PIL bilinear and `round()`,
    center crop 224, ImageNet norm.
  - **train**: torchvision `Resize(255)`, `CenterCrop(224)`, ImageNet norm (the notebook
    `eval_tf` and `train_mobilenet.py` val transform). Resize uses `int()` truncation.

## Results (full test split, n = 1,193)

| Path | Top-1 | Top-3 | Macro-F1 |
|---|---|---|---|
| app | 1051/1193 = **88.10%** | 1169/1193 = 98.00% | 0.874 |
| train | 1050/1193 = **88.01%** | 1170/1193 = 98.07% | 0.873 |

- App vs train top-1 agreement: 99.33% (7 of 1,193 images differ), from the
  preprocessing difference above.
- Per-class top-1, app path: bakso 0.892, bebek_betutu 0.866, gado_gado 0.885,
  gudeg 0.781, nasi_goreng 0.948, pempek 0.902, rawon 0.904, rendang 0.862, sate 0.851,
  soto 0.872. Train path differs on gudeg (0.756), nasi_goreng (0.955), rendang (0.872), soto (0.865).

## Why the existing numbers disagree

1. **0.8659 / 0.866 (notebook test cell, MODEL_CARD, METRICS §7.2) is not reproduced.**
   The notebook test cell uses `model.eval()`, the same transform, and the same 1,193 files.
   This checkpoint gives 1050/1193 (train path), not 1033/1193. The notebook's epoch log
   also disagrees with the metadata history (notebook best val at epoch 4 = 0.8674; metadata
   best at epoch 10 = 0.8674). [INFERENCE] The notebook's reported test number came from a
   weight state other than the one saved to `mobilenetv2_food10.pt`. I did not re-train to confirm.
2. **0.8801 (`training_metadata.json` `test_acc`) is reproduced exactly by this checkpoint
   on the train path** (1050/1193 = 0.88013). `train_mobilenet.py` does not compute test
   accuracy, so the metadata value came from some test evaluation outside the repo. It is
   the number this file actually produces, under the training transform.
3. **Spread 88.0 vs 88.1** is the app-vs-train preprocessing difference (7 images). It is
   within noise for n = 1,193.

## 99-image VLM vs MobileNet (reconstructed sample)

- Sample: `run_gas_vlm.gather_test_images` ordering, `random.seed(42)`, `random.sample(…, 100)`,
  minus `pic_338` (99 images). The VLM run's raw records
  (`cv/features/cv/vlm_v0.1_gas/features_vlm_gas*.jsonl`) and the MobileNet 99-image
  feature file are **not in the repo or on disk**, so the report's 95/99 and 70/99 cannot
  be checked against source records. [NOT RUN: VLM re-query, no API calls by design.]
- MobileNet (app path), path-keyed ground truth (correct): **89/99 = 89.9%**.
- Stem-keyed ground truth: 18 of the 99 images have a filename stem shared by more than one
  class folder (130 stems / 310 images across the full test split), so the stem-keyed
  score depends on which class wins the dict key: **76/99 .. 89/99**.
- **70/99 (70.71%) is outside every reconstruction** (path-keyed 89, stem-keyed range
  76–89). I found no code path in the repo that produces 70/99. [UNEXPLAINED]

Root cause of the stem problem: `run_gas_vlm.py` keys records by `img.stem`, and
`evaluate_real_vs_gt*.py` builds `gt_map` by stem with dict overwrite. Both are wrong when stems
collide. `simulate_from_mobile` also joins by `sample_id` only.

## What the paper may quote

- **May quote**: MobileNetV2 food-10 test top-1 **88.1%** (1051/1193, app path, n = 1,193),
  top-3 98.0%, macro-F1 0.874, from `cv/scripts/eval_food10_reconcile.py` on the checkpoint above.
  Alternative equivalent wording: 88.0% (1050/1193, training transform).
- **Must delete**: 0.8659 / 86.6% and macro-F1 0.858 (notebook, MODEL_CARD, METRICS §7.2) and
  86.6–88.0% as a range; the 0.8801 value may be kept only with "training transform" wording.
- **99-image comparison**: delete "VLM 95.96% vs MobileNet 70.71%". The MobileNet figure is
  unreproducible, and the stem-keyed scoring is invalid. The paper may cite the MobileNet
  path-keyed figure 89/99 (89.9%) only as a reconstruction, and the VLM figure only after its
  records are recovered and re-scored by path.
