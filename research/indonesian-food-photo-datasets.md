# Indonesian food photo datasets for held-out dish identification

Ticket: #10 (parent map #5). Scope: public, labelled Indonesian/SEA food photo datasets that can evaluate dish identification on held-out images, with mixed plates (nasi campur, padang) as a priority. Output: size, classes, TKPI dish-name overlap, licence, download path, and a recommendation.

## Bottom line

- No public dataset found that is both large and held-out at the person level, and also labels mixed plates. Every candidate is single-dish (or detection-boxed), web-scraped, or small.
- Most credible for a paper: **Wibisono et al. 2020 / In-TFK 34-35 class studio set** for controlled dish identity (clean capture, CC BY 4.0 article, publicly downloadable). Weakness: no mixed plates, not street food.
- Largest labelled set with explicit nasi campur and nasi padang: **Rasyidi et al. 2024 (24,427 images, 160 classes)**. Blocker: no public download found; its data availability text was not located in the article.
- The Mendeley 10-class set (`vtjd68bmwt/1`) that trains the current MobileNetV2 is **not** held-out for the paper. It is the training data (see `cv/MODEL_CARD.md`).
- Mixed-plate evaluation needs either a detection-labelled set (Roboflow candidates, unverified) or a new in-house held-out collection. No public dataset found that covers this.

## Comparison

| Dataset | Size | Classes | Mixed plates | TKPI overlap (cv lookup, 10 classes) | Licence | Download path | Verified? |
|---|---|---|---|---|---|---|---|
| Wibisono 2020 / In-TFK 2023 (Kaggle `arizbw`) | 1,644 (paper); 34 classes per paper and Roboflow split, 35 per Kaggle listing | 34-35 traditional foods | No | gado-gado, gudeg, pempek, rawon, rendang (exact); sate, soto, ayam-betutu, nasi-goreng-kampung only partial | CC BY 4.0 (article); Kaggle mirror CC BY-SA 4.0 | Dropbox folders in article; Kaggle `arizbw/traditional-food-knowledge-of-indonesia` (3.37 GB) | Yes (article, Kaggle page) |
| Rasyidi 2024 (BEEI 13(5)) | 24,427 images; 80/20 split; train+val 19,475, test 4,952 | 160 (incl. nasi campur, nasi padang, nasi uduk, gudeg, rawon, rendang, soto padang, gado-gado, ayam betutu, bakso) | Noted as a limitation (one prediction per image); not explicitly labelled | bakso, gudeg, gado-gado, rawon, rendang (direct); soto (padang, not ayam); nasi goreng (inferred) | Not stated in text read | **Not found** | Article yes; release no |
| Mendeley `vtjd68bmwt/1` (Ashari, ITS) | 10 classes; 70/30 train/test per page | 10 | Not documented; restaurant/roadside backgrounds | Not checked against the 10 cv classes here | CC BY 4.0 | Mendeley Data (DOI 10.17632/vtjd68bmwt.1) | Yes (page); training set for current CV model |
| Kaggle `rizkyyk/dataset-food-classification` ("Indonesian Food Dataset") | 6,500 (13 x 500) | 13: ayam goreng, burger, french fries, gado-gado, ikan goreng, mie goreng, nasi goreng, nasi padang, pizza, rawon, rendang, sate, soto ayam | Not documented; single-label | gado-gado, nasi goreng, rawon, rendang, sate, soto (6/10) | MIT (uploader) | Kaggle, 1.19 GB, 5 versions | Yes (Kaggle API); images web-scraped per uploader; provenance and label checks not done |
| Kaggle `faldoae/padangfood` (Afrinanto 2022) | ~900 (9 classes x ~100) | 9: rendang, ayam pop, dendeng balado, gulai tunjang, sate padang, telur balado, gulai nangka, sambal ijo, perkedel | Single dish per class folder | rendang (1/10) | Open Database; contents © original authors | Kaggle, 121 MB | Yes (Kaggle API, dataset page) |
| Kaggle `robertusbagaskara/indonesian-food-image` | Not checked | Not documented | Unknown | Unknown | "Unknown" | Kaggle, 252 MB | Partial (no description) |
| Mendeley LeFood-Set `cchsk79jkt/1` (Arumsari et al., 2025) | 524 before/after image pairs with weights and observer leftover levels | Food categories (not enumerated in fetched page) | Possibly (meal images) — not verified | Unknown | CC BY 4.0 | Mendeley Data (DOI 10.17632/cchsk79jkt.1) | Yes (page); food-class list not read |
| Roboflow `research-dsmj0/indonesian-traditional-food-dataset-split` | 1,643 images | 34 | No | Same set as Wibisono (re-host) | CC BY 4.0 (conflicts with source CC BY-SA 4.0) | Roboflow Universe | Yes (page) |
| Roboflow `bangkit/indonesian-food-pedsx` | 903 images | Not verified | Not verified | Unknown | CC BY 4.0 | Roboflow Universe | Page 403 (bot-blocked); from search summary only |
| Roboflow `yolo-obopn/indonesia-food-tjvly` | 516 images (detection) | Not verified | Detection boxes — could support mixed plates | Unknown | CC BY 4.0 | Roboflow Universe | Page 403; from search summary only |
| Roboflow `fusion-qvvyj/deteksi-makanan-khas-indonesia` | Not verified | Not verified | Detection | Unknown | Not verified | Roboflow Universe | Page 403 |
| HF `lacrisun/IndonesianFoods` | 1K–10K (per tags) | Not documented | Unknown | Unknown | Apache-2.0 | HuggingFace | Card has only licence; no docs |
| HF `worldcuisines/food-kb` | 2,414 rows; up to 8 images each; per-image licences | Knowledge base, not dish labels | Unknown | Unknown | Per image | HuggingFace (1.8 GB) | Yes (card); not a labelled classification set |

## Notes per candidate

### Wibisono et al. 2020 / In-TFK 2023 (strongest for controlled dish ID)
- Source: Wibisono et al., *Journal of Big Data* 7:69, 2020. https://journalofbigdata.springeropen.com/articles/10.1186/s40537-020-00342-5 (DOI 10.1186/s40537-020-00342-5). Article licence CC BY 4.0.
- 34 traditional foods, 1,644 images, studio capture at 45° and 90° (RAW converted to standardised format), 30–50 images per class.
- Regions: 5 Sumatra, 20 Java-Bali-Lombok, 3 Kalimantan, 5 Sulawesi, 3 Papua.
- Split: 70/20/10 train/test/val, plus a 3-fold scheme. Best AUROC 0.99 (fold 2), 0.88 (fold 1); DenseNet121 best.
- Download: Dropbox folders listed in "Availability of data and materials" (`dropbox.com/sh/88stvg9krafl7z8/...`, `dropbox.com/sh/a59su6gmzocebw4/...`). Also Kaggle `arizbw/traditional-food-knowledge-of-indonesia`.
- **Class-count discrepancy**: the article and the Roboflow split say 34; the Kaggle listing says 35 and names the classes (incl. laksa-bogor, lumpia-semarang, nasi-goreng-kampung, soto-banjar, tahu-telur). Reconcile before use; the Kaggle licence is CC BY-SA 4.0, not CC BY.
- In-TFK 2023 (DOI 10.1186/s40537-023-00728-1, https://link.springer.com/article/10.1186/s40537-023-00728-1) reuses the same 34-class studio set and cites Wibisono 2020 as its predecessor. Its availability section links a Dropbox folder (`a59su6gmzocebw4`) and a stats file. Its "publicly open" claim is confirmed only in the text.

### Rasyidi et al. 2024 (largest, mixed-dish classes, no release found)
- Source: Rasyidi et al., *Bulletin of EEI* 13(5):3355–3368, 2024. DOI 10.11591/eei.v13i5.7996. https://beei.org/index.php/EEI/article/download/7996/3915
- 160 classes, 24,427 images, collected by online search, labelled by 10 annotators.
- 80/20 split: train+val 19,475, test 4,952. Best model efficientnet_v2_l: 85.44% top-1, 97.84% top-5. 67 models across 16 architectures.
- Paper names one prediction per image as a limitation for multi-item images (nasi campur, sate with several sides).
- **Blocker**: no data availability statement or download link found in the text read. Searches for a GitHub/Zenodo/Kaggle release returned nothing. Contact the authors before relying on it.

### Mendeley `vtjd68bmwt/1` (current training set, not held-out)
- https://data.mendeley.com/datasets/vtjd68bmwt/1; published 1 Aug 2023, DOI 10.17632/vtjd68bmwt.1, Satrio Hanif Wicaksono Ashari (ITS). CC BY 4.0.
- 10 classes; web-scraped via Google Images (Fatkun), manually checked; 70/30 split on the page. Backgrounds include restaurant and roadside scenes.
- Model card `cv/MODEL_CARD.md`: MobileNetV2 trained on this set; split 2,293 / 573 / 1,193 (test acc 0.866, macro F1 0.858). These totals come from the model card, not the Mendeley page (which only gives the 70/30 split).
- Not usable as a held-out evaluation for the paper.

### Kaggle `rizkyyk/dataset-food-classification`
- https://www.kaggle.com/datasets/rizkyyk/dataset-food-classification. 6,500 images, 13 classes x 500, 1.19 GB, 5 versions, MIT (per Kaggle API).
- Web-scraped; the description says images were "carefully filtered" but gives no protocol. Uploader name differs from the one on the page ("esa maida" in API, "Rizky Yanuar K" in search result). **Verify labels and licence provenance before any use.**
- 6 of 13 classes overlap the cv TKPI lookup; no mixed plates documented.

### Kaggle `faldoae/padangfood`
- https://www.kaggle.com/datasets/faldoae/padangfood. ~900 images, 9 Padang classes (~100 each), 121 MB. Scraped via bing-image-downloader, manually reviewed, one folder per dish.
- Licence: "Database: Open Database, Contents: © Original Authors". Not a clean open licence.
- Single-dish labels. Useful only for Padang dish recognition, not mixed plates.

### Mendeley LeFood-Set `cchsk79jkt/1`
- https://data.mendeley.com/datasets/cchsk79jkt/1; 17 Jan 2025, DOI 10.17632/cchsk79jkt.1, Universitas Brawijaya and Okayama. CC BY 4.0.
- 524 pairs of Indonesian food images before and after eating, with weights and a 7-level observer leftover estimate. Designed for leftover estimation, not dish classification. Possibly relevant to portion or mixed-plate work. Class names not read.

### Roboflow sets
- `research-dsmj0/indonesian-traditional-food-dataset-split`: https://universe.roboflow.com/research-dsmj0/indonesian-traditional-food-dataset-split. 1,643 images, 34 classes, 70/10/20. Declared source is the Kaggle arizbw set. Licence shown as CC BY 4.0, which conflicts with the source's CC BY-SA 4.0 (derived work).
- `bangkit/indonesian-food-pedsx` (903 images, CC BY 4.0), `yolo-obopn/indonesia-food-tjvly` (516 images, detection, CC BY 4.0), and `fusion-qvvyj/deteksi-makanan-khas-indonesia`: Roboflow pages returned HTTP 403 (Cloudflare). Details come from search summaries only. Class names and mixed-plate content are unverified. Detection boxes are the only public route to multi-item plates that I found.

### HuggingFace
- `lacrisun/IndonesianFoods`: Apache-2.0, 1K–10K examples, CSV plus images. No documentation; unusable until inspected.
- `worldcuisines/food-kb`: 2,414 rows, up to 8 images each, per-image licences. A food knowledge base with text, not a dish-labelled classification set.

## TKPI overlap (cv/data/tkpi_macro_lookup.csv, 10 classes)

Classes in the lookup: bakso, bebek_betutu, gado_gado, gudeg, nasi_goreng, pempek, rawon, rendang, sate, soto. Note the lookup also flags bebek_betutu and sate as low mapping confidence.

- Exact or near-exact matches across candidates: gado-gado, gudeg, rawon, rendang (all candidates that list them), pempek (Wibisono), nasi goreng (rizkyyk; Rasyidi inferred).
- No candidate found for bakso in the Wibisono/rizkyyk sets; Rasyidi lists bakso.
- Soto: the lookup is soto ayam (bening), while Rasyidi lists soto padang and rizkyyk lists soto ayam (Wibisono lists soto ayam lamongan and soto banjar). Partial match only.
- Sate: the lookup is a generic average; candidates list sate-specific subtypes (sate-lilit, sate-maranggi, sate-ayam-madura, sate padang). Mismatch unless the TKPI row is split.
- Bebek betutu: no candidate. Wibisono has ayam-betutu (chicken, not duck).
- Nasi padang: in rizkyyk and Rasyidi; not in the TKPI lookup, so no macros.

## Recommendation (ranked)

1. **Wibisono 2020 / In-TFK 34-35 class set** for a controlled, held-out dish-ID benchmark. Clean licence path (article CC BY 4.0), public Dropbox and Kaggle download, person-level split not documented (images are studio, so person leakage is limited). Resolve the 34 vs 35 class count with the authors or by inspecting the Dropbox folder before citing.
2. **Rasyidi 2024** if the authors release it. It is the only candidate with nasi campur and nasi padang at scale. Contact corresponding authors; do not use until a licence and download exist.
3. **Mendeley `vtjd68bmwt/1`**: keep as the training/validation set for the current MobileNetV2 (CC BY 4.0, documented). Do not report it as held-out.
4. **Build a small in-house held-out set of mixed plates** (nasi campur, nasi padang), labelled per image, with person-level grouping. No public set found that covers this. [INFERENCE: this is the gap the paper must fill; public sets do not.]
5. Use Roboflow detection sets only after their pages are inspected (403 here).

Not recommended: Kaggle `rizkyyk` (web-scraped, uncertain provenance, MIT claimed by uploader), Kaggle `faldoae` (Open Database licence, single dish, ~900 images), HF `worldcuisines/food-kb` (knowledge base, not classification).

## Open items

- Confirm Wibisono class count (34 vs 35) against the Dropbox folder.
- Find a Rasyidi 2024 data release or contact the authors.
- Inspect Roboflow yolo-obopn, fusion-qvvyj, bangkit, and HF `lacrisun/IndonesianFoods` (pages blocked or undocumented here).
- Check the LeFood-Set class list for mixed-plate content.
- Regional non-Indonesian sets (THFOOD-100/50, VinaFood21, TypicalVietnameseFoodNet) were not checked for licence. Not in scope for Indonesian evaluation.

## Sources

- Mendeley Indonesian Food Image: https://data.mendeley.com/datasets/vtjd68bmwt/1
- Wibisono et al. 2020: https://journalofbigdata.springeropen.com/articles/10.1186/s40537-020-00342-5
- In-TFK 2023: https://link.springer.com/article/10.1186/s40537-023-00728-1
- Rasyidi et al. 2024: https://beei.org/index.php/EEI/article/download/7996/3915
- Kaggle rizkyyk: https://www.kaggle.com/datasets/rizkyyk/dataset-food-classification
- Kaggle faldoae padangfood: https://www.kaggle.com/datasets/faldoae/padangfood
- Kaggle arizbw TFK: https://www.kaggle.com/datasets/arizbw/traditional-food-knowledge-of-indonesia
- Kaggle robertusbagaskara: https://www.kaggle.com/datasets/robertusbagaskara/indonesian-food-image
- Mendeley LeFood-Set: https://data.mendeley.com/datasets/cchsk79jkt/1
- Roboflow research-dsmj0: https://universe.roboflow.com/research-dsmj0/indonesian-traditional-food-dataset-split
- Roboflow bangkit: https://universe.roboflow.com/bangkit/indonesian-food-pedsx
- HF lacrisun/IndonesianFoods: https://huggingface.co/datasets/lacrisun/IndonesianFoods
- HF worldcuisines/food-kb: https://huggingface.co/datasets/worldcuisines/food-kb
- Local: `cv/MODEL_CARD.md`, `cv/data/tkpi_macro_lookup.csv`
