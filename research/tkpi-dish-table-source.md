# Research: Indonesian food composition table (TKPI) source, cooked dishes, URT, GI, licence

Ticket: Irisearch-org/GlucoSight #8 (`tkpi-dish-table-source`), part of #5.

## Answer

- **Source to use:** Tabel Komposisi Pangan Indonesia (TKPI) 2017, Direktorat Gizi Masyarakat, Kemenkes RI (2018). It is the only full TKPI edition with a text layer. Official record: https://repository.kemkes.go.id/book/777. Full PDF used for this research (mirror, 135 pp): https://repository.stikespersadanabire.ac.id/assets/upload/files/docs_1634523137.pdf
- **Your 55-row CSV is TKPI 2017.** 51 of 55 rows match the 2017 PDF exactly on Air, Energi, Protein, Lemak, Karbohidrat. Four rows differ (table below). The 2020 edition is a scanned image; our OCR could not read it reliably, so we cannot say whether 2020 matches.
- **Licence: not open.** No CC or open licence found on any Kemenkes TKPI copy. The 2020 copy carries "© Copyright 2022 All Rights Reserved by Kemenkes" (https://repository.kemkes.go.id/book/668). The 2017 copy has no licence statement. Do not redistribute the table in the repo. Use values with citation, and request written permission before publishing a derived dataset. [INFERENCE] Whether individual nutrient values are copyrightable is a legal question not answered here.
- **Machine-readable:** no official CSV/JSON/API found. The 2017 PDF text layer is extractable, so a parser over the PDF is feasible. The 2020 PDF has no text layer.
- **Cooked dishes:** TKPI 2017 has a P (processed/cooked) code family. Roughly 90 entries are masakan (dish) entries. Only some of the 10 target dishes are present (table below). Several are absent or only partially covered.
- **Serving sizes (URT):** TKPI itself has no URT column. The best candidate found is the URT list in a Universitas Brawijaya thesis appendix (https://repository.ub.ac.id/id/eprint/129322/8/LAMPIRAN.pdf). Authority is unverified; the Kemenkes source it derives from must be checked.
- **GI:** No primary study giving GI for the 10 target dishes was found. Hakimah et al. 2020 gives menu-level GI for six red-rice packages (not single dishes). A commercial table (itokindo.org) lists GI for ~120 Indonesian dishes, but its sources are not verifiable; do not use it.

**Recommendation (ranked):**
1. Use TKPI 2017 as the composition source. Parse the PDF text layer. Keep only the rows the pipeline uses, with code, page, and source citation. Do not commit the full table. Reconcile the four CSV mismatches against the printed page before use.
2. Email Direktorat Gizi Masyarakat / Kemenkes for written permission to redistribute extracted values (required before any public artifact).
3. For dishes absent from TKPI (nasi goreng, bebek betutu, generic bakso, sate ayam/kambing, generic pempek), compute from recipe-level raw-ingredient entries (TKPI R codes) and flag as computed. Do not label them "Entri langsung tersedia".
4. Serving sizes: start from the UB URT list, flagged as secondary, and confirm against the Kemenkes/Persagi source before the paper.
5. GI: state in the paper that dish-level GI for Indonesian composite meals is not available from a primary source we could verify. Use Hakimah 2020 only as a menu-level reference.

## 1. Edition identification

TKPI 2017 evidence:
- Codes in the form AR/AP, BR/BP, CR/CP, DR/DP, ER/EP, FR/FP, GR/GP, HR/HP, JR/JP, KR/KP, MR/MP, NR/NP, QR/QP. R = raw, P = processed (olahan). Your CSV uses AR/AP codes throughout (21 AR, 34 AP).
- Values are per 100 g BDD, with a "SUMBER" source column (e.g., KZGPI-1990, KZGMI-2001). Your CSV has a `Source` column with value 1 on all rows. This has no counterpart in TKPI 2017 and needs flagging: its meaning is unknown.
- Section 7 maps old TKPI 2009 `IDxxx` codes to the new codes.

Numeric comparison (PDF section 4, all 55 codes found; values per 100 g):

| Code | Name (CSV) | CSV Air / Energi / Prot / Lemak / KH | TKPI 2017 | Note |
|---|---|---|---|---|
| AP014 | Ketupat ketan | 52.0 / 212 / 4.0 / **4.8** / 38.6 | "Ketan, ketupat": fat **4.6** | PDF fat fits energy (Atwater ≈212); PDF more likely right |
| AP022 | Mi kering | 10.0 / 339 / 10.0 / 1.7 / **67.3** | "Mi kering": Air **10.6**, KH **6.3** | PDF KH 6.3 does not fit energy (≈80 kcal vs 339); CSV KH 67.3 fits (≈324 kcal). Likely PDF typo; check printed page |
| AP030 | Bakpia, kue | **36.9** / 272 / 8.7 / 6.7 / 44.1 | "Bakpia, kue": Air **38.9** | Cannot decide from energy; check printed page |
| AP032 | Bantal | 52.3 / **180** / 3.3 / 0.3 / 43.5 | "Bantal": Energi **190** | PDF 190 fits macros (≈190 kcal); CSV 180 does not |

Name variants (same values): AP003 "Tapai beras" = "Beras, tapai"; AP004 "Tepung beras, mentah" = "Beras, tepung, mentah"; AP005 "Nasi beras merah" = "Beras merah, Nasi"; AP009 "Nasi jagung" = "Jagung nasi, mentah"; AP013 "Tepung jagung putih" = "Jagung, tepung putih"; AP016 "Tapai ketan hitam" = "Ketan hitam, tapai".

TKPI 2020 (`https://repository.kemkes.go.id/book/668`, 140 pp, scanned, no text layer): we OCR'd all pages (tesseract, 200 dpi). OCR quality was too poor to read the numeric rows; the four mismatch codes were not found. **2020 comparison: unverified.**

## 2. Columns (TKPI 2017, per 100 g BDD)

Air (g); Energi (kcal); Protein (g); Lemak (g); Karbohidrat (g, by difference); Serat (FIBC/FIBTG; the editors state it is unclear which items carry total vs dietary fibre); Abu; Ca; P; Fe; Na; K; Cu; Zn; Retinol; β-karoten; Karoten total; Thiamin; Riboflavin; Niasin; Vitamin C; BDD (%); SUMBER.

Methods (from the 2017 text): Soxhlet fat; Kjeldahl protein; carbohydrate by difference; AAS minerals; HPLC retinol/carotene; titration vitamin C. Some single raw foods are imputed or borrowed from ASEAN, Malaysian, Thai, Vietnamese, and USDA sources. Composite and processed foods are not imputed.

Caveat for fibre: raw and cooked fibre values are not separated by design; treat Serat as "unknown basis" in the pipeline.

## 3. Entry counts (TKPI 2017, section 4 food tables)

- Unique codes in section 4 (text lines 470–3440, code at line start): **1146**. R (raw) 558, P (processed) 588.
- Per group: DR 166, ER 112, DP 61, CR 52, CP 86, GR 84, GP 95, FR 34, FP 88, BR 33, BP 76, AR 21, AP 114, NR 24, NP 13, MR 18, MP 18, KR 14, KP 4, JR 6, JP 11, HR 11, HP 7, QR 1.
- Caveat: the count uses a start-of-line regex on extracted text; a few rows may be missed. P entries include processed foods and dishes, not just cooked dishes.

## 4. Coverage of the 10 pipeline dishes (`cv/data/tkpi_macro_lookup.csv`)

Values per 100 g from TKPI 2017; lookup values from the current CSV.

| Dish | TKPI 2017 entry | TKPI values (kcal / prot / fat / KH) | Lookup row (kcal / carbs / prot / fat) | Finding |
|---|---|---|---|---|
| gado_gado | DP031 "Gado-gado" (KZGPI-1990) | 137 / 6.1 / 3.2 / 21.0 | 137 / 10.0 / 6.0 / 8.0; "Entri langsung" | Energy matches; carbs, protein, fat do not. Lookup note is wrong |
| gudeg | DP032 "Gudeg, sayur" (nangka muda + santan) | 160 / 3.3 / 9.2 / 16.0 | 150 / 17 / 3.5 / 8 ("Gudeg nangka") | Close; different product (vegetable-style, not the Jogja dish) |
| rendang | FP062 "Rendang sapi, masakan" | 193 / 22.6 / 7.9 / 7.8 | 195 / 3.5 / 22.6 / 10.5; "Entri langsung" | Protein matches; carbs and fat differ. Lookup note is wrong |
| rawon | FP061 "Rawon, masakan / Jawa Timur" | 60 / 5.4 / 2.5 / 4.0 | 120 / 3.5 / 9 / 8 | Lookup values do not match the TKPI entry |
| pempek | GP077–GP083 (many "Pempek … masakan", fish-based) | GP077: 162 / 4.9 / 3.0 / 27.8; GP082 Pempek telur: 156 / 4.7 / 2.7 / 28.3; range 152–243 kcal | 190 / 27 / 9 / 5; "Pempek ikan", "Entri langsung" | No generic "Pempek ikan" entry; variants differ. Lookup note overstates |
| soto | FP074–FP084 (bandung, banjar, betawi, jeroan, kudus, madura, padang, pekalongan, pemalang, sukaraja, sulung) | e.g., FP076 Soto betawi: 135 / 2.5 / 8.8 / 11.5; FP080 Soto padang: 127 kcal | "Soto ayam" 90 kcal | No generic soto ayam. Pick a variant explicitly |
| bakso | AP086 "Mie bakso" only | 114 / – / – / 16.4 (KZGMS-1993) | "Bakso daging sapi", 190 kcal | No meatball entry in TKPI 2017. Lookup row is not TKPI-based |
| sate | FP064 "Sate, penyu, masakan"; FP065 "Sate pusut, masakan"; GP090 "Sate Bandeng" | – | "rata-rata" (average) | No sate ayam or kambing entry |
| nasi_goreng | **absent** (no "nasi goreng" in tables or glossary) | – | 186 / 27 / 4.5 / 6.5; "Entri langsung tersedia" | **Lookup claim is wrong for 2017.** Related: AP089 Nasi gemuk, AP090 Nasi gurih, AP091 Nasi minyak, AP092 Nasi rames |
| bebek_betutu | **absent** | – | lookup note says no entry | Consistent with TKPI 2017. Only FP003 "Bebek, daging, goreng" (300 kcal) is related |

Note: the lookup file marks four rows "Entri langsung tersedia" (gado_gado, nasi_goreng, pempek, rendang). None matches TKPI 2017 exactly. Recommend the dish-table ticket relabel these as "approximate" or "computed" and state which TKPI code is used.

Not verified: whether TKPI 2020 contains nasi goreng or other absent dishes. Our 2020 OCR could not confirm either way.

## 5. Serving sizes (URT)

No URT in TKPI 2017. Candidates:
- **UB thesis Lampiran 1, "Daftar Ukuran Rumah Tangga (URT)"** (https://repository.ub.ac.id/id/eprint/129322/8/LAMPIRAN.pdf). Read. Examples: Nasi 1 piring/1½ gelas = 100 g; Nasi jagung 1 gelas = 200 g; Mie basah 1 gelas = 100 g; Mie kering 1 gelas = 50 g; Bihun 1 gelas = 100 g; Bakso daging 6 biji besar = 100 g; Daging sapi (3×5×2 cm) 1 potong sedang = 50 g; Ayam 1 potong sedang = 50 g; Ikan segar (6×5×4 cm) = 50 g; Tahu (6×6×2 cm) 1 biji sedang = 75 g; Tempe (4×6×1 cm) 1 potong sedang = 25 g; Santan 1 gelas = 200 g; Kelapa parut 1 sdm = 6 g; Kecap 1 sdm = 10 g; Minyak goreng 1 sdm = 10 g; Bayam 25 batang = 100 g; Kangkung 20 batang = 100 g; Tauge 1 gelas = 70 g; Nangka muda 1 potong = 300 g; Terong 1 buah besar = 250 g. Conversion: 1 gelas = 240 ml, 1 sdm = 10 ml.
  - Provenance: an appendix to a Brawijaya thesis. The underlying Kemenkes/Persagi source was not checked. Treat as secondary.
- Kemenkes PMT technical guidance (ayosehat URL) returned HTTP 503 on fetch; not read.
- Kepdirjenkesprimkom 576/2025 (kesprimkom.kemkes.go.id PDF): not read.

No URT was found for gado-gado, rendang, rawon, or soto. These need a recipe-based portion (or a Kemenkes portion standard) before use.

## 6. Glycaemic index (GI) for Indonesian dishes

- **Hakimah N. et al. 2020**, Food Research 4(3):722–730, DOI 10.26656/fr.2017.4(3).377. PDF: https://www.myfoodresearch.com/uploads/8/4/8/5/84855864/_21__fr-2019-377_hakimah_1.pdf. Six red-rice packaged menus (nasi liwet, nasi goreng, nasi bakar, bubur manado, lontong gado-gado, cwiemie), n=10 non-diabetic and n=10 T2DM per design. Non-diabetic GI: liwet 23.8, nasi goreng 44.8, nasi bakar 54.5, bubur manado 53.5, lontong gado-gado 53.8, cwiemie 65.7. Serving sizes: liwet 900 g, goreng 450 g, bakar 535 g, manado 475 g, lontong gado-gado 750 g, cwiemie 220 g. **These are packaged menus of specific portions, not standard dishes. Use only as a reference.**
- **Hakimah 2019** (Indian J Forensic Med Toxicol 13(4):1557–1562, DOI 10.5958/0973-9130.2019.00524.3) is cited by the 2020 paper; not read.
- **itokindo.org table** (https://itokindo.org/wp-content/uploads/tabel-indeks-glikemik-IG-untuk-120-makanan-populer-di-Indonesia-r1.pdf, Jul 2025). Commercial marketing PDF with GI and GL for 120 foods (e.g., gado-gado 38, rawon 42, rendang daging 53, gudeg 54, pempek 65, soto betawi 52, nasi goreng spesial 71, sate ayam 57, bakso sapi 59). Sources are labelled "ASEAN GI DB 2025", "Badan POM RI (Laporan Teknis FD/2025/7890)", and "Indonesian Food J. (Vol.8)", but none is linked or checked. The table also contains an editorial note ("POSISI BENAR"). **Do not use for the paper.** Flag as an unverified secondary lead.
- **Sun L. et al. 2015**, "Glycaemic index and glycaemic load of selected popular foods consumed in Southeast Asia", Br J Nutr 113(5), https://www.cambridge.org/core/journals/british-journal-of-nutrition/article/glycaemic-index-and-glycaemic-load-of-selected-popular-foods-consumed-in-southeast-asia/6AB66E0D9E5ECC2F09D8936FAAF90B5F. Most promising primary source for SE-Asian dishes. Full food list not read (reader output truncated the abstract). **Gap: next step is to read the food list.**
- **Atkinson F.S. et al. 2021**, International Tables of GI and GL (AJCN, https://ajcn.nutrition.org/article/S0002-9165(22)00494-4/fulltext, PubMed 34258626). Indonesian dish rows not checked.
- **Adriani et al. 2025**, Jambura Med Health Sci J 4(2), DOI 10.37905/jmhsj.v4i2.33565. Staples only.
- **Arif, Budiyanto, Hoerudin 2013**, J Litbang Pert 32(2):91–99 (Indonesian GI review). Not read.
- Amerta Nutr "Indeks Glikemik Nasi Putih …" (https://e-journal.unair.ac.id/AMNT/article/download/37018/24969/218710): white rice only.

Conclusion: no verified primary GI for gado-gado, rendang, rawon, soto, sate, bakso, or pempek. Gap to report in the paper.

## 7. Licence

- TKPI 2017 (Kemenkes repository, book 777): no licence statement seen in the PDF text.
- TKPI 2020 (book 668): "© Copyright 2022 All Rights Reserved by Kemenkes".
- Panganku (https://www.panganku.org/id-ID/berita/20) describes TKPI 2017 as developed from TKPI 2009 with imputed and borrowed values, and says the government is not liable. It gives no open licence.
- **Conclusion:** no open licence. Cite the table; do not redistribute it. Request written permission for any derived dataset.

## 8. Discrepancy list (for the dish-table ticket)

1. AP014 fat: CSV 4.8, TKPI 4.6 (energy favours TKPI).
2. AP022 Air and KH: CSV Air 10.0 / KH 67.3; TKPI Air 10.6 / KH 6.3 (energy favours CSV KH). Check printed page.
3. AP030 Air: CSV 36.9, TKPI 38.9. Check printed page.
4. AP032 Energi: CSV 180, TKPI 190 (macros favour TKPI).
5. Lookup table "Entri langsung tersedia" claims for gado_gado, nasi_goreng, pempek, rendang: none matches TKPI 2017.
6. Lookup row values for rawon, gudeg, bakso, soto, sate do not match a single TKPI entry.
7. The CSV `Source` column (value 1 on all rows) has no TKPI 2017 counterpart.

## Method and limits

- Text: TKPI 2017 PDF text layer (5542 lines). Entries parsed with start-of-line code regex. Count and comparison are approximate where wrapped lines or OCR-like artefacts occur.
- 2020 PDF: OCR only (tesseract psm 6, 200 dpi). Unreliable for numbers.
- No code or data changes made. No model trained. The user CSV is not copied into the repo.
