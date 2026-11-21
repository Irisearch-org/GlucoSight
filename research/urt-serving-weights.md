# Research: Standard household serving weights (URT) for the 35 dishes and common single ingredients

Ticket: Irisearch-org/GlucoSight #22 (`urt-serving-weights`), part of #5.

## Answer

- **Best reachable authoritative URT source: Kemenkes Keputusan Direktur Jenderal Kesehatan Primer dan Komunitas HK.02.02/B/576/2025 (2025), PMT lokal juknis, Lampiran 2 (protein) and Lampiran 3 (karbohidrat), plus fat/fruit tables.** It is a regulation, current, has explicit URT→gram values for single ingredients. PDF: https://kesprimkom.kemkes.go.id/assets/uploads/contents/others/2025kepdirjenkesprimkom576.pdf (HTTP 200, checked).
- **It contains no composite dish** (gado-gado, rendang, rawon, soto, sate, pempek, gudeg, kolak, etc.). Dish-level weights therefore cannot be taken from it. **No authoritative dish-level URT for the 35 dishes was found in a readable primary source.**
- **Ingredient-level URT (staples, proteins, tahu/tempe, telur, fruit) agree across 576/2025 and the Kemenkes DBMP.** Use these for ingredient-level computation.
- **Composite-dish weights exist only in two Kemenkes primary documents that are not cleanly readable here:** the *Pedoman Konversi BDD 2014* (recipe-level ingredient lists for some dishes) and the *Buku Foto Makanan SKMI 2014* (photo-based average weights). Their PDFs returned HTTP 500 on the official Poltekkes host; the copies I could read are OCR-garbled. Per-dish grams from them are **unverified**.
- **Recommendation:** keep `serving_g` labelled as an assumed default for dishes without a verified dish-level source. Use 576/2025 and DBMP for ingredient anchors only. Obtain a clean copy of the BDD 2014 guide and the SKMI 2014 photo book from Badan Kebijakan (library record ID 34854 for BDD; SKMI book not yet located at the library) before recalibrating any dish. Do not use the UB thesis appendix list as an authority.

## Ranked sources

| Rank | Source | Authority | Licence (observed) | Dish coverage (of 35) | Grams per portion (examples) | Citation method |
|---|---|---|---|---|---|---|
| 1 | Kemenkes **Keputusan Dirjen Kesehatan Primer dan Komunitas HK.02.02/B/576/2025** (PMT lokal juknis, 2025), Lampiran 2–3 | Kemenkes regulation, current | Regulation; Indonesian statutory texts are excluded from copyright by UU 28/2014 Pasal 41 (list includes *peraturan perundang-undangan*; verify clause wording against the consolidated text before quoting) | 0 dish-level; ingredient anchors for 17 dishes | nasi 1 porsi ¾ gelas = 100 g; mie basah 2 gelas = 200 g; kentang 2 buah sedang = 210 g; talas ½ biji sedang = 125 g; ubi jalar 1 biji sedang = 135 g; telur ayam 1 butir = 55 g; ayam 1 potong sedang = 40 g (tanpa kulit) / 35 g (dengan kulit); daging sapi 1 potong sedang = 35 g; ikan mas 1/3 ekor = 45 g; tahu 2 potong sedang = 110 g; tempe 2 potong sedang = 50 g; santan 1/3 gelas = 40 g; pisang ambon 1 buah sedang = 50 g; pepaya 1 potong besar = 110 g | "Keputusan Direktur Jenderal Kesehatan Primer dan Komunitas Nomor HK.02.02/B/576/2025 (2025), Lampiran [2 atau 3], hlm. [x]". Cite the PDF URL above |
| 2 | Kemenkes **Daftar Bahan Makanan Penukar (DBMP)**, 8 golongan, "Satuan Penukar" tables | Kemenkes food-exchange list (origin per Labkesda Jatim citation: Kemenkes 2017) | Not stated in the copy I read; treat as Kemenkes publication with copyright reserved; cite only | 0 dish-level; ingredient anchors | nasi 100 g = ¾ gelas; mi basah 200 g = 2 gelas; bubur nasi 400 g = 2 gelas; kentang 210 g; singkong 120 g; tahu 110 g = 1 buah besar; tempe 50 g; telur 55 g; ayam tanpa kulit 40 g; ayam dengan kulit 35 g; ikan segar 40 g; udang 35 g (5 ekor); bakso 170 g (10 buah sedang); daging kambing 40 g; daging sapi 35 g; hati sapi 35 g; kacang hijau 20 g (2 sdm); oncom 40 g (2 potong kecil) | "Kementerian Kesehatan RI, Daftar Bahan Makanan Penukar, golongan [x], tabel [x]". **Copy used: secondary** (https://primayahospital.com/wp-content/uploads/2021/03/DAFTAR-BAHAN-PENUKAR-web.pdf, HTTP 200). Primary Kemenkes PDF not located |
| 3 | Kemenkes **Pedoman Konversi Berat Matang-Mentah, BDD dan Resep Makanan Siap Saji dan Jajanan** (Julianti ED et al., Badan Litbangkes, 2014) | Kemenkes/Badan Litbangkes; cited as the conversion guide in SDT 2014 (reference 9) | Badan Litbangkes publication; copyright reserved (standard Kemenkes/BPPK title notice; exact wording not verified in this copy) | Recipe-level entries for some dishes: gado-gado; sate ayam; bika ambon; pempek; lumpia; kue lumpur (garbled); soto mie; lontong sayur; ketupat tahu; sate padang; bubur ayam; mie bakso/mie goreng | Per recipe "1 porsi" (text garbled, so values below are indicative only): sate ayam ayam bakar 5 tusuk = 50 g; bubur ayam bubur 1 porsi = 200 g; lontong sayur ketupat 1 porsi = 140 g; sate padang ketupat 1 porsi = 120 g; soto mie mie basah 1 porsi = 50 g; mie bakso/mie goreng mie basah 1 porsi = 100 g; ketupat tahu ketupat 1 porsi = 200 g | "Julianti ED et al. (2014). Pedoman Konversi Berat Matang-Mentah, Berat Dapat Dimakan (BDD) dan Resep Makanan Siap Saji dan Jajanan. Jakarta: Badan Penelitian dan Pengembangan Kesehatan, Kemenkes RI." Library record: https://www.badankebijakan.kemkes.go.id/perpustakaanbkpkkemkes/index.php?id=34854. **Copy used: Scribd OCR (garbled)**, https://id.scribd.com/document/865957212/. Poltekkes PDF (https://eprints.poltekkesadisutjipto.ac.id/id/eprint/1985/) returned HTTP 500 |
| 4 | Kemenkes **Buku Foto Makanan SKMI 2014** (Tim Survei Konsumsi Makanan Individu, Pusat Teknologi Terapan Kesehatan dan Epidemiologi Klinik, Badan Litbangkes, 2014) | Kemenkes survey instrument. SDT 2014 lists it as the field photo aid (reference 11) | Badan Litbangkes publication; copyright reserved (assumed, as for SDT 2014) | Designed for dish-level estimation; **contents not verified** | Not retrieved. Search snippet of a Scribd copy describes it as containing "foto dan berat rata-rata berbagai jenis makanan". Per-dish values **unverified** | "Tim Survei Konsumsi Makanan Individu (2014). Buku Foto Makanan. Jakarta: Kementerian Kesehatan RI." Poltekkes PDF (https://eprints.poltekkesadisutjipto.ac.id/id/eprint/1911/) returned HTTP 500; Scribd copy (https://id.scribd.com/document/890656261/Buku-Foto-Makanan) returned no text |
| 5 | **DKBM** (Daftar Komposisi Bahan Makanan, Persagi 2001 and 2005; Depkes 1979/1993), URT column | Persagi/Depkes food composition table with URT | Unknown; cite only | Via the Banjar study below; not retrieved | Composite values cited in the Banjar study: laksa 1 porsi = 300 g (DKBM code 578); nasi goreng 1 piring = 60 g (DKBM code 586; looks implausible, treat with caution) | Cite via the Banjar study unless the DKBM PDF is obtained |
| 6 | Banjar URT study, Jurnal Skala Kesehatan 5(2), 2014. "Studi konversi satuan ukuran rumah tangga ke dalam berat (gram) pada beberapa jenis makanan tradisional... Banjar di Banjarmasin" | Peer-reviewed, local | Journal; not checked | Dish-level, local Banjar snacks and dishes: laksa, surabi, nasi goreng (see notes) | laksa 1 porsi = 300 g (DKBM); surabi 1 buah = 50 g (not in DKBM; local measurement); bikang 1 buah = 90 g; lapis kakao 1 potong = 50 g | Jurnal Skala Kesehatan 5(2): https://www.ejurnalskalakesehatan-poltekkesbjm.com/index.php/JSK/article/view/18 (HTTP 200, PDF read) |
| 7 | Aceh Simpang Tiga URT conversion study, SAGO: Gizi dan Kesehatan 1(1):101–112 (2019) | Peer-reviewed, rural Aceh | Journal; not checked | Ingredient/staple level only; no target dishes found on grep | Not used | https://ejournal.poltekkesaceh.ac.id/index.php/gikes/article/download/305/128 (HTTP 200, grep-checked only) |
| 8 | IPB *Jurnal Gizi dan Pangan* 3(1):49–60 (2008), "Konversi satuan ukuran rumah tangga ke dalam satuan berat (gram) pada beberapa jenis pangan sumber protein" | Peer-reviewed | Journal; not checked | Protein foods only; not read beyond the header | Not used | https://journal.ipb.ac.id/index.php/jgizipangan/article/download/4447/2988 (HTTP 200, not read in full) |
| 9 | Universitas Brawijaya thesis appendix, "Lampiran 1. Daftar Ukuran Rumah Tangga (URT)" | **Unattributed** (no Kemenkes/Persagi citation found) | Thesis; not checked | Ingredient-level only | Conflicts with 576/2025: kentang 1 bj bsr = 100 g (vs 210 g for 2 buah sedang); tahu = 75 g (vs 110 g) | https://repository.ub.ac.id/id/eprint/129322/8/LAMPIRAN.pdf (HTTP 200). **Low rank; do not cite as authority** |

Not located or not checked: PERSAGI standalone URT list; *Pedoman Gizi Seimbang* 2014 (no URT tables found, per earlier check); SDT 2014 report itself (no dish-level URT table; it names the BDD guide and photo book as field instruments). SDT 2014 report URL: https://repository.badankebijakan.kemkes.go.id/id/eprint/4741/1/Laporan_SDT2014.pdf (HTTP 200).

## Licence summary

- 576/2025 (regulation): cite freely. UU 28/2014 Pasal 41 excludes *peraturan perundang-undangan* from copyright; verify the clause wording before quoting it in a paper.
- DBMP, BDD 2014, Buku Foto Makanan 2014, SDT 2014 (Badan Litbangkes/Kemenkes publications): treat as copyright reserved. SDT 2014 title page reads "Hak Cipta dilindungi Undang Undang All right reserved". Cite and quote short values; do not redistribute scans or extracted tables in the repo.
- DKBM, Banjar study, Aceh study, IPB study: journal or institutional publications; cite; licence not checked.
- The UB thesis appendix: thesis; licence not checked; authority unverified.

## Per-dish coverage (35 rows)

Status key: **Dish-level (unverified)** = a dish-level gram value exists in a source that I could not verify as primary; **Ingredient anchor** = only single-ingredient URT applies; **None** = no URT found in reached sources. Current `serving_g` is the value in the dish table, shown for comparison only.

| # | Dish (slug) | Current serving_g | Best evidence | Grams per portion (source) | Status |
|---|---|---|---|---|---|
| 1 | asinan | 200 | none (vegetables: 576 gives no gram table) | — | None |
| 2 | ayam-betutu | 200 | ayam 1 potong sedang (576/DBMP) | 40 g (without skin) / 35 g (with skin) per potong; whole-bird portion not given | Ingredient anchor |
| 3 | ayam-bumbu-rujak | 200 | ayam 1 potong sedang (576/DBMP) | as row 2 | Ingredient anchor |
| 4 | ayam-goreng-lengkuas | 200 | ayam 1 potong sedang (576/DBMP) | as row 2 | Ingredient anchor |
| 5 | bika-ambon | 200 | BDD 2014 recipe entry (garbled) | Indicative only; no clean per-portion value | Dish-level (unverified) |
| 6 | bir-pletok | 250 | none (beverage) | — | None |
| 7 | bubur-manado | 250 | bubur nasi (DBMP); bubur manado is corn-based, so not equivalent | bubur nasi 400 g = 2 gelas (DBMP) | Ingredient anchor (different base) |
| 8 | cendol | 250 | none (santan 1/3 gelas = 40 g, 576) | — | None |
| 9 | es-dawet | 250 | none (beverage) | — | None |
| 10 | gado-gado | 250 | BDD 2014 recipe entry (garbled): labu siam 20 g, kangkung 30 g, wortel 20 g, kacang tanah 20 g | Per-portion total not parsed | Dish-level (unverified) |
| 11 | gudeg | 200 | none (nangka muda not in 576; nangka masak 3 biji sedang = 50 g is for ripe fruit) | — | None |
| 12 | gulai-ikan-mas | 200 | ikan mas 1/3 ekor = 45 g (576) | 45 g (ingredient) | Ingredient anchor |
| 13 | keladi | 200 | talas ½ biji sedang = 125 g (576; talas is the same genus group, verify botanical match) | 125 g | Ingredient anchor (closest match in 576) |
| 14 | kerak-telor | 200 | telur ayam 1 butir = 55 g (576/DBMP) | 55 g per egg (ingredient) | Ingredient anchor |
| 15 | klappertart | 200 | kelapa 1 potong kecil = 15 g; santan 1/3 gelas = 40 g (576) | ingredient only | None (dish) |
| 16 | kolak | 250 | pisang kepok 1 buah = 45 g; ubi jalar 1 biji sedang = 135 g (576) | ingredient only | Ingredient anchor |
| 17 | kue-lumpur | 200 | BDD 2014 entry (garbled, possible tepung terigu 10 g) | not parsed | Dish-level (unverified, weak) |
| 18 | kunyit-asam | 250 | none (beverage) | — | None |
| 19 | laksa-bogor | 300 | laksa 1 porsi = 300 g (DKBM code 578, Banjar study) | 300 g per porsi (DKBM, via Banjar) | Dish-level (unverified; Banjar laksa, not Bogor) |
| 20 | lumpia-semarang | 200 | BDD 2014 entry (garbled, lumpia: tepung terigu 7 g) | not parsed | Dish-level (unverified, weak) |
| 21 | mie-aceh | 300 | mie basah 2 gelas = 200 g (576/DBMP); mie basah 1 porsi = 100 g (BDD, mie bakso/mie goreng) | 100–200 g mie basah per portion (ingredient) | Ingredient anchor |
| 22 | nagasari | 150 | pisang ambon 1 buah sedang = 50 g; tepung beras 8 sdm = 50 g (576) | ingredient only | None (dish) |
| 23 | nasi-goreng-kampung | 250 | nasi ¾ gelas = 100 g (576/DBMP); DKBM "nasi goreng 1 piring = 60 g" (implausible, not used) | nasi base 100 g per ¾ gelas | Ingredient anchor |
| 24 | papeda | 250 | none (sagu: UB appendix only, 1 sdm = 6 g, low rank; not in 576) | — | None |
| 25 | pempek-palembang | 150 | BDD 2014 entry (garbled: tepung sagu 35 g, tepung terigu 5 g; ikan tenggiri 30 g) | not parsed | Dish-level (unverified, weak) |
| 26 | rawon-surabaya | 250 | daging sapi 1 potong sedang = 35 g (576) | ingredient only | Ingredient anchor |
| 27 | rendang | 100 | daging sapi 1 potong sedang = 35 g (576); santan 1/3 gelas = 40 g | ingredient only | Ingredient anchor |
| 28 | rujak-cingur | 200 | tahu 2 potong sedang = 110 g; tempe 2 potong sedang = 50 g (576) | ingredient only (cingur: no URT found) | Ingredient anchor |
| 29 | sate-ayam-madura | 120 | BDD 2014: sate ayam, ayam bakar 5 tusuk = 50 g | 50 g chicken (5 tusuk; BDD, readable only partially) | Dish-level (unverified, partial) |
| 30 | sate-lilit | 120 | ikan segar 1 potong sedang = 40 g (576) (ingredient only) | ingredient only | Ingredient anchor |
| 31 | sate-maranggi | 120 | daging sapi 1 potong sedang = 35 g (576) | ingredient only | Ingredient anchor |
| 32 | soerabi | 100 | surabi 1 buah = 50 g (Banjar study, not in DKBM; local) | 50 g per buah (Banjar; not the same as Bandung/Solo serabi) | Dish-level (unverified, local) |
| 33 | soto-ayam-lamongan | 250 | ayam 1 potong sedang = 40 g (576); soto mie mie basah 1 porsi = 50 g (BDD) | ingredient only | Ingredient anchor |
| 34 | soto-banjar | 250 | none (Banjar study names soto Banjar in the text, no URT row) | — | None |
| 35 | tahu-telur | 200 | tahu 2 potong sedang = 110 g; telur ayam 1 butir = 55 g (576) | ingredient only (sum 165 g raw) | Ingredient anchor |

Summary: 0 dishes have a verified dish-level URT. 8 have an unverified dish-level value (bika-ambon, gado-gado, kue-lumpur, laksa-bogor, lumpia-semarang, pempek-palembang, sate-ayam-madura, soerabi); 4 of these are weak or partial (kue-lumpur, lumpia-semarang, pempek-palembang, sate-ayam-madura). 17 have ingredient anchors only. 10 have no URT found (asinan, bir-pletok, cendol, es-dawet, gudeg, klappertart, kunyit-asam, nagasari, papeda, soto-banjar).

## Recommendation

1. **Ingredient level:** adopt 576/2025 as the citable Kemenkes URT source. Cross-check values against DBMP (they agree on the staples and proteins above). Replace the UB thesis list entirely.
2. **Dish level:** no verified dish-level URT is available. Keep `serving_g` as an assumed default and mark it as such in the dish table. Do not claim a Kemenkes or DBMP source for any dish.
3. **Obtain the two primary Kemenkes documents in clean form before recalibrating dish weights:**
   - Pedoman Konversi BDD 2014 (library record https://www.badankebijakan.kemkes.go.id/perpustakaanbkpkkemkes/index.php?id=34854). A clean PDF is needed; the Poltekkes host currently returns HTTP 500.
   - Buku Foto Makanan SKMI 2014. This is the most relevant source for "one standard portion". It is not yet located in a readable copy. Request it from Badan Kebijakan (perpustakaan.badankebijakan.kemkes.go.id) or the Poltekkes repository.
4. **Local sources for specific dishes:** laksa (300 g, DKBM via Banjar) and surabi (50 g per buah, Banjar) are the only dish-level candidates with a printed gram value. Treat them as local-variant values, not as national standards.
5. **Licence:** do not redistribute DBMP, BDD, SKMI, or SDT tables in the repo. Cite them and quote short values only.

## Unverified

- Per-dish values from the BDD 2014 guide (OCR garbled; the parse above is indicative). A clean PDF is required.
- SKMI 2014 photo book contents (not retrieved). Its composite-dish weights could change the recommendation.
- DBMP primary Kemenkes PDF (only a hospital copy was read; cited as secondary).
- DKBM (Persagi 2001/2005) URT values, cited only as quoted by the Banjar study (laksa 300 g). The nasi goreng value (60 g per piring) looks implausible and was not used.
- Licence wording for DBMP, BDD, and SKMI primary copies (not seen in full).
- Botanical match of talas (576) to keladi; treat as an ingredient-level approximation.
- PERSAGI standalone URT list: not located.
- Aceh (2019) and IPB (2008) studies: grep- or header-checked only.
- 576/2025 Pasal 41 wording (UU 28/2014): checked via secondary commentary (Lex Privatum), not the consolidated text.
- No GI, cooked-to-raw conversion, or post-meal glucose data is covered by this ticket.

## Citations (all URLs checked with HTTP status where noted)

- Kemenkes Keputusan Dirjen HK.02.02/B/576/2025: https://kesprimkom.kemkes.go.id/assets/uploads/contents/others/2025kepdirjenkesprimkom576.pdf (200)
- DBMP (secondary copy): https://primayahospital.com/wp-content/uploads/2021/03/DAFTAR-BAHAN-PENUKAR-web.pdf (200)
- BDD 2014 library record: https://www.badankebijakan.kemkes.go.id/perpustakaanbkpkkemkes/index.php?id=34854
- BDD 2014 Scribd copy (OCR): https://id.scribd.com/document/865957212/Pedoman-Konversi-Berat-Matang-Mentah-Berat-Dapat-Dimakan-dan-Resep-Makanan-Siap-Saji-dan-Jajanan (200, garbled)
- SKMI 2014 Buku Foto Makanan (Poltekkes, 500 at fetch time): https://eprints.poltekkesadisutjipto.ac.id/id/eprint/1911/
- SDT 2014 report: https://repository.badankebijakan.kemkes.go.id/id/eprint/4741/1/Laporan_SDT2014.pdf (200)
- Banjar URT study: https://www.ejurnalskalakesehatan-poltekkesbjm.com/index.php/JSK/article/view/18 (200)
- Aceh URT study: https://ejournal.poltekkesaceh.ac.id/index.php/gikes/article/download/305/128 (200)
- IPB URT protein study: https://journal.ipb.ac.id/index.php/jgizipangan/article/download/4447/2988 (200)
- UB thesis appendix: https://repository.ub.ac.id/id/eprint/129322/8/LAMPIRAN.pdf (200)
- UU 28/2014 (copyright, Pasal 41): https://dgip.go.id/unduhan/download/uu-nomor-28-tahun-2014-tentang-hak-cipta-34-2014 (search-result snippet, not read in full)
