# CV Track — Literature Review: from a 10-class classifier to a food model that can feed fusion

**Date:** 2026-10-18
**Scope:** research only — no code changed, no model trained. Answers the five
questions raised against the track's current state (`cv/MODEL_CARD.md`,
`cv/vlm_vs_mobilenet.md`, `docs/METRICS.md` §2, `docs/REVIEW_FINDINGS.md` C5,
`docs/INTEGRATION_AUDIT.md`, `docs/BIGIDEAS_MEAL_RESPONSE_FINDINGS.md`).
**Status:** findings only. The recommendation section flags where this project's
own `CLAUDE.md` framing may need to change; that decision is explicitly left to
the PM/user, not made here.

Every claim below is sourced from a paper, dataset page, or primary vendor
document actually fetched during this review — URL/DOI and the specific
figure cited inline. Secondary write-ups (blog posts) are labelled as such and
used only where they supply evidence (e.g. of production deployment) that no
peer-reviewed source covers yet.

---

## 1. State of the art: fine-grained / long-tailed food recognition

The field's own framing of this problem matches GlucoSight's exactly — "food
recognition belongs to fine-grained classification... [and] the food domain
exemplifies a long-tailed distribution problem, where certain dishes are
significantly less prevalent" (Rodriguez-De-Vera et al., **"LOFI: LOng-tailed
FIne-Grained Network for Food Recognition,"** CVPR 2024 Workshop on Multimedia
Assisted Dietary Management,
[openaccess.thecvf.com](https://openaccess.thecvf.com/content/CVPR2024W/MTF/papers/Rodriguez-De-Vera_LOFI_LOng-tailed_FIne-Grained_Network_for_Food_Recognition_CVPRW_2024_paper.pdf)).
LOFI combines a multi-task fine-grained classification head (using lexical
similarity between dish names during training), a Graph Confidence
Propagation module, an **equalization loss** and a **normed linear layer** in
place of standard cross-entropy/softmax, and reports **+4.6 mAP on
UECFood-256** and **+2.2 mAP on S-UECFood-100** over prior SOTA. None of these
are generic ImageNet tricks — all four exist specifically to counter
long-tailed class imbalance and high inter-class similarity.

**Scale of the problem.** Min et al., **"Large Scale Visual Food
Recognition" (Food2K),** arXiv:2103.16107 / IEEE TPAMI 2023
([arxiv.org/abs/2103.16107](https://arxiv.org/abs/2103.16107)), is the
largest public food dataset: **2,000 categories, 1,036,564 images**, built
because "we have witnessed remarkable advancements in generic visual
recognition... yet [progress] largely lags in the food domain." Their own
comparison table of prior benchmarks is directly useful context for how small
this project's 10-class set is: Food-101 (101 classes/101K images),
Vireo Food-172 (172/110K), ISIA Food-500 (500/400K), FoodX-251 (251/159K),
UEC Food-256 (256/25K). Their proposed model (PRENet — progressive region
enhancement network) is explicitly designed because "existing methods for
food recognition need large-scale labeled samples... they cannot handle food
categories with few samples" — the same complaint as this project's 173–293
images-per-class Mendeley set, at Food2K's scale rather than ten classes.

**Formal long-tailed benchmarks and methods.** Jiang et al.,
**"Long-Tailed Food Classification,"** arXiv:2210.14748
([ar5iv.labs.arxiv.org/html/2210.14748](https://ar5iv.labs.arxiv.org/html/2210.14748)),
construct **Food101-LT** (Pareto-distributed subsample of Food-101, imbalance
ratio 150, as few as 5 images for a tail class) and **VFN-LT** (consumption-
frequency-weighted real US diet data, imbalance ratio 288, as few as **1**
image for a tail class), and propose a 2-phase undersample-head /
visual-aware-oversample-tail framework. **This is the closest published proxy
to GlucoSight's actual problem**: a food class list where most classes have
only a handful of training images, because that is what a long-tailed real
diet looks like.

**Directly on-target for the clinical population.** He et al.,
**"Long-Tailed Continual Learning for Visual Food Recognition,"**
IEEE Transactions on Multimedia, 2025,
([pmc.ncbi.nlm.nih.gov/articles/PMC12680007](https://pmc.ncbi.nlm.nih.gov/articles/PMC12680007/)),
introduce a 186-American-food dataset and **three benchmarks split by
clinical population: VFN186-LT, VFN186-INSULIN, and VFN186-T2D** — "real-world
food consumption for healthy populations, insulin takers and individuals with
type 2 diabetes without taking insulin." This is a direct precedent for
building a long-tailed food-recognition benchmark stratified by *exactly*
GlucoSight's target population (T2D), using knowledge-distillation and a
CAM+CutMix augmentation to improve rare-class generalisation without
catastrophic forgetting of common classes.

**Few-shot approaches, for when there simply isn't a large labelled set per
class.** Jiang, Min et al., **"Few-shot Food Recognition via Multi-view
Representation Learning,"** ACM TOMM 2020
([dl.acm.org/doi/fullHtml/10.1145/3391624](https://dl.acm.org/doi/fullHtml/10.1145/3391624)),
note "there is more than 8,000 food categories... like other object
categories, there is a long-tailed distribution for real-world food data,
where many categories have few food samples" and propose fusing
category-level with **ingredient-level** features specifically because
ingredient composition generalises across dishes the model has never seen
labelled examples of. Wu et al., **"Few-shot Food Recognition with
Pre-trained Model,"** ACM 2022
([dl.acm.org/doi/pdf/10.1145/3552485.3554939](https://dl.acm.org/doi/pdf/10.1145/3552485.3554939)),
fine-tune **CLIP** with a lightweight adapter on a handful of examples per
class, explicitly because "CLIP has shown remarkable zero-shot capability on
a wide range of vision tasks [but] performs poorly in the domain-specific food
recognition task" without adaptation — evaluated on VIREO Food-172 and
UECFood-256.

**What this answers:** nobody in this literature treats "hundreds of classes,
few examples each" as solved by scaling up a plain closed-set CNN classifier
(which is this project's current architecture, per `CLAUDE.md`'s own tech
stack: "CV: PyTorch, EfficientNet-B3, ViT, timm"). The published answers are
long-tailed-loss engineering (equalization loss, normed classifier heads,
class-balanced resampling), auxiliary signal fusion (ingredient/name-component
labels), or few-shot transfer from a large pretrained vision-language
embedding (CLIP-style adapters) — because for a genuinely long-tailed 400+
dish distribution, "collect more photographs per class" does not scale the
way it did for the 10-class, 173–293-images-per-class Mendeley set.

---

## 2. Indonesian / Southeast Asian food recognition — what exists

**The dataset this project already trains on.** Ashari, **"Indonesian Food
Image,"** Mendeley Data v1, DOI
[10.17632/vtjd68bmwt.1](https://data.mendeley.com/datasets/vtjd68bmwt/1)
(2023). Ten classes, images gathered by scraping Google Images with the
Fatkun browser extension per food-name keyword, manually checked, 70/30
train/test split. No weighed or verified macro data of any kind is attached —
it is a pure image-classification set, assembled by a single contributor via
web scraping. This confirms the model card's own description and is worth
flagging explicitly: **the training images were never curated against a
nutrition source; the TKPI macro values were joined afterward, by food name,
as a completely separate step** (same structure as the risk the model card
already documents in §4 of `cv/MODEL_CARD.md`).

**Larger Indonesian efforts, in order of scale:**

| Source | Classes | Images | Best reported accuracy | Macro ground truth? |
|---|---|---|---|---|
| Sarwinda et al., 2020, IC2IE, DOI [10.1109/ic2ie50715.2020.9274636](https://doi.org/10.1109/ic2ie50715.2020.9274636) | 10 | 200 total | "100%" on a 30-image test split | No |
| Mendeley "Indonesian Food Image" (this project's set) | 10 | ~4,059 | 86.6–88.0% (self-contradictory, METRICS.md §7.2) | No (joined separately) |
| Wibisono et al., 2020, *Journal of Big Data*, DOI [10.1186/s40537-020-00342-5](https://doi.org/10.1186/s40537-020-00342-5) | 34 | 1,644, studio-controlled lighting/distance | AUROC > 0.95 (DenseNet121/ResNet50/InceptionV3/NASNetMobile) | No |
| Darojat et al., 2021, J-PTIIK Universitas Brawijaya, [j-ptiik.ub.ac.id](https://j-ptiik.ub.ac.id/index.php/j-ptiik/article/download/10096/4481/) | 9 | 585 (900 after flip augmentation) | **test accuracy 0.60**, train/eval accuracy 0.91 — explicit overfitting | No |
| **Rasyidi et al., 2024,** *Bulletin of Electrical Engineering and Informatics*, DOI [10.11591/eei.v13i5.7996](https://beei.org/index.php/EEI/article/download/7996/3915) | **160** | **24,427** (scraped, 10 human annotators) | **EfficientNetV2-L 85.44%** top-1 (67 models × 16 architectures compared) | **No** |

Rasyidi et al. 2024 is the closest published analogue to "the next step up"
from this project's current set: 16× the class count at roughly the same
top-1 accuracy as this project's 10-class number (85.44% vs. the disputed
86.6–88.0%, METRICS.md §7.2/§7.3). The paper's own images were collected "using
an online search engine" and "divided and distributed to 10 human annotators
... there is the potential for errors in labeling" — i.e. the same scraped,
manually-checked methodology this project's own Mendeley set used, at larger
scale, and it still carries **no nutrition annotation of any kind** (confirmed
from the full paper text — the dataset is built purely for food *recognition*,
for an elderly-nutrition-monitoring app, with macro lookup left as future
work).

**Regional comparison (non-Indonesian Southeast Asian cuisines), for calibrating
what's achievable at 100+ classes:**

- **THFOOD-100** (Thai), Theera-Ampornpunt & Treepong, IEEE 2024,
  [ieeexplore.ieee.org/document/10759670](https://ieeexplore.ieee.org/document/10759670):
  100 classes, 53,459 images, best model **96%** top-1 (23 CNN/ViT
  architectures compared, cyclical learning-rate schedule).
- **VinaFood21** (Vietnamese), arXiv:2108.02929
  ([arxiv.org/abs/2108.02929](https://arxiv.org/abs/2108.02929)): 21 classes,
  13,950 images, EfficientNet-B0, **74.81%** average accuracy.
- **myFood11** (Malaysian), [github.com/mohdsubhi/myFood11](https://github.com/mohdsubhi/myFood11):
  11 classes, small scraped set, cited from Subhi & Ali, IEEE-EMBS 2018.
- **TypicalVietnameseFoodNet**, Cao & Duong, ICCAS 2021,
  DOI [10.23919/iccas52745.2021.9649805](https://doi.org/10.23919/iccas52745.2021.9649805):
  **94.84%** on their own custom architecture/dataset.

**What this answers:** Indonesian-specific food recognition work exists up to
160 classes (Rasyidi et al. 2024) and regional SEA work exists up to 100
classes with very high accuracy (THFOOD-100, 96%) — both well short of the
**455 distinct dishes** measured in `docs/BIGIDEAS_MEAL_RESPONSE_FINDINGS.md`
B7/B9 and cited as CV11 in `docs/METRICS.md`. Crucially, **none of the
Indonesian or regional SEA datasets found carries macro/nutrition ground
truth** — every one is a pure classification set, with nutrition joined
afterward (if at all) from a separate table like TKPI. This is a structural
gap in the literature, not a gap specific to this project's data pipeline.

---

## 3. VLM-based (zero/few-shot) food recognition vs. a trained closed-set classifier

**Primary research evidence.**

- Lo et al., **"Dietary Assessment With Multimodal ChatGPT: A Systematic
  Analysis,"** IEEE J. Biomedical and Health Informatics, 2024,
  DOI [10.1109/jbhi.2024.3417280](https://doi.org/10.1109/jbhi.2024.3417280):
  GPT-4V, zero-shot, **no fine-tuning or food-specific adaptation**, tested
  on a self-collected wearable-camera dietary dataset. Key finding directly
  relevant to a 10-class-to-many-dishes problem: **prompt-based cultural
  steering works** — "by guiding the model with specific language prompts
  (e.g., African cuisine), it shifts from recognizing common staples like
  rice and bread to accurately identifying regional dishes like banku and
  ugali." The same paper notes GPT-4V "can leverage surrounding objects as
  scale references to deduce portion sizes."
- O'Hara et al., **"Using artificial intelligence to estimate the nutritional
  content of meal photos: an evaluation of ChatGPT-4,"** Proc. Nutrition
  Society, Cambridge, 2024,
  [cambridge.org](https://www.cambridge.org/core/journals/proceedings-of-the-nutrition-society/article/using-artificial-intelligence-to-estimate-the-nutritional-content-of-meal-photos-an-evaluation-of-chatgpt4/13884EF11C5B86381C09748533D2E5B3):
  38 meal photos vs. McCance & Widdowson ground truth. Absolute agreement
  poor (p<0.05 difference for 11/16 nutrients, >10% difference for 12/16);
  **ranking/quartile agreement acceptable-to-good for all 16 nutrients**
  (good rs ≥0.50 for 11/16); ChatGPT underestimated 15/16 nutrients.
- Yanai lab (Japan), **"Using One-Shot Prompting of Non-Fine-Tuned Commercial
  AI to Assess Nutrients from Photographs of Japanese Meals,"** PMC,
  [pmc.ncbi.nlm.nih.gov/articles/PMC11724398](https://pmc.ncbi.nlm.nih.gov/articles/PMC11724398/):
  GPT-4o, one-shot, no fine-tuning, 22 meals vs. weighed food records. Poor
  for energy/fat/protein/carbs/salt (ICC ≤0.36), but **fiber ICC 0.71 —
  better than the comparison dietitians' 0.57.** Flags model-version
  instability: "gpt-4o-mini" and the updated "gpt-4o-2024-08-06" both scored
  markedly worse than "gpt-4o-2024-05-13" on the identical task.
- **DietAI24**, Yan et al., *Current Developments in Nutrition*, 2025,
  [cdn.nutrition.org](https://cdn.nutrition.org/article/S2475-2991(25)02366-2/fulltext):
  GPT-4 Turbo + USDA FNDDS retrieval (5,624 standardized food items), **zero
  food-specific training data**, evaluated against ASA24 and Nutrition5k.
  Reduced MAE **76–83% for calories, 80–86% for fat, 71–78% for carbohydrate,
  67–88% for protein** versus three commercial apps (Calorie Mama, Foodvisor,
  SnapCalorie) *and* a trained ViT, on the same images. This is the strongest
  single piece of evidence that an LLM-plus-nutrition-database pipeline with
  no training data can outperform both commercial closed-set apps and a
  dedicated trained vision model.
- CalorieVoL / Tanabe & Yanai, MDPI *Nutrients* 2025,
  [pmc.ncbi.nlm.nih.gov/articles/PMC11990770](https://pmc.ncbi.nlm.nih.gov/articles/PMC11990770/)
  (extends ICPR-W 2024 CalorieLLaVA and MMM 2025 CalorieVoL,
  [mm.cs.uec.ac.jp](https://mm.cs.uec.ac.jp/e/pub/conf24/250109tanabe_0.pdf)):
  GPT-4V and LLaVA zero-shot on Nutrition5k; injecting an **external** volume
  estimate (from a dedicated depth/segmentation model, not from the raw image
  alone) measurably improves MAE and correlation. Food*LMM* (cited within)
  "has demonstrated strong performance... However, its accuracy is limited by
  insufficient volume recognition capabilities" — the recurring theme that
  VLMs are weak specifically at volume, not identity.
- **DiningBench**, Jin et al., ACL 2026 Main, arXiv:2604.10425
  ([arxiv.org/abs/2604.10425](https://arxiv.org/abs/2604.10425)): 29
  open-source and proprietary VLMs, 3,021 distinct dishes, multi-view images,
  verification-based nutrition ground truth — the first benchmark built to
  stress-test exactly this question at scale. Abstract confirms VLMs "excel
  at general reasoning [but] struggle significantly with fine-grained visual
  discrimination and precise nutritional reasoning," with five identified
  failure modes. *(Specific per-model MAPE numbers — Gemini-class ~24%,
  GPT-5-class ~32% — are quoted by a practitioner write-up citing this paper,
  danilchenko.dev, below; this review read only the DiningBench abstract and
  could not independently confirm the exact figures from the paper body, so
  treat those numbers as reported-by-a-secondary-source pending direct
  verification of the full paper.)*

**Production deployment evidence — directly on this project's exact
application (glucose/diabetes + photo food logging):**

- **Dexcom press release, July 2025,**
  [investors.dexcom.com](https://investors.dexcom.com/news/news-details/2025/Dexcom-Launches-Revolutionary-AI-Powered-Meal-Logging-Feature-Across-Glucose-Biosensing-Portfolio/default.aspx):
  "AI-Powered Smart Food Logging" shipped to **Stelo** (FDA-cleared
  over-the-counter glucose biosensor) and **Dexcom G7** — "take a photo of
  their food... AI will automatically identify the ingredients." This is a
  primary vendor source confirming VLM-style photo food recognition is
  already in production, at scale, specifically paired with a CGM product.
- **LogMeal × Evidenze Health Group (Grupo Pulso) case study,**
  [logmeal.com/case-study](https://logmeal.com/case-study/ai-based-food-intake-monitoring-for-type-2-diabetes-management):
  AI food-recognition API deployed in a **T2D-specific** digital coaching
  platform, "validated in hospital settings in Spain and Turkey," used for
  carbohydrate/glycemic-load monitoring with clinician-facing alerts.
- Multiple independent consumer/prosumer diabetes apps found with
  photo-to-AI carb estimation as their core feature — **Insuu**
  ([insuu.ai](https://insuu.ai/)), **GlucIQ**
  ([gluciq.com](https://gluciq.com/)), **GloGlo**
  ([gloglo.com/app](https://gloglo.com/app)), **GlukoMate**
  ([adoztech.com](https://www.adoztech.com/en/projects/glukomate), explicitly
  built on Gemini via Firebase AI Logic), and **January AI**
  ([nutritioninsight.com](https://www.nutritioninsight.com/news/january-ai-health-app-medical-records-glucose.html),
  "nearly 200,000" users, Mayo Clinic Platform integration, listed in the CMS
  Medicare App Library's diabetes/obesity category). None of these report
  using a trained closed-set classifier; all use a general VLM API.
- **Secondary/practitioner source**, danilchenko.dev, **"LLM Food Recognition
  in Production: What Shipping soba Taught Me"**
  ([danilchenko.dev](https://www.danilchenko.dev/posts/food-recognition-llm/)):
  a solo-developer carb-counting app for T1D. Reports (self-measured, not
  peer-reviewed): every VLM tested (multiple Gemini/GPT variants) "misjudges
  portion weight by 25–35%," that gap "dwarfs any gap between frontier models
  on food identity," and that model choice was "the smallest lever" versus
  prompt engineering (explicit scale anchoring: "the photo is approximately
  31 cm wide") and a user-facing portion-correction UI. Production cost
  $0.005–0.012/scan, 2.6–8.3s latency, multi-model API fallback via
  OpenRouter. Cited only as deployment-pattern evidence (prompt design,
  fallback architecture, portion as the residual weak point) — not as a
  benchmark number, since it is not peer-reviewed.

**What this answers, on cost/latency/accuracy tradeoffs:**
*Accuracy* — VLMs decisively win on open-vocabulary dish identity (this
project's own `cv/vlm_vs_mobilenet.md` already measured Gemini 95.96% vs.
MobileNetV2 70.71% top-1 on the same 99 held-out images); VLMs are
consistently weaker, or at best comparable, on *absolute* macro numbers
unless paired with a nutrition-database retrieval step (DietAI24's design,
and this project's own finding that TKPI-lookup-from-VLM-class, MAE 0.05g,
beats VLM-direct-macro-output, MAE 5.36g, in `vlm_vs_mobilenet.md`).
*Cost/latency* — real, but small at this project's scale: a few cents and a
few seconds per call, acceptable for feature-extraction pipelines and
demo-scale usage, problematic only for a tight-latency live-camera UX, which
this project's own report already observed in miniature (one HTTP 503 and
~84 minutes wall-time for 100 calls due to rate-limit pacing). *Deployment
precedent* — strong: a VLM-style photo food-recognition pipeline is the
standard production architecture for diabetes-specific apps in 2025–2026,
including inside an FDA-cleared CGM product (Dexcom) and a clinically
validated T2D coaching platform (LogMeal/Evidenze), not a research novelty.

---

## 4. Portion/macro estimation from a single photo — and whether CLAUDE.md's framing is mis-weighted

**The primary ground-truth dataset and paper.** Thames et al.,
**"Nutrition5k: Towards Automatic Nutritional Understanding of Generic
Food,"** CVPR 2021, DOI
[10.1109/cvpr46437.2021.00879](https://doi.org/10.1109/cvpr46437.2021.00879),
arXiv:2103.03375
([arxiv.org/html/2103.03375v2](https://arxiv.org/html/2103.03375v2)), dataset
at [github.com/google-research-datasets/Nutrition5k](https://github.com/google-research-datasets/Nutrition5k).
**5,006 real, weighed dishes**, 250+ ingredients, incrementally weighed and
scanned (RGB video, overhead RGB-D for 3,500 of them). This is the field's
closest thing to a controlled experiment isolating exactly the question this
project needs answered — *what does a single 2D photo alone get you,
compared to when portion/mass is known or depth is available*. Their own
Table 3 (MAE and MAE as % of mean ground truth):

| Model | Calorie MAE | Carb MAE | What it needs |
|---|---|---|---|
| Baseline (predict the mean) | 150.8 (60.2%) | 12.5 g (62.1%) | nothing |
| **2D portion-independent** (predicts macro *density*; mass supplied externally) | 24.1 (9.5%) | 2.7 g (13.9%) | mass known |
| **2D direct prediction** (image alone; must infer mass too) | **70.6 (26.1%)** | **6.1 g (31.9%)** | **image only** |
| RGB + depth as 4th channel | 47.6 (18.8%) | 4.6 g (23.8%) | a depth sensor |
| RGB + explicit volume estimate from depth | 41.3 (16.5%) | 4.5 g (22.0%) | a depth sensor |

The gap between row 2 and row 3 is the entire answer to "what is achievable
from a single 2D photo with no depth, no scale reference, no multi-view":
**error roughly triples (carb MAE 13.9% → 31.9%) the moment the model must
also recover portion mass from the photo alone**, and the only rows that
close most of that gap (rows 4–5) require a depth sensor this project's
target device (a phone camera photographing a plate) does not have.

**Independent confirmation that this ceiling is architecture-general, not a
Nutrition5k artefact.** CalorieVoL (§3 above) shows the same pattern for
VLMs specifically: GPT-4V/LLaVA zero-shot on Nutrition5k improves measurably
only when an *external* volume estimate (from a separate depth/segmentation
model) is injected into the prompt — the VLM cannot resolve it from the RGB
image alone either. The soba.dev practitioner report (§3, secondary source)
independently reaches the same conclusion from production use across
multiple current-generation VLMs: "every model I tested misjudges portion
weight by 25–35%... that error dwarfs any gap between frontier models on
food identity."

**Relation to this project's own C5/B7 finding.** `docs/REVIEW_FINDINGS.md`
C5 and `docs/BIGIDEAS_MEAL_RESPONSE_FINDINGS.md` B7 independently measured,
on real paired meal-and-glucose data, that portion size is **2.5–5.0% of
carbohydrate variance** against **97.5% for which dish was eaten**, and that
a single 2D photo is "geometrically incapable of recovering portion volume."
The external literature converges on exactly the same conclusion from a
completely different angle (controlled weighed-ground-truth experiments
rather than this project's variance decomposition): **portion-from-a-single-
2D-photo is a real, well-studied, and still-unsolved problem with a stable
external error floor (roughly 2–3× worse than when mass is supplied), not a
problem this project's CV track is merely under-resourced to solve.** No
paper found in this review claims to have closed that gap without adding
information the photo itself doesn't contain (a depth sensor, a reference
object, an AR anchor, or — the solution every production app in §3 actually
ships — asking the user to confirm or correct the portion).

**Direct answer to the question asked:** yes, `CLAUDE.md`'s framing of the CV
track's objective as **"food recognition + macro estimation,"** stated as two
co-equal goals, is mis-weighted given what both this project's own BIG IDEAs
work and the external literature now show. Macro estimation is not an
independent capability to build — it is dish identification followed by a
lookup, with portion contributing a small, external-literature-confirmed
share of the error that cannot be meaningfully reduced by any modeling effort
applied to the photo alone. The CV track's actual, defensible objective is
**dish identification: accuracy and breadth** (see §§1–2); "macro estimation"
should be demoted from a modeling target to "whatever a sufficiently broad
lookup table returns for whatever dish was identified," with
`portion_reported` (already in the v1.1 contract) remaining the only portion
signal worth collecting, exactly as the existing PM decision in C5 states.

---

## 5. Public Indonesian food benchmark with carbohydrate/macro ground truth

**The official source.** TKPI — Tabel Komposisi Pangan Indonesia, Ministry
of Health, published at [panganku.org](https://www.panganku.org/en-EN/view).
Nutrient-composition data only; no images.

**Derived datasets found, all tracing back to TKPI/panganku.org:**

- **Kaggle, "Indonesian Food and Drink Nutrition Dataset,"** Anas Fikri
  Hanif, [kaggle.com/datasets/anasfikrihanif](https://www.kaggle.com/datasets/anasfikrihanif/indonesian-food-and-drink-nutrition-dataset):
  **1,346 rows**, each with calories/protein/fat/carbohydrate **and a scraped
  image URL** per food name. Also mirrored on Hugging Face as
  `eriko-syah/indonesian-food`. This is the largest Indonesian food+macro
  pairing found — but the images are **one representative web photo per food
  *name***, not photographs of real plated meals with portion variation, and
  the dataset's own description says only that image links were "added" after
  the nutrition data was cleaned — i.e. the same after-the-fact join risk this
  project's own `cv/data/tkpi_macro_lookup.csv` already carries, at 1,346 food
  names instead of 10, but still not a corpus of real meal photos with
  verified macros.
- **Hugging Face, `ULM-DS-Lab/food-composition-matrix`**
  ([huggingface.co/datasets/ULM-DS-Lab/food-composition-matrix](https://huggingface.co/datasets/ULM-DS-Lab/food-composition-matrix)):
  a cleaner structured TKPI extraction, **1,150 rows**, paired with a
  **7,790-row USDA SR Legacy** subset for cross-reference — no images at all,
  pure composition table. ULM-DS-Lab (Universitas Lambung Mangkurat)
  describes itself as working on "Indonesian food image understanding" and
  "postprandial glucose prediction" as active research areas — the only group
  found explicitly working at this project's exact intersection — but no
  public image+macro benchmark from them was found in this review.
- **Indriani et al., 2025,** *Bulletin of EEI*, DOI
  [10.11591/eei.v14i2.9189](https://doi.org/10.11591/eei.v14i2.9189): uses
  TKPI 2017 (1,146 items) purely as a tabular ML problem — predicting
  *missing* micronutrients from reported macros (random forest, 74.2% average
  accuracy on a binarized high/low task). Confirms TKPI's main published use
  in the literature is nutrient-table imputation, not image-to-macro
  modeling.
- Rasyidi et al. 2024's 160-class, 24,427-image set (§2) — confirmed, on
  reading the full paper, to carry **no nutrition annotation**.

**What this answers:** no public dataset was found — Indonesian or broader
Southeast Asian — that pairs **real meal photographs** (not one stock image
per food name) with **verified or weighed** macro ground truth, at any class
count larger than this project's own 10. The field-wide equivalent of what
this project needs (Nutrition5k) is a US-cafeteria dataset with weighed
ground truth; nothing analogous exists for Indonesian cuisine. **This is a
genuine, confirmed gap in the published literature, not a symptom of
insufficient search** — training or validating a next-generation Indonesian
macro-estimation model against anything better than the current 10-row TKPI
join would require building that dataset (weighed ground truth, Indonesian
dishes, real plated-meal photos), not finding one.

---

## Recommendations

Ordered by what would most move this project's actual Zone A / MAE numbers,
not by generic ML best practice.

1. **Flag for the user/PM: `CLAUDE.md`'s framing of the CV track's objective
   as "food recognition + macro estimation" (two co-equal goals) should be
   revisited.** §4 above shows this isn't just this project's C5/B7 finding —
   it is independently confirmed by Nutrition5k, CalorieVoL, and production
   practitioner reports that single-2D-photo portion estimation has a stable,
   external error floor no modeling effort on the photo alone can clear. The
   CV track's real, defensible objective is **dish identification — accuracy
   and breadth** — with macro output demoted to "whatever the lookup table
   returns." This changes what Sprint 3+ CV tasks should be funded, and is a
   PM/user decision, not made here.

2. **Replace the 10-class closed-set classifier with an open-vocabulary VLM
   for dish identification**, not with a bigger closed-set classifier. §1
   shows closed-set scaling tops out around 100–160 classes even with
   dedicated long-tailed architectures (THFOOD-100: 100 classes/96%; Rasyidi
   et al.: 160 classes/85.44%) — well short of the measured 455-dish breadth
   (CV11/B9) — and §2 confirms no Indonesian dataset exists at that scale with
   macro ground truth to even train one. §3 shows a general VLM already beats
   this project's own trained classifier by a wide margin (95.96% vs. 70.71%,
   `vlm_vs_mobilenet.md`, already measured in-repo) and is the production
   default for T2D-specific photo food logging (Dexcom, LogMeal/Evidenze,
   January AI). Keep the existing TKPI-lookup-from-class pattern this
   project's own report already validated as superior to asking the VLM for
   grams directly (0.05g MAE vs. 5.36g MAE, `vlm_vs_mobilenet.md`).

3. **Expand the TKPI lookup table, not the portion-estimation effort.** The
   binding constraint per C5/B9 is **class breadth** (10 rows vs. 455+
   dishes), not portion precision. §5 found no ready-made broader table with
   verified macros, so this is dataset-construction work: build out
   `cv/data/tkpi_macro_lookup.csv` toward the ~1,346-name scope of the TKPI-
   derived Kaggle set (§5) rather than toward better grams-from-pixels.

4. **Do not fund a single-2D-photo portion/volume-estimation research
   effort.** §4's literature review confirms this is not a cheap unsolved
   corner — it needs information (depth, reference object, or user input)
   that a plain photo doesn't contain, and every production system found in
   §3 solves it the same way this project's C5 fix already specifies:
   `portion_reported`, asked of the user. This recommendation directly
   supports the existing PM decision in C5/REVIEW_FINDINGS.md — the
   literature gives no reason to revisit it, and several reasons (Nutrition5k
   Table 3, CalorieVoL, soba.dev) to consider it closed.

5. **If cost/latency/offline-use rules out a VLM API in the final prototype**,
   the fallback from §1 is a CLIP-adapter few-shot approach (Wu et al. 2022)
   or a long-tailed-loss-engineered classifier (LOFI's equalization loss +
   normed head) trained on as broad an Indonesian image set as can be
   assembled — but this is a substantially harder and slower path than
   calling a VLM, and should only be chosen if the VLM option is ruled out for
   a concrete, stated reason (this project's own 503/rate-limit experience in
   `vlm_vs_mobilenet.md` is a real example of the kind of reason that would
   justify it).

6. **Before any of the above, resolve METRICS.md §7.2/§7.3** (the
   self-contradictory 0.8659/0.866/0.8801 test-accuracy figures and the
   70.71%-vs-86.6% sample-vs-full-split discrepancy) so that whatever
   replaces the current classifier is compared against a real baseline number,
   not three disagreeing ones.
