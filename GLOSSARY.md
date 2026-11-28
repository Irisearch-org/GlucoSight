# GlucoSight — shared vocabulary

Cross-track terms used by more than one sub-team. Track-local terms stay in
that track's own docs (`agents/{track}/CLAUDE.md`, `nlp/docs/`, etc.) unless
they start getting used across tracks, at which point they move here.

## Language

**Dish table**:
The single shared table of foods as eaten (whole dishes and single
ingredients), each with its macro composition per 100 g and a standard
serving weight. The only place a food's macros live. Built from TKPI
(Tabel Komposisi Pangan Indonesia) 2017.
_Avoid_: TKPI lookup, macro lookup — those name the data's source, not the
project's shared artifact.

**Matcher**:
The shared retrieval function that maps a food-name string (VLM-named from
a photo, or typed in manual entry) to its nearest rows in the dish table,
with a similarity score and its margin over the runner-up.
Similarity-based, not exact-key. Served: local hashed character 2–4-gram cosine (ADR 0006).
_Avoid_: classifier, lookup — "classifier" implies a fixed trained label
set, "lookup" implies an exact key; this is neither.

**Dish identification**:
The CV track's objective: naming which dish(es) appear in a meal photo, with
a confidence per dish. Deliberately excludes macro and portion estimation —
those come from the dish table and the `portion_reported` tap, not from CV.
_Avoid_: food recognition — the project's earlier framing, which bundled
macro estimation into the same objective and is being retired.

**Mixed plate**:
A meal photo containing more than one distinct dish (e.g. nasi padang: rice
plus several side dishes), as opposed to a single-dish photo. Dish
identification on a mixed plate returns a list, not one label; the dish
table and matcher are then applied per item and the macros summed.
_Avoid_: composite plate, multi-item meal — "mixed plate" is the term
already used in `agents/cv/CLAUDE.md`'s Task 5; this just canonicalizes it
instead of introducing a synonym.

**Baseline**:
One of the trivial predictors B0 (population constant), B1 (persistence)
or B2 (persistence plus cohort-mean excursion). A comparison row in every
results table; never what the app serves.
_Avoid_: model, mean model — a baseline learns nothing per meal.

**Served model**:
The trained or public machine-learning model a stage actually runs in the
app. A stage claims a contribution only when its served model beats the
baseline on held-out people.
_Avoid_: production model, real model.

**Manual entry**:
The user listing a meal's items one at a time (whole dishes or single
ingredients such as beef or potato, each with an amount) instead of a
photo. Required when dish identification falls below the confidence
threshold. Resolves through the same matcher and dish table as the photo
path.
_Avoid_: fallback lookup, safe mode — there is one dish table, not a
second lookup.

**Primary nav**:
The app's single navigation component, listing Home, Prediction, History
and Profile. Shown as a tab bar on phones and a left sidebar on desktop; it
is the same component, not two.
_Avoid_: bottom nav, navbar, sidebar, dashboard navbar — each names one
layout of the component, or is a generic web term.

**Home**:
The first screen after sign-in (Beranda): greeting, the call to start a
prediction, and the latest result.
_Avoid_: dashboard — the app has no separate dashboard screen.

**Landing**:
The public marketing page shown before sign-in. Not part of the signed-in
app and has no Primary nav.
_Avoid_: home page, homepage — "Home" is the signed-in screen.

**Error range**:
The measured held-out error shown beside every served estimate: the ±25
mg/dL around a finger-scan glucose, and the ± band around each forecast
horizon. It is what keeps a weak estimate honest; a number is never shown
without it where one exists.
_Avoid_: confidence, accuracy score — the app deliberately shows no
confidence percentage.
