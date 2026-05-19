# How race has been used as a variable in neuroimaging studies

A dashboard summarizing how race or ethnicity appears in the neuroimaging literature. Two corpora, two classifiers:

- **Drive reading library (headline)**: 1,238 hand-curated PDFs read full-text by Claude Haiku 4.5 and classified into nine mutually exclusive categories.
- **PubMed broad sweep (context)**: 4,887 papers (2000 – May 2026) retrieved via NCBI E-utilities, classified from abstracts by a regex heuristic and hand-validated on a 78-paper stratified sample.

Live dashboard: see `index.html`.

## Categories

Each paper is assigned to one of:

1. **Primary variable** — race or ethnic group is the variable being examined (own-race face perception, Black-White medical comparisons, ethnic variation in disease phenotypes).
2. **Disparities, causal** — race-linked exposures (discrimination, structural racism, neighborhood disadvantage, SES, childhood adversity) modeled as mediators of brain differences.
3. **Disparities, descriptive** — group differences reported with health-equity framing but without a causal mechanism.
4. **Methodological critique** — the paper interrogates how race is used in neuroimaging research itself (atlas bias, AI fairness, sample representation).
5. **Nuisance covariate** — race entered as an adjustment variable; no race effect interpreted.
6. **Matching variable** — race used to balance comparison groups, not modeled.
7. **Sample descriptor only** — race appears only in the demographics table.
8. **False positive** — incidental mention (pan-ethnic disorder, case-report patient descriptors).
9. **Cannot determine** — text insufficient.

## Headline finding

In the curated Drive corpus, 37.8% of papers use race substantively (categories 1–4); in the broader PubMed sweep, the regex catches only 11.4%. The gap is mostly driven by under-detection: Haiku reading the methods section finds nuisance-covariate adjustments that abstract-only classifiers miss. The true rate of nuisance-covariate usage in the broader PubMed literature is almost certainly closer to 30% than 12%.

## Reproducing the analysis

All processing scripts are in `scripts/`:

- `pull_corpus.py` — paginated PubMed E-utilities pull, parses XML to JSONL.
- `classify.py` — regex classifier over abstracts (PubMed sweep).
- `classify_with_haiku.py` — Haiku classifier over local PDFs (Drive corpus). Requires `ANTHROPIC_API_KEY` and the `anthropic` + `pypdf` packages.
- `classify_pubmed_with_haiku.py` — Haiku classifier over PubMed abstracts (not yet run in the published version).

Aggregated outputs (used by the dashboard) are in `data/`:

- `dashboard_data.json` — PubMed regex results.
- `drive_dashboard_data.json` — Drive Haiku results.

## Caveats

- The Drive library is a hand-curated list, and likely a more accurate picture of the neuroimaging literature (as opposed to, e.g. cardiac/breast MRI)
- The PubMed sweep covers only PubMed-indexed work. Web of Science, Embase, PsycINFO, and preprint servers would broaden coverage; full-text retrieval would also reveal nuisance-covariate use the abstract-only pass missed.
- "Race operationalization" is rarely stated in abstracts (538 unspecified vs. 20 self-report and 1 ancestry-based among the substantive-use PubMed papers). Most studies do not document how race was measured.

## License

MIT (see `LICENSE`).
