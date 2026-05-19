# How race has been used as a variable in neuroimaging studies

A dashboard summarizing how race or ethnicity appears in the neuroimaging literature. Two complementary PubMed searches and two classifiers:

- **Focused search (headline)**: 1,238 papers retrieved with `MRI AND brain AND race` on 2026-05-14, full-text PDFs read by Claude Haiku 4.5 and classified into nine mutually exclusive categories. This is the more methodologically reliable estimate of how race is used in the brain-MRI literature, because the classifier reads the methods section rather than abstract only.
- **Broad sweep (context)**: 4,887 papers retrieved on 2026-05-14 with a union of seven neuroimaging terms (Neuroimaging, MRI, PET, EEG, MEG, NIRS, DTI) and six race-and-ethnicity terms (Racial Groups, Ethnicity, race, racial, ethnicity, ethnic), classified from abstracts by a regex heuristic and hand-validated on a 78-paper stratified sample.

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

In the focused brain-MRI × race corpus, 37.8% of papers use race substantively (categories 1–4). In the broader PubMed sweep, the regex catches only 11.4%. The gap is mostly driven by under-detection: Haiku reading the methods section finds nuisance-covariate adjustments that abstract-only classifiers miss. The true rate of nuisance-covariate usage in the broader PubMed literature is almost certainly closer to 30% than 12%.

## Search strategies

**Focused search (run 2026-05-14):**

```
("magnetic resonance imaging"[MeSH Terms] OR ("magnetic"[All Fields] AND "resonance"[All Fields] AND "imaging"[All Fields]) OR "magnetic resonance imaging"[All Fields] OR "mri"[All Fields]) AND ("brain"[MeSH Terms] OR "brain"[All Fields] OR "brains"[All Fields] OR "brain's"[All Fields]) AND ("racial groups"[MeSH Terms] OR ("racial"[All Fields] AND "groups"[All Fields]) OR "racial groups"[All Fields] OR "race"[All Fields])
```

**Broad search (run 2026-05-14):**

```
("Neuroimaging"[MeSH] OR "Magnetic Resonance Imaging"[MeSH] OR "Positron-Emission Tomography"[MeSH] OR "Electroencephalography"[MeSH] OR "Magnetoencephalography"[MeSH] OR "Spectroscopy, Near-Infrared"[MeSH] OR "Diffusion Tensor Imaging"[MeSH] OR neuroimaging[TIAB] OR "magnetic resonance imaging"[TIAB] OR fMRI[TIAB] OR "diffusion tensor"[TIAB] OR "positron emission tomography"[TIAB] OR electroencephalography[TIAB] OR magnetoencephalography[TIAB] OR "near-infrared spectroscopy"[TIAB]) AND ("Racial Groups"[MeSH] OR "Ethnicity"[MeSH] OR race[TIAB] OR racial[TIAB] OR ethnicity[TIAB] OR ethnic[TIAB]) AND ("2000"[PDAT] : "2026"[PDAT])
```

The focused search differs from the broad search in three respects: it restricts to MRI as the imaging modality, it requires the term *brain* anywhere in the record (excluding cardiac, breast, abdominal, and other non-brain MRI work), and it uses only race-related terminology, excluding papers that mention *ethnic* or *ethnicity* without using the word *race*. The 1,238 focused-search records are therefore a near-subset of the 4,887 broad-search records.

## Reproducing the analysis

All processing scripts are in `scripts/`:

- `pull_corpus.py` — paginated PubMed E-utilities pull, parses XML to JSONL.
- `classify.py` — regex classifier over abstracts (broad sweep).
- `classify_with_haiku.py` — Haiku classifier over local PDFs (focused search). Requires `ANTHROPIC_API_KEY` and the `anthropic` + `pypdf` packages.
- `classify_pubmed_with_haiku.py` — Haiku classifier over PubMed abstracts (companion run).

Aggregated outputs (used by the dashboard) are in `data/`:

- `dashboard_data.json` — broad-sweep regex results.
- `drive_dashboard_data.json` — focused-search Haiku results.

## Caveats

- The focused search restricts to brain MRI with explicit race terminology. The Haiku-derived proportions should not be generalized to ethnicity-only papers, to non-MRI neuroimaging (PET, EEG, MEG, NIRS, structural-only DTI work), or to MRI of non-brain anatomy (cardiac, breast, abdominal, musculoskeletal MRI).
- The broad sweep covers only PubMed-indexed work; Web of Science, Embase, PsycINFO, and preprint servers (bioRxiv/medRxiv) would broaden coverage. Full-text retrieval would also reveal nuisance-covariate use the abstract-only pass missed.
- Race operationalization is rarely stated in abstracts (538 unspecified vs. 20 self-report and 1 ancestry-based among substantive-use papers in the broad sweep). Most studies do not document how race was measured.
- No inter-rater reliability is reported. Each corpus was classified by a single procedure (regex or LLM), not by multiple human coders.

## License

MIT (see `LICENSE`).
