"""
Re-classify the PubMed corpus (corpus.jsonl) via Haiku, using title + abstract.

Usage:
  export ANTHROPIC_API_KEY=sk-ant-...
  pip install anthropic
  python3 classify_pubmed_with_haiku.py [--workers 12] [--limit N] [--input corpus.jsonl]

Cost: ~$0.0005 per record (~$2.50 for 4,887). Wall time: 5-10 min with 12 workers.

Output:
  pubmed_haiku_classified.jsonl  one row per record, resumable across re-runs
  pubmed_haiku_classified.errors.log
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path

try:
    from anthropic import Anthropic
except ImportError:
    sys.exit("Install: pip install anthropic")


MODEL = "claude-haiku-4-5-20251001"
MAX_RETRIES = 4

CATEGORIES = [
    "primary_variable",
    "disparities_causal",
    "disparities_descriptive",
    "nuisance_covariate",
    "matching_variable",
    "sample_descriptor_only",
    "methodological_critique",
    "false_positive",
    "cannot_determine",
]

SYSTEM_PROMPT = """You are a research assistant classifying how race or ethnicity is used in a neuroimaging paper based on its title and abstract.

Classify the primary use of race or ethnicity into exactly one of these mutually exclusive categories:

1. primary_variable — the paper's aim is to compare racial or ethnic groups on brain measures, or to interpret a race-related effect (e.g., own-race face perception, ethnic differences in disease phenotypes, Black vs White comparisons).
2. disparities_causal — the paper models race-linked exposures (racial discrimination, structural racism, neighborhood disadvantage, SES, childhood adversity, racialized stress) as causes or mediators of brain differences.
3. disparities_descriptive — the paper documents group differences with explicit health-equity or disparities framing, but does not model a specific causal mechanism.
4. nuisance_covariate — race is entered as an adjustment variable in statistical models; no race effect is interpreted. (Note: abstracts often say "adjusted for race" without elaborating — that's still nuisance_covariate.)
5. matching_variable — race is used to balance comparison groups (matched by age, sex, race) but not modeled.
6. sample_descriptor_only — race appears only in the demographics description of the sample with no analytic role.
7. methodological_critique — the paper critiques HOW race is used in neuroimaging research (atlas/template bias, AI fairness, race-as-construct, sample representation, normative data critique). NOT for papers studying racial bias as a *brain phenomenon* — those are primary_variable.
8. false_positive — race/ethnicity terminology appears only incidentally (e.g., "pan-ethnic disorder", case-report patient descriptor) with no substantive role.
9. cannot_determine — the abstract is too short, missing, or unclear to classify.

Important: when classifying from abstract-only, many papers will be ambiguous between sample_descriptor_only and nuisance_covariate. If the abstract mentions race in the methods/analysis (even briefly), prefer nuisance_covariate. Reserve sample_descriptor_only for cases where race appears solely in a demographics sentence.

Also extract these tags from the abstract:
- modalities: subset of [fMRI, sMRI, DTI, PET, EEG, MEG, NIRS, MRI_unspec]
- study_type: one of [empirical, review, meta_analysis, case_report, commentary, methods_tools]
- population: one of [clinical, healthy, mixed, unknown]
- race_operationalization: one of [self_report, ancestry_genetic, unspecified]
- evidence: one short quote (under 30 words) from the abstract supporting your primary_category choice
- confidence: 0.0 to 1.0
- reasoning: under 30 words

Respond with ONLY a valid JSON object with these exact keys: primary_category, modalities, study_type, population, race_operationalization, evidence, confidence, reasoning. No prose around the JSON."""


def classify_one(client: Anthropic, rec: dict) -> dict:
    title = (rec.get("title") or "").strip()
    abstract = (rec.get("abstract") or "").strip()
    if not abstract or len(abstract.split()) < 25:
        return {
            "pmid": rec["pmid"],
            "year": (rec.get("year") or "")[:4],
            "journal": rec.get("journal", ""),
            "title": title,
            "primary_category": "cannot_determine",
            "modalities": [], "study_type": "unknown", "population": "unknown",
            "race_operationalization": "unspecified",
            "evidence": "", "confidence": 0.95,
            "reasoning": "Abstract missing or too short",
        }
    text = f"TITLE: {title}\n\nABSTRACT: {abstract}"
    if len(text) > 6000:
        text = text[:6000]
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            msg = client.messages.create(
                model=MODEL,
                max_tokens=512,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": text + "\n\nReturn the JSON object now."}],
            )
            raw = "".join(b.text for b in msg.content if b.type == "text").strip()
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            data = json.loads(raw)
            data["pmid"] = rec["pmid"]
            data["year"] = (rec.get("year") or "")[:4]
            data["journal"] = rec.get("journal", "")
            data["title"] = title
            data["_input_tokens"] = msg.usage.input_tokens
            data["_output_tokens"] = msg.usage.output_tokens
            if data.get("primary_category") not in CATEGORIES:
                data["_warning"] = f"Unknown category: {data.get('primary_category')!r}"
                data["primary_category"] = "cannot_determine"
            return data
        except Exception as e:
            last_err = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Haiku failed after {MAX_RETRIES} retries: {last_err}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="corpus.jsonl", help="Path to PubMed corpus JSONL (default ./corpus.jsonl)")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--output", default="pubmed_haiku_classified.jsonl")
    args = ap.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("Set ANTHROPIC_API_KEY env var before running.")

    in_path = Path(args.input)
    if not in_path.exists():
        sys.exit(f"Not found: {in_path}\nRun this from the same folder as the PubMed run (where corpus.jsonl lives).")
    recs = [json.loads(l) for l in in_path.open()]
    if args.limit:
        recs = recs[: args.limit]
    print(f"Records to classify: {len(recs)}", flush=True)

    # Resume
    out_path = Path(args.output)
    done = set()
    if out_path.exists():
        for line in out_path.open():
            try:
                done.add(json.loads(line)["pmid"])
            except Exception:
                pass
        print(f"Resuming: {len(done)} already done", flush=True)
    todo = [r for r in recs if r["pmid"] not in done]
    print(f"To classify: {len(todo)}", flush=True)

    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    err_log = Path("pubmed_haiku_classified.errors.log").open("a")
    total_in = total_out = n_done = 0
    start = time.time()
    with out_path.open("a") as fout, cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(classify_one, client, r): r for r in todo}
        for fut in cf.as_completed(futures):
            r = futures[fut]
            try:
                res = fut.result()
                fout.write(json.dumps(res, ensure_ascii=False) + "\n")
                fout.flush()
                total_in += res.get("_input_tokens", 0)
                total_out += res.get("_output_tokens", 0)
                n_done += 1
                if n_done % 100 == 0 or n_done == len(todo):
                    elapsed = time.time() - start
                    rate = n_done / elapsed if elapsed > 0 else 0
                    cost = total_in * 1.0e-6 + total_out * 5.0e-6  # Haiku 4.5: $1/$5 per M
                    print(f"  [{n_done}/{len(todo)}] rate={rate:.1f}/s tokens={total_in}+{total_out} cost~=${cost:.2f}", flush=True)
            except Exception as e:
                err_log.write(f"{r['pmid']}\t{type(e).__name__}: {e}\n")
                err_log.write(traceback.format_exc() + "\n")
                err_log.flush()
                fout.write(json.dumps({"pmid": r["pmid"], "primary_category": "cannot_determine", "error": str(e)[:200]}) + "\n")
                fout.flush()
                n_done += 1
    err_log.close()
    elapsed = time.time() - start
    cost = total_in * 1.0e-6 + total_out * 5.0e-6
    print(f"\nDone. {n_done} records in {elapsed:.0f}s. Tokens: {total_in} in / {total_out} out. Cost ~${cost:.2f}.")
    print(f"Output: {out_path.resolve()}")


if __name__ == "__main__":
    main()
