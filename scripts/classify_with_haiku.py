"""
Classify PDFs in a local folder by how race/ethnicity is used.

Usage:
  export ANTHROPIC_API_KEY=sk-ant-...
  pip install anthropic pypdf
  python3 classify_with_haiku.py /path/to/Race_Coef [--workers 8] [--limit N]

Output:
  drive_classified.jsonl  one row per PDF, written incrementally so re-runs resume
  drive_classified.errors.log

Cost expectation: ~$0.005 per PDF (about $5 for 1,000 PDFs) using Haiku 4.5.
Runtime: ~3-5 ms per page of extraction + 1-3 s per Haiku call; with 8 workers, ~10 min for 1,000 PDFs.
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
    sys.exit("Install: pip install anthropic pypdf")
try:
    from pypdf import PdfReader
except ImportError:
    sys.exit("Install: pip install anthropic pypdf")


# ---- Config ----------------------------------------------------------------
MODEL = "claude-haiku-4-5-20251001"
MAX_PDF_CHARS = 18000  # ~4500 tokens of input text per paper
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

SYSTEM_PROMPT = """You are a research assistant classifying how race or ethnicity is used in a neuroimaging paper.

You will be given the first portion of a paper (title + abstract + introduction + part of methods). Classify the primary use of race or ethnicity into exactly one of these mutually exclusive categories:

1. primary_variable — the paper's aim is to compare racial or ethnic groups on brain measures, or to interpret a race-related effect (e.g., own-race face perception, ethnic differences in disease phenotypes, Black vs White comparisons).
2. disparities_causal — the paper models race-linked exposures (racial discrimination, structural racism, neighborhood disadvantage, SES, childhood adversity, racialized stress) as causes or mediators of brain differences.
3. disparities_descriptive — the paper documents group differences with explicit health-equity or disparities framing, but does not model a specific causal mechanism.
4. nuisance_covariate — race is entered as an adjustment variable in statistical models; no race effect is interpreted.
5. matching_variable — race is used to balance comparison groups (matched by age, sex, race) but not modeled.
6. sample_descriptor_only — race appears only in the demographics table or one-sentence sample description, with no analytic role.
7. methodological_critique — the paper critiques HOW race is used in neuroimaging research (atlas/template bias, AI fairness, race-as-construct, sample representation, normative data critique). NOT for papers studying racial bias as a *brain phenomenon* — those are primary_variable.
8. false_positive — race/ethnicity terminology appears only incidentally (e.g., "pan-ethnic disorder", case-report patient descriptor) with no substantive role.
9. cannot_determine — the provided text is too short or unclear to classify.

Also extract these tags:
- modalities: subset of [fMRI, sMRI, DTI, PET, EEG, MEG, NIRS, MRI_unspec] based on what the methods use. MRI_unspec only if MRI is mentioned but no subtype is specified.
- study_type: one of [empirical, review, meta_analysis, case_report, commentary, methods_tools]
- population: one of [clinical, healthy, mixed, unknown]
- race_operationalization: one of [self_report, ancestry_genetic, unspecified]
- year: 4-digit publication year if you can extract it, else null
- evidence: one short quote (under 30 words) from the text supporting your primary_category choice
- title: the paper's title if visible in the text, else null

Respond with ONLY a valid JSON object with these exact keys: primary_category, modalities, study_type, population, race_operationalization, year, evidence, title, confidence (0.0-1.0), reasoning (under 30 words). No prose around the JSON."""


# ---- PDF extraction --------------------------------------------------------

def extract_pdf_text(path: Path, max_chars: int = MAX_PDF_CHARS) -> str:
    """Extract title + abstract + intro + methods region (~first 5 pages)."""
    try:
        reader = PdfReader(str(path))
        parts = []
        n_chars = 0
        for i, page in enumerate(reader.pages[:7]):  # first 7 pages caps most papers
            txt = page.extract_text() or ""
            parts.append(txt)
            n_chars += len(txt)
            if n_chars >= max_chars:
                break
        text = "\n".join(parts)
        # collapse whitespace
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text[:max_chars]
    except Exception as e:
        raise RuntimeError(f"PDF extract failed: {e}")


# ---- Classify --------------------------------------------------------------

def classify_one(client: Anthropic, path: Path) -> dict:
    text = extract_pdf_text(path)
    if len(text.strip()) < 200:
        return {
            "file": path.name,
            "primary_category": "cannot_determine",
            "modalities": [], "study_type": "unknown", "population": "unknown",
            "race_operationalization": "unspecified", "year": None,
            "evidence": "", "title": None,
            "confidence": 0.95, "reasoning": "PDF yielded too little extractable text",
        }
    # Call Haiku with retries
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            msg = client.messages.create(
                model=MODEL,
                max_tokens=512,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": f"<paper>\n{text}\n</paper>\n\nReturn the JSON object now."}],
            )
            raw = "".join(b.text for b in msg.content if b.type == "text")
            # Strip code fences if model added them
            raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
            raw = re.sub(r"\s*```$", "", raw)
            data = json.loads(raw)
            data["file"] = path.name
            data["_input_tokens"] = msg.usage.input_tokens
            data["_output_tokens"] = msg.usage.output_tokens
            # Validate primary_category
            if data.get("primary_category") not in CATEGORIES:
                data["_warning"] = f"Unknown category: {data.get('primary_category')!r}; coerced to cannot_determine"
                data["primary_category"] = "cannot_determine"
            return data
        except Exception as e:
            last_err = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Haiku failed after {MAX_RETRIES} retries: {last_err}")


# ---- Main ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", help="Path to local folder containing PDFs (e.g., Race_Coef synced from Drive)")
    parser.add_argument("--workers", type=int, default=8, help="Parallel workers (default 8)")
    parser.add_argument("--limit", type=int, default=0, help="Process at most N PDFs (0 = all)")
    parser.add_argument("--output", default="drive_classified.jsonl", help="Output JSONL path")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit("Set ANTHROPIC_API_KEY env var before running.")

    folder = Path(args.folder).expanduser().resolve()
    if not folder.is_dir():
        sys.exit(f"Not a directory: {folder}")
    pdfs = sorted(folder.glob("*.pdf"))
    if args.limit:
        pdfs = pdfs[: args.limit]
    print(f"Found {len(pdfs)} PDFs in {folder}", flush=True)

    # Resume: skip files already in output JSONL
    out_path = Path(args.output)
    done = set()
    if out_path.exists():
        for line in out_path.open():
            try:
                done.add(json.loads(line)["file"])
            except Exception:
                pass
        print(f"Resuming: {len(done)} files already classified", flush=True)
    todo = [p for p in pdfs if p.name not in done]
    print(f"To classify: {len(todo)}", flush=True)

    client = Anthropic(api_key=api_key)
    err_log = Path("drive_classified.errors.log").open("a")
    total_in = total_out = 0
    n_done = 0
    start = time.time()
    with out_path.open("a") as fout, cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(classify_one, client, p): p for p in todo}
        for fut in cf.as_completed(futures):
            p = futures[fut]
            try:
                result = fut.result()
                fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                fout.flush()
                total_in += result.get("_input_tokens", 0)
                total_out += result.get("_output_tokens", 0)
                n_done += 1
                if n_done % 20 == 0 or n_done == len(todo):
                    elapsed = time.time() - start
                    rate = n_done / elapsed if elapsed > 0 else 0
                    cost = total_in * 1.0e-6 + total_out * 5.0e-6  # Haiku 4.5: $1/$5 per M
                    print(f"  [{n_done}/{len(todo)}] rate={rate:.1f}/s tokens={total_in}+{total_out} cost~=${cost:.2f}", flush=True)
            except Exception as e:
                err_log.write(f"{p.name}\t{type(e).__name__}: {e}\n")
                err_log.write(traceback.format_exc() + "\n")
                err_log.flush()
                fout.write(json.dumps({"file": p.name, "primary_category": "cannot_determine", "error": str(e)[:200]}) + "\n")
                fout.flush()
                n_done += 1

    err_log.close()
    elapsed = time.time() - start
    cost = total_in * 1.0e-6 + total_out * 5.0e-6
    print(f"\nDone. {n_done} files in {elapsed:.0f}s. Tokens: {total_in} in / {total_out} out. Cost ~${cost:.2f}.")
    print(f"Output: {out_path.resolve()}")


if __name__ == "__main__":
    main()
