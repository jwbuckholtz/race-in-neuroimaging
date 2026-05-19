"""
Heuristic classifier for race-use in neuroimaging abstracts.

For each abstract:
  - primary_category: one of 9 (see CATEGORIES)
  - modalities: list of detected imaging modalities
  - study_type: from PubMed pub_types
  - population: clinical / healthy / mixed / unknown
  - race_operationalization: self_report / ancestry_genetic / unspecified
  - confidence: float in [0, 1] — heuristic score, NOT a statistical probability
"""
import json
import re
from pathlib import Path

OUT = Path(__file__).parent
JSONL_IN = OUT / "corpus.jsonl"
JSONL_OUT = OUT / "classified.jsonl"

CATEGORIES = [
    "primary_variable",          # 1
    "disparities_causal",        # 2a
    "disparities_descriptive",   # 2b
    "nuisance_covariate",        # 3
    "matching_variable",         # 4
    "sample_descriptor_only",    # 5
    "methodological_critique",   # 6
    "false_positive",            # 7
    "cannot_determine",          # 8
]

# ---- Regex toolkit ----------------------------------------------------------
# Word-boundary helpers — use \b on each side. Use re.IGNORECASE everywhere.
RACE_TERMS = (
    r"\b(race|racial|races|ethnicity|ethnic|ethnicities|"
    r"african[\s-]american|black|caucasian|white|european[\s-]american|"
    r"hispanic|latino|latina|latinx|latine|"
    r"asian[\s-]american|east[\s-]asian|south[\s-]asian|"
    r"native[\s-]american|american[\s-]indian|alaska[\s-]native|"
    r"pacific[\s-]islander|native[\s-]hawaiian|"
    r"BAME|BIPOC|multiethnic|multi-ethnic|multiracial|biracial|"
    r"minoritized|underrepresented[\s-]minorit(?:y|ies))\b"
)

# Stricter "definitive" race terms — categorical group labels
RACE_GROUP_LABELS = (
    r"\b(african[\s-]american|black|caucasian|white|european[\s-]american|"
    r"hispanic|latino|latina|latinx|asian[\s-]american|east[\s-]asian|"
    r"native[\s-]american|american[\s-]indian|pacific[\s-]islander|"
    r"BAME|BIPOC)\b"
)

# Covariate / adjustment phrasing
COVARIATE_PATTERNS = [
    re.compile(p, re.I) for p in [
        r"(adjust(ed|ing)?\s+for|controll(ed|ing)?\s+for|account(ed|ing)?\s+for|"
        r"covariat\w*\s+(includ|were|comprised|of|:)|including\s+as\s+covariat\w*|"
        r"after\s+adjust|regress(ed|ing)?\s+out|residualiz)"
        r"[^.]{0,180}" + RACE_TERMS,
        RACE_TERMS + r"[^.]{0,80}(as\s+(a\s+)?covariate|"
        r"as\s+nuisance|as\s+control\s+variable|"
        r"in\s+the\s+models?|in\s+regression|in\s+(linear|logistic|mixed)\s+models?)",
    ]
]

# Matching/stratification phrasing
MATCHING_PATTERNS = [
    re.compile(p, re.I) for p in [
        r"(match(ed|ing)?|stratif(ied|ying)?|balanced)\s+(for|on|by)[^.]{0,80}" + RACE_TERMS,
        RACE_TERMS + r"[^.]{0,40}match(ed|ing)?",
    ]
]

# Disparities — causal mechanism framing.
# Must explicitly model race-linked exposures (discrimination, adversity, SES, neighborhood)
# as causes/mediators of brain differences.
DISPARITIES_CAUSAL_PATTERNS = [
    re.compile(p, re.I) for p in [
        r"\b(structural|systemic|institutional)\s+racism",
        r"\b(racial(ized)?|race[\s-]based)\s+(discrimination|trauma|stress|adversity)",
        r"\bracial\s+discrimination\b",
        r"\b(perceived|experienced|everyday)\s+(racial\s+)?discrimination",
        r"\b(police\s+contact|police\s+brutality|police\s+violence)[^.]{0,200}(brain|neural|amygdala|cortex|MRI|fMRI)",
        r"\b(mediat\w+|account\w+\s+for|explain\w+)[^.]{0,200}(racial|race|ethnic)[^.]{0,200}(brain|neural|cortical|volume|connectivity|gray\s+matter|white\s+matter|amygdala|hippocamp|MRI|fMRI)",
        r"\b(adversity|SES|socioeconomic\s+status|poverty|discrimination|neighborhood\s+disadvantage|childhood\s+adversity)[^.]{0,80}(mediat\w+|account\w+\s+for|driv\w+|explain\w+)[^.]{0,300}(racial|race|ethnic)",
        r"\b(racial|race|ethnic)[^.]{0,80}difference[^.]{0,200}(mediat\w+|account\w+\s+for|explain\w+\s+by|driven\s+by|due\s+to|reflect\w+|attribut\w+\s+to)[^.]{0,150}(SES|socioeconomic|adversity|discrimination|neighborhood|opportunity|stress)",
        r"\b(false\s+appearance|apparent|so[\s-]called)[^.]{0,80}race[\s-]related",
        r"\bracialized\s+(exposures?|experience|society|adversity)",
        r"\b(maternal|childhood|early[\s-]life)\s+(SES|socioeconomic|adversity|opportunity|deprivation)[^.]{0,200}(racial|race|ethnic)[^.]{0,200}(brain|neural|MRI|fMRI|volume|cortical)",
    ]
]

# Disparities — descriptive framing
DISPARITIES_DESCRIPTIVE_PATTERNS = [
    re.compile(p, re.I) for p in [
        r"\b(racial|race|ethnic|health)\s+disparit\w+",
        r"\b(racial|race|ethnic)\s+inequit\w+",
        r"\bunderrepresent\w+",
        r"\bhealth\s+equity\b",
        r"\bminority\s+health\b",
        r"\bBAME\b",
        r"\bBIPOC\b",
    ]
]

# Methodological critique — must explicitly critique how race is USED IN RESEARCH METHODOLOGY,
# not describe racial bias as a neural/social phenomenon.
METHODOLOGICAL_CRITIQUE_PATTERNS = [
    re.compile(p, re.I) for p in [
        # Title-level signals (very strong indicators)
        r"\b(notion|concept|construct|use|usage|inclusion|reporting|operationalization)\s+of\s+(race|racial|ethnicity)",
        r"\b(rethinking|reconsider|critique|critical[\s-]analysis|problemati|reckoning|reframing|interrogating)\b[^.]{0,40}\b(race|racial|ethnic)",
        r"\b(diversifying|diversity|representation|under(\-|\s)?representation)\s+(in|of)\s+(neuroimag|fMRI|MRI|PET|EEG|brain|cognitive\s+neuroscience|neuroscience)",
        r"\b(call|need)\s+for\s+(the\s+)?(inclusion|reporting|diversity|consideration)\s+of\s+(race|racial|ethnic)",
        # Body-level: explicit critique of race-as-research-variable
        r"\brace\s+(as\s+a\s+)?(social|biological|genetic)\s+construct",
        r"\bracial\s+essentialism\b",
        r"\b(algorithmic|model|prediction|machine[\s-]learning|AI|deep[\s-]learning)\s+(bias|fairness|inequity)[^.]{0,200}(racial|race|ethnic)",
        r"\b(racial|ethnic)\s+(bias|inequity|disparit\w+)\s+in\s+(algorithm|machine[\s-]learning|model|prediction|atlas|template|normative\s+data)",
        r"\b(template|atlas|normative\s+data|reference\s+data|brain\s+model)s?\s+(developed|derived|trained|constructed|built)\s+(primarily|mainly|exclusively|only|chiefly|largely)\s+(from|on|in|using)[^.]{0,60}(White|Caucasian|European|non[\s-]Hispanic\s+white|NHW)",
        r"\b(generalizab|representativ)\w+[^.]{0,80}(racial|race|ethnic|minorit)",
        r"\b(racial|ethnic)\s+demographics?\s+(in|are\s+rarely|are\s+seldom|are\s+not)\s+report",
        r"\brarity\s+of\s+report",
        r"\b(homogene|monoculture|monoethnic)\w*\s+(sample|cohort|participant|dataset)",
        r"\bperform(s|ed|ance)?\s+(unequally|worse|poorly|inconsist\w+|unevenly)\s+(across|on|in|for)[^.]{0,40}(racial|ethnic|minorit)",
    ]
]

# Primary variable of interest — group comparison framing
PRIMARY_VARIABLE_TITLE_PATTERNS = [
    re.compile(p, re.I) for p in [
        r"\b" + RACE_GROUP_LABELS + r"[^.]{0,40}(versus|vs\.?|compared|comparison|differ)",
        r"(differ\w+|compar\w+|distinct|variation)\s+(by|across|between|among)[^.]{0,40}" + RACE_TERMS,
        r"(racial|race|ethnic)\s+(differ\w+|variation|effects?)",
        r"\bin\s+" + RACE_GROUP_LABELS + r"\s+(versus|vs\.?|and|compared\s+to)\s+" + RACE_GROUP_LABELS,
    ]
]

PRIMARY_VARIABLE_ABSTRACT_PATTERNS = [
    re.compile(p, re.I) for p in [
        r"(aim|objective|goal|purpose|investigat\w+|examin\w+|determin\w+)[^.]{0,150}(race|racial|ethnic)[^.]{0,150}(difference|effect|comparison|variation)",
        r"(race|racial|ethnic)[\s-]related\s+(difference|effect|variation)",
        r"(compar\w+|contrast\w+)[^.]{0,80}" + RACE_GROUP_LABELS + r"[^.]{0,80}" + RACE_GROUP_LABELS,
    ]
]

# False positive — incidental mention only
FALSE_POSITIVE_PATTERNS = [
    re.compile(p, re.I) for p in [
        r"\bpan[\s-]?ethnic\b",
        r"\bethnic(ity)?\s+of\s+the\s+(patient|case|proband)",
        r"\bcase\s+report\b",  # case reports rarely use race substantively
    ]
]

# Modalities — title/abstract regex
MODALITY_PATTERNS = {
    "fMRI": re.compile(r"\b(fMRI|functional\s+(MRI|magnetic\s+resonance|neuroimaging)|BOLD|resting[\s-]state)\b", re.I),
    "sMRI": re.compile(r"\b(structural\s+(MRI|magnetic\s+resonance|neuroimaging)|gray\s+matter\s+volume|cortical\s+thickness|VBM|voxel[\s-]based\s+morphometry|FreeSurfer)\b", re.I),
    "DTI": re.compile(r"\b(DTI|diffusion\s+tensor|diffusion[\s-]weighted|tractograph|fractional\s+anisotropy|white\s+matter\s+microstructure)\b", re.I),
    "PET": re.compile(r"\b(PET|positron[\s-]emission|FDG|amyloid\s+PET|tau\s+PET)\b"),
    "EEG": re.compile(r"\b(EEG|electroencephalogra|ERP|event[\s-]related\s+potential)\b", re.I),
    "MEG": re.compile(r"\b(MEG|magnetoencephalogra)\b", re.I),
    "NIRS": re.compile(r"\b(NIRS|fNIRS|near[\s-]infrared\s+spectroscop)\b", re.I),
}
MRI_GENERIC = re.compile(r"\b(MRI|magnetic\s+resonance\s+imaging)\b", re.I)

# Race operationalization
OPERATIONALIZATION_PATTERNS = {
    "self_report": re.compile(r"\b(self[\s-]reported?|self[\s-]identified?|participant[\s-]reported)[^.]{0,40}(race|racial|ethnic)|"
                              r"(race|racial|ethnic)[^.]{0,40}(self[\s-]reported?|self[\s-]identified?)", re.I),
    "ancestry_genetic": re.compile(r"\b(genetic\s+ancestry|continental\s+ancestry|ancestry[\s-]informative|admixture|principal\s+components?\s+of\s+ancestry)\b", re.I),
}

# Population
CLINICAL_KEYWORDS = re.compile(
    r"\b(patient|disease|disorder|syndrome|depression|schizophren|alzheim|"
    r"dement|parkinson|epilep|stroke|tbi|traumatic\s+brain|"
    r"anxiety|ptsd|autism|adhd|addiction|substance\s+use|"
    r"cancer|tumor|migraine|sclerosis|huntington|"
    r"clinical|diagnos|mci|cognitive\s+impairment|mood|psychiat)\b", re.I)
HEALTHY_KEYWORDS = re.compile(r"\b(healthy|typical(ly)?[\s-]developing|normative|community[\s-]based|"
                              r"general\s+population)\b", re.I)


# ---- Classifier -------------------------------------------------------------

def _has(text: str, patterns) -> bool:
    return any(p.search(text) for p in patterns)


def _race_terms_in(text: str) -> int:
    return len(re.findall(RACE_TERMS, text, re.I))


def classify(rec: dict) -> dict:
    title = rec.get("title") or ""
    abstract = rec.get("abstract") or ""
    full = f"{title}\n{abstract}"
    pub_types = rec.get("pub_types") or []
    mesh = rec.get("mesh") or []
    is_case_report = any("case reports" in pt.lower() for pt in pub_types)
    is_review = any("review" in pt.lower() for pt in pub_types) or any("meta-analysis" in pt.lower() for pt in pub_types)

    n_race = _race_terms_in(full)
    title_has_race = bool(re.search(RACE_TERMS, title, re.I))
    title_says_primary = _has(title, PRIMARY_VARIABLE_TITLE_PATTERNS)
    title_says_critique = _has(title, METHODOLOGICAL_CRITIQUE_PATTERNS)

    # ---- Special: no abstract --------------------------------------------------
    if not abstract or len(abstract.split()) < 30:
        # Try to classify from title alone
        if title_says_critique:
            return _result(rec, "methodological_critique", 0.7, full, pub_types)
        if title_says_primary:
            return _result(rec, "primary_variable", 0.6, full, pub_types)
        if is_case_report:
            return _result(rec, "false_positive", 0.75, full, pub_types)
        return _result(rec, "cannot_determine", 0.85, full, pub_types)

    # ---- 6. Methodological critique (strict; title or strong body signals) ---
    if title_says_critique or _has(abstract, METHODOLOGICAL_CRITIQUE_PATTERNS):
        return _result(rec, "methodological_critique", 0.8 if title_says_critique else 0.6, full, pub_types)

    # ---- Case reports default to false_positive unless title is race-focused --
    if is_case_report and not title_says_primary:
        return _result(rec, "false_positive", 0.75, full, pub_types)

    # ---- 2a. Disparities, causal ---------------------------------------------
    if _has(full, DISPARITIES_CAUSAL_PATTERNS):
        return _result(rec, "disparities_causal", 0.7, full, pub_types)

    # ---- 1. Primary variable of interest --------------------------------------
    # Title-level primary signal is strongest evidence
    if title_says_primary:
        # If body also has explicit disparities framing, route to 2b
        n_disp = sum(1 for p in DISPARITIES_DESCRIPTIVE_PATTERNS if p.search(full))
        if n_disp >= 1 and re.search(r"\bdisparit", full, re.I):
            return _result(rec, "disparities_descriptive", 0.7, full, pub_types)
        return _result(rec, "primary_variable", 0.75, full, pub_types)

    in_aim = _has(abstract, PRIMARY_VARIABLE_ABSTRACT_PATTERNS)
    if in_aim:
        if re.search(r"\bdisparit", full, re.I):
            return _result(rec, "disparities_descriptive", 0.65, full, pub_types)
        return _result(rec, "primary_variable", 0.6, full, pub_types)

    # ---- 2b. Disparities, descriptive (without group-comparison aim) --------
    # Require BOTH disparity language AND multiple race terms.
    n_disp = sum(1 for p in DISPARITIES_DESCRIPTIVE_PATTERNS if p.search(full))
    has_strong_disp = bool(re.search(r"\bdisparit", full, re.I))
    if (n_disp >= 2 or has_strong_disp) and n_race >= 3:
        return _result(rec, "disparities_descriptive", 0.55, full, pub_types)

    # ---- 4. Matching variable ------------------------------------------------
    if _has(full, MATCHING_PATTERNS):
        return _result(rec, "matching_variable", 0.65, full, pub_types)

    # ---- 3. Nuisance covariate -----------------------------------------------
    if _has(full, COVARIATE_PATTERNS):
        return _result(rec, "nuisance_covariate", 0.7, full, pub_types)

    # ---- 7. False positive: incidental mention only -------------------------
    if _has(full, FALSE_POSITIVE_PATTERNS) and n_race <= 2:
        return _result(rec, "false_positive", 0.55, full, pub_types)

    # ---- 5. Sample descriptor only (default for low-mention abstracts) ------
    if n_race >= 1:
        return _result(rec, "sample_descriptor_only", 0.5, full, pub_types)

    # No race terms detected at all — false positive
    return _result(rec, "false_positive", 0.7, full, pub_types)


def _detect_modalities(text: str) -> list[str]:
    found = []
    for name, p in MODALITY_PATTERNS.items():
        if p.search(text):
            found.append(name)
    # If MRI is mentioned generically but no specific subtype, tag as "MRI (unspec)"
    if not any(m in found for m in ("fMRI", "sMRI", "DTI")) and MRI_GENERIC.search(text):
        found.append("MRI_unspec")
    return found


def _detect_study_type(pub_types: list[str]) -> str:
    pts_lower = {pt.lower() for pt in pub_types}
    if any("review" in pt for pt in pts_lower) and "meta-analysis" in pts_lower:
        return "meta_analysis"
    if "meta-analysis" in pts_lower:
        return "meta_analysis"
    if any("review" in pt for pt in pts_lower):
        return "review"
    if any("case reports" in pt for pt in pts_lower):
        return "case_report"
    if any("editorial" in pt or "comment" in pt or "letter" in pt for pt in pts_lower):
        return "commentary"
    return "empirical"


def _detect_population(text: str) -> str:
    has_clin = bool(CLINICAL_KEYWORDS.search(text))
    has_healthy = bool(HEALTHY_KEYWORDS.search(text))
    if has_clin and has_healthy:
        return "mixed"
    if has_clin:
        return "clinical"
    if has_healthy:
        return "healthy"
    return "unknown"


def _detect_operationalization(text: str) -> str:
    if OPERATIONALIZATION_PATTERNS["ancestry_genetic"].search(text):
        return "ancestry_genetic"
    if OPERATIONALIZATION_PATTERNS["self_report"].search(text):
        return "self_report"
    return "unspecified"


def _result(rec: dict, category: str, conf: float, full: str, pub_types: list[str]) -> dict:
    return {
        "pmid": rec["pmid"],
        "year": rec["year"][:4] if rec.get("year") else "",
        "journal": rec.get("journal", ""),
        "title": rec.get("title", ""),
        "primary_category": category,
        "confidence": round(conf, 2),
        "modalities": _detect_modalities(full),
        "study_type": _detect_study_type(pub_types),
        "population": _detect_population(full),
        "race_operationalization": _detect_operationalization(full),
        "n_race_terms": _race_terms_in(full),
    }


def main():
    n_in = n_out = 0
    cat_counts = {c: 0 for c in CATEGORIES}
    with JSONL_IN.open() as fin, JSONL_OUT.open("w") as fout:
        for line in fin:
            rec = json.loads(line)
            n_in += 1
            res = classify(rec)
            cat_counts[res["primary_category"]] += 1
            fout.write(json.dumps(res, ensure_ascii=False) + "\n")
            n_out += 1
    print(f"Classified {n_out}/{n_in} records.\n")
    print("Primary category distribution:")
    for c in CATEGORIES:
        print(f"  {cat_counts[c]:5d}  {c}")


if __name__ == "__main__":
    main()
