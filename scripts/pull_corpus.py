"""
Pull PubMed corpus: neuroimaging x race/ethnicity, 2000-2026.

Pipeline:
  1. ESearch with usehistory=y to register the full result set
  2. Page through EFetch (rettype=abstract, retmode=xml) in batches of 200
  3. Parse XML -> jsonl: {pmid, year, journal, title, abstract, mesh, pub_types, authors}
"""
import json
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

OUT = Path(__file__).parent
RAW_XML_DIR = OUT / "raw_xml"
RAW_XML_DIR.mkdir(exist_ok=True)
JSONL = OUT / "corpus.jsonl"

QUERY = (
    '('
    '"Neuroimaging"[MeSH] OR "Magnetic Resonance Imaging"[MeSH] OR '
    '"Positron-Emission Tomography"[MeSH] OR "Electroencephalography"[MeSH] OR '
    '"Magnetoencephalography"[MeSH] OR "Spectroscopy, Near-Infrared"[MeSH] OR '
    '"Diffusion Tensor Imaging"[MeSH] OR '
    'neuroimaging[TIAB] OR "magnetic resonance imaging"[TIAB] OR fMRI[TIAB] OR '
    '"diffusion tensor"[TIAB] OR "positron emission tomography"[TIAB] OR '
    'electroencephalography[TIAB] OR magnetoencephalography[TIAB] OR '
    '"near-infrared spectroscopy"[TIAB]'
    ') AND ('
    '"Racial Groups"[MeSH] OR "Ethnicity"[MeSH] OR race[TIAB] OR racial[TIAB] OR '
    'ethnicity[TIAB] OR ethnic[TIAB]'
    ') AND ("2000"[PDAT] : "2026"[PDAT])'
)

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def fetch(url: str, retries: int = 5) -> bytes:
    for i in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return r.read()
        except Exception as e:
            if i == retries - 1:
                raise
            time.sleep(2 ** i)
    raise RuntimeError("unreachable")


def esearch_history() -> tuple[int, str, str]:
    params = {
        "db": "pubmed",
        "term": QUERY,
        "usehistory": "y",
        "retmode": "json",
        "retmax": "0",
    }
    url = f"{EUTILS}/esearch.fcgi?{urllib.parse.urlencode(params)}"
    j = json.loads(fetch(url))
    er = j["esearchresult"]
    return int(er["count"]), er["webenv"], er["querykey"]


def efetch_batch(webenv: str, qkey: str, retstart: int, retmax: int = 200) -> bytes:
    params = {
        "db": "pubmed",
        "query_key": qkey,
        "WebEnv": webenv,
        "retstart": str(retstart),
        "retmax": str(retmax),
        "retmode": "xml",
        "rettype": "abstract",
    }
    url = f"{EUTILS}/efetch.fcgi?{urllib.parse.urlencode(params)}"
    return fetch(url)


def text(el) -> str:
    if el is None:
        return ""
    # Concatenate all text, including from nested <i>, <sup>, etc.
    return "".join(el.itertext()).strip()


def parse_article(article) -> dict | None:
    medline = article.find("MedlineCitation")
    if medline is None:
        return None
    pmid_el = medline.find("PMID")
    if pmid_el is None:
        return None
    pmid = pmid_el.text

    art = medline.find("Article")
    title = text(art.find("ArticleTitle")) if art is not None else ""

    # Abstract — concatenate all <AbstractText> sections
    abs_parts = []
    if art is not None:
        for at in art.findall(".//AbstractText"):
            label = at.get("Label")
            txt = text(at)
            if label:
                abs_parts.append(f"{label}: {txt}")
            else:
                abs_parts.append(txt)
    abstract = " ".join(p for p in abs_parts if p)

    # Year (best effort)
    year = ""
    pubdate = art.find(".//Journal/JournalIssue/PubDate") if art is not None else None
    if pubdate is not None:
        y = pubdate.find("Year")
        if y is not None and y.text:
            year = y.text
        else:
            md = pubdate.find("MedlineDate")
            if md is not None and md.text:
                year = md.text[:4]
    if not year:
        # Fallback: PubmedData/History
        for pdate in article.findall(".//PubMedPubDate"):
            y = pdate.find("Year")
            if y is not None and y.text:
                year = y.text
                break

    # Journal
    journal = ""
    if art is not None:
        j = art.find(".//Journal/Title")
        journal = text(j) if j is not None else ""

    # MeSH
    mesh = []
    mh_list = medline.find("MeshHeadingList")
    if mh_list is not None:
        for mh in mh_list.findall("MeshHeading"):
            d = mh.find("DescriptorName")
            if d is not None and d.text:
                mesh.append(d.text)

    # Publication types
    pub_types = []
    if art is not None:
        ptl = art.find("PublicationTypeList")
        if ptl is not None:
            for pt in ptl.findall("PublicationType"):
                if pt.text:
                    pub_types.append(pt.text)

    # Authors (first 3 + last for citation)
    authors = []
    if art is not None:
        al = art.find("AuthorList")
        if al is not None:
            for a in al.findall("Author"):
                ln = a.find("LastName")
                fn = a.find("ForeName") or a.find("Initials")
                if ln is not None and ln.text:
                    name = ln.text
                    if fn is not None and fn.text:
                        name = f"{ln.text} {fn.text}"
                    authors.append(name)

    return {
        "pmid": pmid,
        "year": year,
        "journal": journal,
        "title": title,
        "abstract": abstract,
        "mesh": mesh,
        "pub_types": pub_types,
        "authors": authors,
    }


def main():
    count, webenv, qkey = esearch_history()
    print(f"Corpus size: {count}", flush=True)

    batch_size = 200
    written = 0
    with JSONL.open("w") as fout:
        for start in range(0, count, batch_size):
            xml_path = RAW_XML_DIR / f"batch_{start:06d}.xml"
            if xml_path.exists():
                data = xml_path.read_bytes()
            else:
                data = efetch_batch(webenv, qkey, start, batch_size)
                xml_path.write_bytes(data)
                time.sleep(0.4)  # respect NCBI rate limits
            root = ET.fromstring(data)
            for article in root.findall("PubmedArticle"):
                rec = parse_article(article)
                if rec is None:
                    continue
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1
            print(f"  fetched {start + batch_size}/{count}; wrote {written}", flush=True)

    print(f"Done. Wrote {written} records to {JSONL}", flush=True)


if __name__ == "__main__":
    main()
