"""Build the retrieval corpus and a FAISS index over it.

The corpus is deliberately small and entirely local:

  * three NCHS methodology PDFs (public domain), which describe how the linked
    mortality file is constructed and what its limitations are, and
  * the model card this project generates, so the copilot can cite the model's
    own reported metrics and limitations rather than inventing them.

Chunks keep their source file and page number, because a note that cites
"[source: file, page 7]" is only worth anything if the page number is real.
"""

from __future__ import annotations

import json
import re
import ssl
import sys
import urllib.request
from dataclasses import dataclass, asdict
from pathlib import Path

import certifi

ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = ROOT / "corpus"
INDEX_DIR = ROOT / "index"
MODEL_CARD = ROOT / "artifacts" / "model_card.json"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ~600 tokens with ~100 overlap, counted in words (roughly 0.75 words/token).
CHUNK_WORDS = 450
OVERLAP_WORDS = 75

PDFS = {
    "linked-mortality-file-description.pdf":
        "https://www.cdc.gov/nchs/data/datalinkage/public-use-linked-mortality-file-description.pdf",
    "linked-mortality-data-dictionary.pdf":
        "https://www.cdc.gov/nchs/data/datalinkage/public-use-linked-mortality-files-data-dictionary.pdf",
    "lmf-methodology-analytic-considerations.pdf":
        "https://www.cdc.gov/nchs/media/pdfs/2026/01/lmf2022-methodology-analytic-considerations.pdf",
}


@dataclass
class Chunk:
    chunk_id: int
    source: str
    page: int | None
    text: str

    def citation(self) -> str:
        if self.page is None:
            return f"[source: {self.source}]"
        return f"[source: {self.source}, page {self.page}]"


def fetch_pdfs() -> None:
    # Python builds from python.org do not use the macOS system trust store, so
    # urllib has no root certificates and every HTTPS fetch fails with
    # CERTIFICATE_VERIFY_FAILED. Point it at certifi's bundle explicitly rather
    # than depending on however the local interpreter was installed.
    context = ssl.create_default_context(cafile=certifi.where())

    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in PDFS.items():
        dest = CORPUS_DIR / name
        if dest.exists() and dest.stat().st_size > 0:
            print(f"  cached   {name}")
            continue
        print(f"  download {name}")
        with urllib.request.urlopen(url, context=context) as response:
            payload = response.read()
        if not payload.startswith(b"%PDF-"):
            raise SystemExit(f"{url} did not return a PDF")
        dest.write_bytes(payload)


def clean(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def window(text: str, source: str, page: int | None, start_id: int) -> list[Chunk]:
    words = text.split()
    if not words:
        return []
    chunks, position, chunk_id = [], 0, start_id
    step = CHUNK_WORDS - OVERLAP_WORDS
    while position < len(words):
        body = " ".join(words[position:position + CHUNK_WORDS])
        if len(body.split()) >= 25 or not chunks:
            chunks.append(Chunk(chunk_id, source, page, body))
            chunk_id += 1
        if position + CHUNK_WORDS >= len(words):
            break
        position += step
    return chunks


def render_model_card(card: dict) -> list[tuple[str, str]]:
    """Flatten the model card into readable, citable sections."""
    sections: list[tuple[str, str]] = [
        ("purpose", card["purpose"]),
        ("outcome definition",
         f"{card['outcome']['definition']}. {card['outcome']['why_36_months']}"),
        ("cohort",
         "; ".join(f"{k}: {v}" for k, v in card["cohort"].items())),
        ("predictors",
         f"Used: {', '.join(card['predictors']['used'])}. "
         f"Imputation: {card['predictors']['imputation']}. "
         + " ".join(f"{k} dropped because {v}."
                    for k, v in card["predictors"]["dropped"].items())),
        ("validation",
         f"Scheme: {card['validation']['scheme']}. " + " ".join(
             f"{m['Model']}: AUC {m['AUC']} {m['AUC 95% CI (DeLong)']}, "
             f"Brier {m['Brier']}, calibration slope {m['Calibration slope']}."
             for m in card["validation"]["metrics"])
         + f" Cox concordance {card['validation']['concordance_cox']['c_index']}."),
        ("proportional hazards",
         card["validation"]["proportional_hazards"]["interpretation"]),
        ("limitations", " ".join(card["limitations"])),
    ]
    top = card["coefficients"]["logistic_glm"]
    ordered = sorted((c for c in top if c["term"] != "(Intercept)"),
                     key=lambda c: abs(c["estimate_log_odds"]), reverse=True)[:12]
    sections.append((
        "strongest logistic coefficients",
        " ".join(f"{c['label']}: odds ratio {c['odds_ratio']} "
                 f"(95% CI {c['ci_95'][0]}-{c['ci_95'][1]}, p={c['p_value']})."
                 for c in ordered)))
    return sections


def build_chunks() -> list[Chunk]:
    from pypdf import PdfReader

    chunks: list[Chunk] = []
    for name in PDFS:
        reader = PdfReader(CORPUS_DIR / name)
        for page_number, page in enumerate(reader.pages, start=1):
            text = clean(page.extract_text() or "")
            if len(text.split()) < 25:
                continue
            chunks.extend(window(text, name, page_number, len(chunks)))
        print(f"  {name}: {len(reader.pages)} pages")

    if MODEL_CARD.exists():
        card = json.loads(MODEL_CARD.read_text())
        for heading, body in render_model_card(card):
            chunks.extend(window(clean(f"{heading}. {body}"),
                                 "model_card.json", None, len(chunks)))
        print("  model_card.json: added")
    else:
        print("  model_card.json: MISSING - run `make models` first", file=sys.stderr)

    readme = ROOT / "README.md"
    if readme.exists() and "## Methodology" in readme.read_text():
        section = readme.read_text().split("## Methodology", 1)[1].split("\n## ", 1)[0]
        chunks.extend(window(clean(section), "README.md", None, len(chunks)))
        print("  README.md methodology: added")

    return chunks


def main() -> int:
    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer

    print("Corpus")
    fetch_pdfs()
    chunks = build_chunks()
    if not chunks:
        raise SystemExit("No chunks produced - is the corpus empty?")

    print(f"\nEmbedding {len(chunks)} chunks with {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    vectors = model.encode([c.text for c in chunks],
                           normalize_embeddings=True,
                           show_progress_bar=False).astype("float32")

    # Inner product on normalised vectors == cosine similarity.
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_DIR / "corpus.faiss"))
    (INDEX_DIR / "chunks.json").write_text(
        json.dumps({"model": EMBEDDING_MODEL,
                    "chunks": [asdict(c) for c in chunks]}, indent=2))

    by_source: dict[str, int] = {}
    for chunk in chunks:
        by_source[chunk.source] = by_source.get(chunk.source, 0) + 1
    print(f"\nIndexed {len(chunks)} chunks, dimension {vectors.shape[1]}")
    for source, count in sorted(by_source.items()):
        print(f"  {source:<45} {count:>4} chunks")
    print(f"\nWrote {INDEX_DIR.relative_to(ROOT)}/corpus.faiss and chunks.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
