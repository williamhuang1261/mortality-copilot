"""Fine-tune the retrieval bi-encoder and report whether it actually helped.

Training pairs are generated with the Inverse Cloze Task (Lee et al., 2019,
"Latent Retrieval for Weakly Supervised Open Domain Question Answering"): sample
a sentence from a chunk, use it as the query, and use the REST of that chunk as
the positive. Removing the sampled sentence matters -- leaving it in makes the
positive contain the query verbatim, and the model learns exact string overlap
rather than anything about the corpus.

The split is by chunk, not by pair, so no evaluation chunk contributes any
training pair.

This fine-tunes an embedding model for retrieval. It is NOT instruction-tuning
or LoRA on a generative model, and the README says so.
"""

from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INDEX_DIR = ROOT / "index"
MODEL_DIR = ROOT / "models" / "minilm-ft"
REPORT = ROOT / "docs" / "retrieval_eval.md"

SEED = 20260823
QUERIES_PER_CHUNK = 4
TEST_FRACTION = 0.20
EPOCHS = 2
BATCH_SIZE = 16
TOP_K = 5


def sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip())
            if len(s.split()) >= 8]


def make_pairs(chunks: list[dict], chunk_ids: list[int],
               rng: random.Random, per_chunk: int) -> list[tuple[str, str, int]]:
    """(query, positive_passage, source_chunk_id) via the inverse cloze task."""
    pairs = []
    for cid in chunk_ids:
        parts = sentences(chunks[cid]["text"])
        if len(parts) < 2:
            continue
        for sentence in rng.sample(parts, min(per_chunk, len(parts))):
            remainder = " ".join(p for p in parts if p != sentence)
            if len(remainder.split()) < 20:
                continue
            pairs.append((sentence, remainder, cid))
    return pairs


def evaluate(model, chunks: list[dict], eval_pairs: list[tuple[str, str, int]]) -> dict:
    """Recall@5 and MRR, retrieving each query against the whole corpus."""
    import numpy as np

    corpus = model.encode([c["text"] for c in chunks],
                          normalize_embeddings=True, show_progress_bar=False)
    queries = model.encode([q for q, _, _ in eval_pairs],
                           normalize_embeddings=True, show_progress_bar=False)
    similarity = queries @ corpus.T
    ranked = np.argsort(-similarity, axis=1)

    hits, reciprocal = 0, 0.0
    for row, (_, _, gold) in enumerate(eval_pairs):
        order = list(ranked[row])
        rank = order.index(gold) + 1
        if rank <= TOP_K:
            hits += 1
        reciprocal += 1.0 / rank
    n = len(eval_pairs)
    return {"n_queries": n, "recall_at_5": hits / n, "mrr": reciprocal / n}


def seed_everything() -> None:
    """Seed every source of randomness the training touches.

    Seeding only the data split is not enough: DataLoader shuffling and weight
    initialisation are separately random, and two runs of this script produced
    recall@5 deltas of -0.037 and +0.000. A report that commits numbers has to
    reproduce them.
    """
    import numpy as np
    import torch
    from transformers import set_seed

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(False)  # CPU matmul reductions still vary
    set_seed(SEED)


def main() -> int:
    import torch
    from sentence_transformers import SentenceTransformer, InputExample, losses
    from torch.utils.data import DataLoader

    seed_everything()

    if not (INDEX_DIR / "chunks.json").exists():
        raise SystemExit("index/chunks.json not found. Run `make index` first.")

    payload = json.loads((INDEX_DIR / "chunks.json").read_text())
    chunks = payload["chunks"]
    base_name = payload["model"]

    rng = random.Random(SEED)
    ids = list(range(len(chunks)))
    rng.shuffle(ids)
    n_test = max(1, int(len(ids) * TEST_FRACTION))
    test_ids, train_ids = sorted(ids[:n_test]), sorted(ids[n_test:])

    train_pairs = make_pairs(chunks, train_ids, rng, QUERIES_PER_CHUNK)
    eval_pairs = make_pairs(chunks, test_ids, rng, QUERIES_PER_CHUNK)

    print(f"Corpus: {len(chunks)} chunks")
    print(f"  train {len(train_ids)} chunks -> {len(train_pairs)} pairs")
    print(f"  test  {len(test_ids)} chunks -> {len(eval_pairs)} queries")
    print(f"  split is by chunk, so no test chunk contributes a training pair\n")

    print(f"Evaluating base model ({base_name})")
    base_model = SentenceTransformer(base_name)
    before = evaluate(base_model, chunks, eval_pairs)
    print(f"  recall@{TOP_K} {before['recall_at_5']:.3f}   MRR {before['mrr']:.3f}\n")

    print(f"Fine-tuning: {EPOCHS} epochs, batch {BATCH_SIZE}, "
          f"MultipleNegativesRankingLoss, CPU")
    model = SentenceTransformer(base_name)
    examples = [InputExample(texts=[q, p]) for q, p, _ in train_pairs]
    generator = torch.Generator()
    generator.manual_seed(SEED)
    loader = DataLoader(examples, shuffle=True, batch_size=BATCH_SIZE,
                        drop_last=len(examples) > BATCH_SIZE, generator=generator)
    model.fit(train_objectives=[(loader, losses.MultipleNegativesRankingLoss(model))],
              epochs=EPOCHS, warmup_steps=max(1, len(loader) // 10),
              show_progress_bar=False)

    MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(MODEL_DIR))
    after = evaluate(model, chunks, eval_pairs)
    print(f"\n  recall@{TOP_K} {after['recall_at_5']:.3f}   MRR {after['mrr']:.3f}")

    d_recall = after["recall_at_5"] - before["recall_at_5"]
    d_mrr = after["mrr"] - before["mrr"]

    print(f"\n{'Metric':<12} {'Base':>8} {'Fine-tuned':>12} {'Delta':>9}")
    print(f"{'recall@' + str(TOP_K):<12} {before['recall_at_5']:>8.3f} "
          f"{after['recall_at_5']:>12.3f} {d_recall:>+9.3f}")
    print(f"{'MRR':<12} {before['mrr']:>8.3f} {after['mrr']:>12.3f} {d_mrr:>+9.3f}")

    # Three-way, because the interesting outcome here is "mixed" and an
    # or-condition on either metric would let a single favourable number
    # declare victory over a regression in the other.
    improved = d_recall > 0.01 and d_mrr > 0.01
    regressed = d_recall < -0.01 and d_mrr < -0.01
    if improved:
        verdict = ("Both metrics improved. The fine-tuned weights would be worth "
                   "adopting; doing so requires rebuilding the index with them.")
    elif regressed:
        verdict = ("**Both metrics regressed.** The fine-tune made retrieval worse "
                   "and is not adopted.")
    else:
        if d_mrr > 0.01 >= d_recall:
            reading = (
                f"MRR improved while recall@{TOP_K} did not. The model reordered "
                "passages it was already retrieving rather than retrieving better "
                "ones, which is a real but narrow gain.")
        elif d_recall > 0.01 >= d_mrr:
            reading = (
                f"recall@{TOP_K} improved without a better mean rank: more correct "
                "passages reached the top 5, but no higher within it.")
        else:
            reading = "Neither metric moved materially."
        verdict = (
            "**Mixed, and therefore not adopted.** "
            f"recall@{TOP_K} moved {d_recall:+.3f} and MRR moved {d_mrr:+.3f}. "
            f"{reading} On {len(train_pairs)} training pairs over a "
            f"{len(chunks)}-chunk corpus, deltas this small are within the range "
            "a different seed would produce.")

    adoption = (
        "**The copilot continues to use the base model.** `index/chunks.json` "
        "records the model the index was built with, and `pipeline/copilot.py` "
        "loads exactly that model, so the embeddings and the query encoder can "
        "never drift apart. Adopting the tuned weights means rebuilding the "
        "index with them, which is deliberately not automatic on a result like "
        "this one."
    )

    REPORT.write_text(f"""# Retrieval evaluation

Generated by `pipeline/08_finetune.py` — regenerate with `make finetune`.
Seed {SEED}, {EPOCHS} epochs, batch {BATCH_SIZE}, CPU only.

## Method

Training pairs come from the **inverse cloze task**: sample a sentence from a
chunk, use it as the query, and use the rest of that chunk as the positive
passage. The sampled sentence is removed from the positive — leaving it in
would let the model win by exact string overlap instead of learning anything
about this corpus.

The split is **by chunk**, not by pair, so no evaluation chunk contributes any
training pair. Each query is retrieved against the entire {len(chunks)}-chunk
corpus.

| Split | Chunks | Queries |
| --- | --- | --- |
| Train | {len(train_ids)} | {len(train_pairs)} |
| Test | {len(test_ids)} | {len(eval_pairs)} |

## Results

| Metric | Base | Fine-tuned | Delta |
| --- | ---: | ---: | ---: |
| recall@{TOP_K} | {before['recall_at_5']:.3f} | {after['recall_at_5']:.3f} | {d_recall:+.3f} |
| MRR | {before['mrr']:.3f} | {after['mrr']:.3f} | {d_mrr:+.3f} |

{verdict}

{adoption}

## Reading these numbers honestly

The corpus is {len(chunks)} chunks. Picking 5 at random would give recall@5 of
about {5 / len(chunks):.3f}, so the base model is already operating far above
chance and there is limited headroom. A large improvement here would be more
suspicious than a small one.

This fine-tunes a **bi-encoder for retrieval**. It is not instruction-tuning and
not LoRA on a generative model. The accurate way to describe it is "fine-tuned an
embedding model for retrieval".
""")
    print(f"\nWrote {REPORT.relative_to(ROOT)} and {MODEL_DIR.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
