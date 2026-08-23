"""Tests for the retrieval fine-tune and its committed report.

The training itself is not exercised here -- it needs torch, which CI does not
install. What is tested is the pair construction (where a subtle mistake makes
the evaluation meaningless) and the arithmetic in the committed report.
"""

from __future__ import annotations

import importlib.util
import random
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "docs" / "retrieval_eval.md"


def _module():
    path = ROOT / "pipeline" / "08_finetune.py"
    spec = importlib.util.spec_from_file_location("finetune", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CHUNKS = [
    {"chunk_id": 0, "text": (
        "The linked mortality file connects survey records to death certificates. "
        "Participants who did not link are assumed to be alive at follow-up. "
        "Follow-up time is measured in months from the date of examination. "
        "Underlying cause of death is recoded into a small number of categories.")},
    {"chunk_id": 1, "text": (
        "Survey weights account for the complex multistage sampling design. "
        "Analyses that ignore the weights describe the sample and not the nation. "
        "Variance estimation requires the primary sampling unit and stratum. "
        "The public use file suppresses some identifying detail.")},
    {"chunk_id": 2, "text": "Too short to split."},
]


def test_sentence_splitter_drops_fragments():
    module = _module()
    parts = module.sentences("Short one. " + "A much longer sentence that clearly "
                             "carries enough content to be usable as a query.")
    assert len(parts) == 1
    assert parts[0].startswith("A much longer")


def test_inverse_cloze_removes_the_query_from_its_own_positive():
    """If the query stays in the positive, the model learns string matching."""
    module = _module()
    pairs = module.make_pairs(CHUNKS, [0, 1], random.Random(0), per_chunk=4)
    assert pairs
    for query, positive, _ in pairs:
        assert query not in positive, "query leaked into its own positive passage"


def test_pairs_reference_their_source_chunk():
    module = _module()
    pairs = module.make_pairs(CHUNKS, [0, 1], random.Random(0), per_chunk=4)
    for query, _, chunk_id in pairs:
        assert query in CHUNKS[chunk_id]["text"]


def test_chunks_too_short_to_split_produce_no_pairs():
    module = _module()
    assert module.make_pairs(CHUNKS, [2], random.Random(0), per_chunk=4) == []


def test_train_and_eval_chunks_do_not_overlap():
    """Split is by chunk; a shared chunk would leak training text into eval."""
    module = _module()
    rng = random.Random(module.SEED)
    train = module.make_pairs(CHUNKS, [0], rng, per_chunk=4)
    evaluation = module.make_pairs(CHUNKS, [1], rng, per_chunk=4)
    assert {c for _, _, c in train}.isdisjoint({c for _, _, c in evaluation})


# ------------------------------------------------------------- the report

@pytest.fixture(scope="module")
def report() -> str:
    if not REPORT.exists():
        pytest.skip("docs/retrieval_eval.md not generated yet (run `make finetune`)")
    return REPORT.read_text(encoding="utf-8")


def test_report_deltas_are_arithmetically_consistent(report):
    rows = re.findall(r"\|\s*(recall@5|MRR)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([-+][\d.]+)\s*\|",
                      report)
    assert len(rows) == 2, "expected a base/fine-tuned/delta row per metric"
    for metric, base, tuned, delta in rows:
        assert float(tuned) - float(base) == pytest.approx(float(delta), abs=0.001), (
            f"{metric}: {tuned} - {base} != {delta}"
        )


def test_report_states_the_adoption_decision(report):
    assert "not adopted" in report or "worth adopting" in report
    assert "copilot continues to use the base model" in report or "rebuilding the index" in report


def test_report_does_not_overclaim_the_kind_of_fine_tuning(report):
    """"Fine-tuned an LLM" would be false; this is a bi-encoder."""
    assert "not instruction-tuning" in report
    assert "embedding model for retrieval" in report


def test_report_describes_the_inverse_cloze_construction(report):
    assert "inverse cloze" in report.lower()
    assert "removed from the positive" in report


def test_report_puts_the_numbers_in_context(report):
    """A reader needs the chance baseline to judge a 0.8 recall."""
    assert "at random" in report and "0.067" in report
