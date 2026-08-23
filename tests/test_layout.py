"""Structural tests for the repository.

These run in CI on every push. They deliberately touch no downloaded data and
no model, so CI never depends on CDC uptime or on a local LLM being present.
"""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def read(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


def test_license_is_present_and_attributed():
    licence = read("LICENSE")
    assert "MIT License" in licence
    assert "William Huang" in licence


@pytest.mark.parametrize(
    "target",
    ["setup", "setup-rag", "data", "features", "eda", "models",
     "index", "demo", "finetune", "test", "all", "clean"],
)
def test_makefile_declares_pipeline_target(target: str):
    """Every documented stage must be a real target, not just prose in help."""
    makefile = read("Makefile")
    assert f"\n{target}:" in makefile, f"Makefile is missing the '{target}' target"


def test_makefile_marks_targets_phony():
    phony = [line for line in read("Makefile").splitlines() if line.startswith(".PHONY")]
    declared = {t for line in phony for t in line.split(":", 1)[1].split()}
    for target in ("setup", "data", "models", "test", "clean"):
        assert target in declared, f"'{target}' should be declared .PHONY"


def test_data_directory_is_not_committed():
    """Downloaded CDC data must never enter git history."""
    ignored = read(".gitignore")
    assert "data/" in ignored
    assert "models/" in ignored
    assert "index/" in ignored


def test_rag_dependencies_are_isolated_from_core():
    """torch must not be a prerequisite for running the statistical pipeline."""
    core = read("requirements.txt").lower()
    rag = read("requirements-rag.txt").lower()
    assert "sentence-transformers" not in core
    assert "faiss" not in core
    assert "sentence-transformers" in rag
    assert "faiss-cpu" in rag


def test_expected_source_directories_exist():
    for directory in ("R", "pipeline", "sql", "tests", "prompts", "docs"):
        assert (ROOT / directory).is_dir(), f"missing directory: {directory}"
