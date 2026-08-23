"""Python side of mortality-copilot: data loading, retrieval and the CLI.

Pipeline stages keep their numeric prefixes (`01_load_duckdb.py`,
`02_features.py`, `06_index.py`) because they run in order and the numbers
document that order. The copilot is not a stage -- it is a tool the user invokes
repeatedly -- so it lives in an importable module, reachable as
`python -m pipeline.copilot`.

The package is named `pipeline` rather than `py` because a top-level `py`
package shadows the `py` library that pytest imports internally, which breaks
the test suite as soon as the repository root is on sys.path.
"""
