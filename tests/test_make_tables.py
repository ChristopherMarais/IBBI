"""The README benchmark summary built by benchmarks/make_tables.py."""

import importlib.util
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _make_tables():
    spec = importlib.util.spec_from_file_location("make_tables", ROOT / "benchmarks" / "make_tables.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_ranked_table_sorts_bolds_and_marks_winner():
    mt = _make_tables()
    rows = [("a", [0.5, 0.9]), ("b", [0.7, 0.2]), ("c", [None, 0.1])]
    table, winner = mt.ranked_table(rows, ["score ↑", "error ↓"], 0, lower_better=(1,))
    lines = table.splitlines()
    assert winner == "b"
    assert lines[2].startswith("| 🥇 | b | **0.700** | 0.200 |")
    assert lines[3].startswith("| 🥈 | a | 0.500 | 0.900 |")
    assert lines[4] == "| 🥉 | c | – | **0.100** |"  # missing values rank last; lowest error is bold


def test_readme_summary_matches_results():
    mt = _make_tables()
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    block = re.search(r"<!-- BENCHMARK_SUMMARY_START -->\n(.*?)\n<!-- BENCHMARK_SUMMARY_END -->", readme, re.S).group(1)
    assert block == mt.summary_table(ROOT / "benchmarks" / "results"), "run: python benchmarks/make_tables.py --readme-only"
