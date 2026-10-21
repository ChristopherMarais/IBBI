# Notes for Claude

## Contributing: pull requests only
- Make every change through a pull request. Start a new branch (or worktree) off origin/main for each topic.
- Never commit to, push to or merge into main. Leave the owner's local main checkout alone.
- Never merge pull requests. Open them; the owner reviews, merges and deploys.
- Once a PR is merged, put any later changes in a NEW PR. Check `gh pr view N --json state` before pushing more
  commits to an existing branch.
- Keep PRs small and reviewable, one topic each, with tests. Put new tests in new files where possible so open PRs
  merge cleanly.
- Every PR body: plain English, what changed on each page/feature, which tests were run locally, a checklist of
  what to try, and "Closes #N" when it fixes an issue.

## Deployments and servers: the owner only
- Never deploy. Don't run or dispatch deploy workflows, and don't publish releases (if they trigger deploys).
- Don't run anything against production or staging servers.
- Don't start any workflow that touches a server (backups, reminders, scheduled jobs, ...) unless the owner asks
  for that exact workflow by name.

## Secrets
- Never commit secret files (.env.prod and similar). They live only on the server and are gitignored.
- Never print or paste passwords, keys or tokens. To check whether two copies match, compare fingerprints
  (e.g. `md5sum`), never the values.
- Never handle the owner's login tokens; the owner logs in themselves.

## Destructive actions
- Never delete anything on external services (repos, files, branches, history, models, data) unless told to
  explicitly. Do a dry run, back up, make the smallest change, and diff afterwards.
- Never run `git clean -fdx` on a server, and never prune Docker volumes blindly (databases live in volumes).
- Ask before cleaning up branches, worktrees or caches; keep a restore list (names + SHAs) when you do.

## Local environment
- Don't touch the owner's own running stack. Use your own Docker Compose project name (`-p <name>`), one stack at a
  time, and run `down -v` afterwards.
- Keep the number of worktrees low; delete them once their PR is merged.
- Don't add OS-specific code; the app runs in Docker.

## Testing
- Locally, run ONLY the test modules your change touches (search the tests for the IDs, classes and strings you
  changed). Never run the full suite locally; CI runs it.
- After template, CSS or JS changes, check the page in a real (headless) browser: look for console errors and take
  screenshots at desktop and phone widths.

## Working style
- Before building from an issue, check it against the code; many describe things that already exist.
- Explain jargon plainly. If the owner says they don't understand a question, ask it again more concretely.
- Keep user-facing text short.
- Follow the house style guide (wherever it's written down) on every new page.
- Put reports and checklists in the chat reply or a local .md file, not in hosted pages.

---

## IBBI in particular

### What it is
`ibbi` (Intelligent Bark Beetle Identifier) is a Python package on PyPI: detectors and hierarchical classifiers for bark
and ambrosia beetles, a two-stage identification pipeline, and an evaluator for the Bark and Ambrosia Beetle Detection
Benchmark. It is an inference package; training code lives elsewhere. Owner: Chris (GitHub ChristopherMarais).

- Model weights: Hugging Face org **IBBI-bio** (`IBBI-bio/ibbi_*`). Benchmark data:
  `IBBI-bio/bark-ambrosia-beetle-benchmark` (pinned revision in `src/ibbi/utils/data.py`). Never delete or overwrite
  anything there.
- Docs site: <https://gcmarais.com/IBBI/> (MkDocs Material, `docs/` + `mkdocs.yml`).

### Layout
- `src/ibbi/__init__.py` – public API (`create_model`, `create_pipeline`, `list_models`, `get_dataset`, `Evaluator`,
  `Explainer`, `get_taxonomy`).
- `src/ibbi/models/` – `_registry.py` (names and aliases), `detectors.py` (YOLO / RT-DETR), `classifiers.py`
  (hierarchical classifiers), `zero_shot.py`, `codino.py` + `_codino.py` (Co-DINO on plain PyTorch), `_gpu.py` (GPU
  preprocessing fast path).
- `src/ibbi/pipeline.py` – detector + classifier pipeline.
- `src/ibbi/evaluate/` – benchmark metrics. `_reference_evaluator.py` is the benchmark's own evaluator, **vendored
  unchanged: never edit or reformat it** (excluded from ruff).
- `src/ibbi/explain/` – LIME and SHAP. `src/ibbi/utils/` – Hub downloads, cache, dataset access.
- `src/ibbi/data/` – `ibbi_model_summary.csv` (model table) and `species_taxonomy.csv` (175 species).
- `benchmarks/` – `run_benchmark.py` (runs a model; needs a GPU and the full benchmark), `make_tables.py` (writes the
  tables), `results/*.json` (small headline results kept in the repo).
- `tests/` – `conftest.py` builds a synthetic benchmark, a tiny classifier and fake detectors, so the default tests
  run offline on CPU. Tests marked `slow` download real weights and only run with `--run-slow`.

### Environment (Poetry, not pixi)
IBBI is a Python package and uses **Poetry** (`pyproject.toml`, `poetry.lock`; version from git tags via
poetry-dynamic-versioning). `poetry.toml` sets `virtualenvs.create = false`, so Poetry installs into whatever
environment is active. The owner's own environment is the conda env `IBBI` (`environment.yml`: Python 3.11 + Poetry +
CUDA); don't install into or upgrade it. For your own checks, make a throwaway venv outside the repo and activate it
before running Poetry or pip.

```bash
poetry install --with dev                               # in an activated env of your own
pytest -q -p no:warnings tests/test_data.py             # run only the modules you touched
ruff check src tests benchmarks                         # what CI runs
```

CI does not use the lock file (see below). To reproduce a CI failure, install the way CI does instead:
`pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`, then `pip install -e . pytest ruff`.
Change dependencies in `pyproject.toml` and regenerate `poetry.lock` (`poetry lock`) in the same PR.

### CI and releases
- `.github/workflows/tests.yml` (every PR and push to main): ruff, then the fast tests on Python 3.11 and 3.12 with
  CPU PyTorch and the **latest** dependency versions installed by pip (no lock file), so new upstream releases can
  break CI without a code change. Check `gh run list --workflow tests.yml` and `gh run view <id> --log-failed`.
- `publish-to-pypi.yml` and `docs.yml` run when a **release is published**: never create or publish a release, and
  never dispatch these workflows.

### Conventions
- Ruff, line length 150 (`pyproject.toml`). Google-style docstrings (they feed the API docs).
- Don't hand-edit generated tables: the benchmark tables in `docs/benchmark.md` and the summary in `README.md`
  (between `BENCHMARK_SUMMARY` markers) come from `benchmarks/make_tables.py`.
- Open text files with `encoding="utf-8"`: the README and tables contain non-ASCII characters and the owner works on
  Windows.
- Tests must not depend on packages that are only pulled in by other dependencies (e.g. `httpx`, which
  huggingface-hub 2.x replaced with `httpx2`); use small fakes instead.
