# src/ibbi/utils/data.py

"""
Access to the Bark and Ambrosia Beetle Detection Benchmark, the single dataset used by the `ibbi` package.

The benchmark (v2.0.1, https://huggingface.co/datasets/IBBI-bio/bark-ambrosia-beetle-benchmark, DOI
10.5281/zenodo.22695714) is a specimen-disjoint, species-level COCO detection dataset with four splits:

    train         6,096 images, 65 trainable species
    iid_test        620 images, held-out specimens of the trained species (650 scored annotations)
    inat_test        74 images, iNaturalist field photographs of trained species (photographic domain shift)
    semantic_ood  7,701 images, 110 species never seen in training, banded by taxonomic distance

`iid_test` and `inat_test` contain `iscrowd=1` regions: real specimens on the same image that are not scored. They
must be ignored, not counted as misses; `ibbi.Evaluator` and `ibbi.evaluate.benchmark` handle this.

Only the files of the requested split are downloaded and they are cached under `ibbi.get_cache_dir()`. The older
datasets (`IBBI-bio/ibbi_test_data`, `IBBI-bio/ibbi_ood_data`, `IBBI-bio/ibbi_shap_dataset`) are deprecated and no
longer used by the package.
"""

import json
import random
import time
import warnings
from collections import defaultdict
from collections.abc import Iterator, Sequence
from functools import cached_property
from importlib import resources
from pathlib import Path
from typing import Any

import pandas as pd
from huggingface_hub import hf_hub_download, snapshot_download
from PIL import Image, ImageOps

from .cache import get_cache_dir

Image.MAX_IMAGE_PIXELS = None

BENCHMARK_REPO_ID = "IBBI-bio/bark-ambrosia-beetle-benchmark"
BENCHMARK_VERSION = "2.0.1"
BENCHMARK_REVISION = "8dc5a58e09c429e7d91077437846f424f5ab0c4e"  # v2.0.1 on the Hugging Face Hub
BENCHMARK_SPLITS = ("train", "iid_test", "inat_test", "semantic_ood")
TAXONOMY_LEVELS = ("subfamily", "tribe", "genus", "species")

# Metadata needed by every split (small files).
_METADATA_FILES = [
    "detection/annotations_coco/*.json",
    "species_taxonomy.csv",
    "image_licences.csv",
    "README.md",
    "CHANGELOG.md",
    "evaluation/*",
]


def _default_root(revision: str) -> Path:
    tag = BENCHMARK_VERSION if revision == BENCHMARK_REVISION else revision[:12]
    return get_cache_dir() / "datasets" / f"bark-ambrosia-beetle-benchmark-{tag}"


def download_benchmark(
    splits: str | Sequence[str] | None = None,
    local_dir: str | Path | None = None,
    revision: str = BENCHMARK_REVISION,
    images: bool = True,
) -> Path:
    """Downloads (or completes) a local copy of the benchmark and returns its root directory.

    Args:
        splits (str | Sequence[str] | None): Splits whose images should be downloaded. `None` downloads all four.
        local_dir (str | Path | None): Target directory. Defaults to `<ibbi cache>/datasets/bark-ambrosia-beetle-benchmark-<version>`.
        revision (str): Dataset revision on the Hugging Face Hub. Defaults to the pinned v2.0.1 commit, so results are
            reproducible; pass "main" for the newest version.
        images (bool): If False, only the annotations, taxonomy, licences and evaluation scripts are downloaded.

    Returns:
        Path: The benchmark root (it contains `detection/annotations_coco/`, `detection/images/<split>/`,
        `species_taxonomy.csv` and `evaluation/`).
    """
    if splits is None:
        splits = list(BENCHMARK_SPLITS)
    elif isinstance(splits, str):
        splits = [splits]
    bad = [s for s in splits if s not in BENCHMARK_SPLITS]
    if bad:
        raise ValueError(f"Unknown split(s) {bad}. Available: {list(BENCHMARK_SPLITS)}")
    root = Path(local_dir) if local_dir is not None else _default_root(revision)
    patterns = list(_METADATA_FILES)
    if images:
        patterns += [f"detection/images/{s}/*" for s in splits]
    _snapshot_with_retry(root, revision, patterns)
    return root


def _snapshot_with_retry(root: Path, revision: str, patterns: list[str], max_attempts: int = 30, wait_s: int = 320) -> None:
    """snapshot_download with waiting on HTTP 429.

    Every benchmark image is a separate file on the Hub, so a full split can exceed the Hub's rate limit
    (about 1,000 requests per 5 minutes for anonymous and free accounts). Files already downloaded are skipped on each
    retry, so the download simply resumes.
    """
    from huggingface_hub.errors import HfHubHTTPError

    for attempt in range(1, max_attempts + 1):
        try:
            snapshot_download(BENCHMARK_REPO_ID, repo_type="dataset", revision=revision, local_dir=str(root), allow_patterns=patterns, max_workers=4)
            return
        except HfHubHTTPError as e:
            status = getattr(getattr(e, "response", None), "status_code", None)
            if status != 429 or attempt == max_attempts:
                raise
            print(f"Hugging Face Hub rate limit reached; resuming in {wait_s // 60} min (attempt {attempt}/{max_attempts}) ...")
            time.sleep(wait_s)


def _load_rgb(path: Path, expect_wh: tuple[int, int] | None = None) -> Image.Image:
    """Opens an image as 8-bit RGB in the frame the COCO annotations use.

    The benchmark stores image sizes in the displayed (EXIF-rotated) frame. If the rotated size does not match but the
    raw size does, the raw frame is used, so boxes always line up with the returned image.
    """
    im = Image.open(path)
    raw = im
    rot = ImageOps.exif_transpose(im)
    if expect_wh is not None and tuple(rot.size) != tuple(expect_wh) and tuple(raw.size) == tuple(expect_wh):
        rot = raw
    if rot.mode != "RGB":
        if rot.mode.startswith("I;16") or rot.mode in ("I", "F"):
            import numpy as np

            arr = np.asarray(rot, dtype=np.float32)
            arr = (255 * (arr - arr.min()) / max(float(arr.max() - arr.min()), 1e-6)).astype("uint8")
            rot = Image.fromarray(arr)
        rot = rot.convert("RGB")
    return rot


class BenchmarkDataset(Sequence):
    """One split of the benchmark as an indexable sequence of examples.

    Each item is a dictionary::

        {
            "image": PIL.Image (RGB, loaded lazily),
            "image_id": int,                 # COCO image id, used in predictions
            "file_name": str, "image_path": str, "width": int, "height": int,
            "objects": {
                "bbox": [[x, y, w, h], ...],  # absolute pixels, COCO order
                "category": [species name, ...],
                "category_id": [int, ...],    # global benchmark category id (1..175)
                "iscrowd": [0 | 1, ...],      # 1 = real specimen that is not scored
                "subfamily": [...], "tribe": [...], "genus": [...],
            },
        }

    Use `dataset.coco_path` with `ibbi.evaluate.benchmark` for crowd-aware scoring.
    """

    def __init__(self, root: str | Path, split: str, indices: list[int] | None = None):
        if split not in BENCHMARK_SPLITS:
            raise ValueError(f"Unknown split '{split}'. Available: {list(BENCHMARK_SPLITS)}")
        self.root = Path(root)
        self.split = split
        self.coco_path = self.root / "detection" / "annotations_coco" / f"{split}.json"
        if not self.coco_path.exists():
            raise FileNotFoundError(f"{self.coco_path} not found. Run ibbi.download_benchmark(splits='{split}') first.")
        self._indices = indices

    @cached_property
    def coco(self) -> dict[str, Any]:
        with open(self.coco_path) as f:
            return json.load(f)

    @cached_property
    def _images(self) -> list[dict[str, Any]]:
        return sorted(self.coco["images"], key=lambda im: im["id"])

    @cached_property
    def _anns_by_image(self) -> dict[int, list[dict[str, Any]]]:
        out = defaultdict(list)
        for a in self.coco["annotations"]:
            out[a["image_id"]].append(a)
        return out

    @cached_property
    def categories(self) -> dict[int, str]:
        """Global category id -> species name for the categories of this split."""
        return {c["id"]: c["name"] for c in self.coco["categories"]}

    @cached_property
    def taxonomy(self) -> pd.DataFrame:
        return get_taxonomy()

    @cached_property
    def _lineage(self) -> dict[str, dict[str, str]]:
        t = self.taxonomy.set_index("scientificName")
        return {sp: {lvl: t.at[sp, lvl] for lvl in ("subfamily", "tribe", "genus")} for sp in t.index}

    def _order(self) -> list[int]:
        return self._indices if self._indices is not None else list(range(len(self._images)))

    def __len__(self) -> int:
        return len(self._order())

    def image_path(self, idx: int) -> Path:
        im = self._images[self._order()[idx]]
        return self.root / "detection" / "images" / self.split / Path(im["file_name"]).name

    def _item(self, idx: int) -> dict[str, Any]:
        im = self._images[self._order()[idx]]
        path = self.image_path(idx)
        anns = self._anns_by_image.get(im["id"], [])
        names = [self.categories.get(a["category_id"], str(a["category_id"])) for a in anns]
        lin = [self._lineage.get(n, {}) for n in names]
        return {
            "image_id": im["id"],
            "file_name": im["file_name"],
            "image_path": str(path),
            "width": im["width"],
            "height": im["height"],
            "objects": {
                "bbox": [list(map(float, a["bbox"])) for a in anns],
                "category": names,
                "category_id": [a["category_id"] for a in anns],
                "iscrowd": [int(a.get("iscrowd", 0)) for a in anns],
                "subfamily": [d.get("subfamily") for d in lin],
                "tribe": [d.get("tribe") for d in lin],
                "genus": [d.get("genus") for d in lin],
            },
        }

    def __getitem__(self, idx):  # type: ignore[override]
        if isinstance(idx, slice):
            return BenchmarkDataset(self.root, self.split, [self._order()[i] for i in range(*idx.indices(len(self)))])
        if idx < 0:
            idx += len(self)
        item = self._item(idx)
        item["image"] = _load_rgb(Path(item["image_path"]), (item["width"], item["height"]))
        return item

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for i in range(len(self)):
            yield self[i]

    def records(self) -> Iterator[dict[str, Any]]:
        """Iterates over the examples without opening the images (fast metadata access)."""
        for i in range(len(self)):
            yield self._item(i)

    def select(self, indices: Sequence[int]) -> "BenchmarkDataset":
        """Returns a view restricted to the given positions (like `datasets.Dataset.select`)."""
        order = self._order()
        return BenchmarkDataset(self.root, self.split, [order[i] for i in indices])

    def shuffle(self, seed: int = 0) -> "BenchmarkDataset":
        order = list(self._order())
        random.Random(seed).shuffle(order)
        return BenchmarkDataset(self.root, self.split, order)

    def __repr__(self) -> str:
        return f"BenchmarkDataset(split='{self.split}', n_images={len(self)}, root='{self.root}')"


def get_dataset(
    split: str = "iid_test",
    local_dir: str | Path | None = None,
    revision: str = BENCHMARK_REVISION,
    download: bool = True,
    **kwargs,
) -> BenchmarkDataset:
    """Loads one split of the Bark and Ambrosia Beetle Detection Benchmark.

    Args:
        split (str): One of "train", "iid_test", "inat_test", "semantic_ood". Defaults to "iid_test".
        local_dir (str | Path | None): Where the benchmark is (or should be) stored. Defaults to the ibbi cache.
        revision (str): Dataset revision. Defaults to the pinned v2.0.1 commit.
        download (bool): Download missing files from the Hugging Face Hub. Defaults to True.
        **kwargs: Accepted for backwards compatibility; `repo_id` other than the benchmark raises an error.

    Returns:
        BenchmarkDataset: The split, indexable and iterable; each item has "image" and "objects" keys.
    """
    repo_id = kwargs.pop("repo_id", BENCHMARK_REPO_ID)
    if repo_id != BENCHMARK_REPO_ID:
        raise ValueError(
            f"'{repo_id}' is no longer supported. ibbi uses only the Bark and Ambrosia Beetle Detection Benchmark "
            f"('{BENCHMARK_REPO_ID}'); the old datasets are deprecated."
        )
    root = Path(local_dir) if local_dir is not None else _default_root(revision)
    images_dir = root / "detection" / "images" / split
    if download and (not images_dir.exists() or not any(images_dir.iterdir())):
        print(f"Downloading the '{split}' split of {BENCHMARK_REPO_ID} (v{BENCHMARK_VERSION}) to {root} ...")
        download_benchmark(split, local_dir=root, revision=revision)
    return BenchmarkDataset(root, split)


def get_ood_dataset(local_dir: str | Path | None = None, revision: str = BENCHMARK_REVISION, **kwargs) -> BenchmarkDataset:
    """Deprecated: returns the benchmark's `semantic_ood` split (110 species never seen in training).

    The old `IBBI-bio/ibbi_ood_data` dataset is no longer used. Call `ibbi.get_dataset("semantic_ood")` instead.
    """
    warnings.warn(
        "get_ood_dataset() is deprecated and now returns the benchmark's 'semantic_ood' split; use ibbi.get_dataset('semantic_ood') instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    kwargs.pop("repo_id", None)
    kwargs.pop("split", None)
    return get_dataset("semantic_ood", local_dir=local_dir, revision=revision, **kwargs)


def get_shap_background_dataset(
    image_size: tuple[int, int] = (224, 224),
    n_images: int = 32,
    seed: int = 0,
    revision: str = BENCHMARK_REVISION,
) -> list[dict]:
    """Returns a small background set for SHAP, sampled from the benchmark's `train` split.

    Only the sampled images are downloaded. The old `IBBI-bio/ibbi_shap_dataset` is deprecated.

    Args:
        image_size (tuple[int, int]): Size (width, height) the images are resized to. Should match the size of the
            images being explained. Defaults to (224, 224).
        n_images (int): Number of background images. Defaults to 32.
        seed (int): Sampling seed. Defaults to 0.
        revision (str): Dataset revision. Defaults to the pinned v2.0.1 commit.

    Returns:
        list[dict]: `[{"image": PIL.Image}, ...]`, ready for `ibbi.Explainer.with_shap`.
    """
    root = _default_root(revision)
    ann = root / "detection" / "annotations_coco" / "train.json"
    if not ann.exists():
        download_benchmark(local_dir=root, revision=revision, images=False)
    images = sorted(json.load(open(ann))["images"], key=lambda im: im["id"])
    chosen = random.Random(seed).sample(images, min(n_images, len(images)))
    out = []
    for im in chosen:
        rel = f"detection/images/train/{Path(im['file_name']).name}"
        local = root / rel
        if not local.exists():
            hf_hub_download(BENCHMARK_REPO_ID, rel, repo_type="dataset", revision=revision, local_dir=str(root))
        out.append({"image": _load_rgb(local, (im["width"], im["height"])).resize(image_size)})
    return out


def get_taxonomy() -> pd.DataFrame:
    """Taxonomy of all 175 benchmark species (subfamily, tribe, genus, species, benchmark role, distance band).

    `benchmark_role` is "trainable" for the 65 species the IBBI models know and "semantic_ood" for the 110 held-out
    species; `distance_band_vs_trainable` is "near_genus", "mid_tribe" or "far_tribe" for held-out species.
    Source: `species_taxonomy.csv` of the benchmark v2.0.1 (CC BY 4.0).
    """
    with resources.files("ibbi.data").joinpath("species_taxonomy.csv").open("r") as f:
        return pd.read_csv(f)


def taxonomic_distance_matrix(species: Sequence[str] | None = None) -> pd.DataFrame:
    """Taxonomic distance between species: 0 same species, 1 same genus, 2 same tribe, 3 same subfamily, 4 otherwise.

    Args:
        species (Sequence[str] | None): Species names to include. Defaults to all 175 benchmark species.

    Returns:
        pd.DataFrame: Symmetric matrix indexed by species name.
    """
    t = get_taxonomy().drop_duplicates("scientificName").set_index("scientificName")
    names = list(species) if species is not None else list(t.index)
    unknown = [n for n in names if n not in t.index]
    if unknown:
        raise KeyError(f"Species not in the benchmark taxonomy: {unknown[:5]}{'...' if len(unknown) > 5 else ''}")
    lin = {n: (t.at[n, "subfamily"], t.at[n, "tribe"], t.at[n, "genus"], n) for n in names}
    rows = []
    for a in names:
        row = []
        for b in names:
            shared = 0
            for x, y in zip(lin[a], lin[b]):
                if x != y:
                    break
                shared += 1
            row.append(4 - shared)
        rows.append(row)
    return pd.DataFrame(rows, index=pd.Index(names), columns=pd.Index(names), dtype=float)
