"""Download, extract and load the torchmil datasets (Camelyon16, PANDA, RSNA-ICH)."""

import os
import shutil
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchmil.data import collate_fn
from torchmil.datasets.camelyon16mil_dataset import CAMELYON16MILDataset
from torchmil.datasets.pandamil_dataset import PANDAMILDataset
from torchmil.datasets.rsnamil_dataset import RSNAMILDataset

HF_URL = "https://huggingface.co/datasets/torchmil/{repo}/resolve/main/dataset/"
DEFAULT_DATA_DIR = "/content"  # Google Colab default

# `base`: sub-folder holding the bag files; `root_subdir`: dataset root inside data_dir.
DATASETS = {
    "camelyon16": dict(
        repo="Camelyon16_MIL", cls=CAMELYON16MILDataset, features="UNI",
        base="patches_512", label_dirs=["labels", "patch_labels", "coords"], root_subdir="",
    ),
    "panda": dict(
        repo="PANDA_MIL", cls=PANDAMILDataset, features="UNI",
        base="patches_512", label_dirs=["labels", "patch_labels", "coords"], root_subdir="",
    ),
    "rsna": dict(
        repo="RSNA_ICH_MIL", cls=RSNAMILDataset, features="resnet50",
        base="", label_dirs=["labels", "slice_labels"], root_subdir="root",
    ),
}


def _download(url, dest):
    """Download `url` to `dest` unless it already exists."""
    if dest.exists():
        print(f"[skip] {dest.name} already downloaded")
        return
    print(f"[download] {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, length=16 * 1024 * 1024)
    tmp.rename(dest)


def _extract(archive, target):
    """Extract `archive` into `target`.

    Some archives already contain a top-level folder named like `target`
    (e.g. RSNA `labels.tar.gz` -> `labels/...`); those are extracted into the parent.
    """
    target = Path(target)
    with tarfile.open(archive) as tar:
        names = [n.lstrip("./") for n in tar.getnames()]
        nested = all(n.startswith(target.name) for n in names if n)
        dest = target.parent if nested else target
        dest.mkdir(parents=True, exist_ok=True)
        print(f"[extract] {archive.name} -> {dest}")
        tar.extractall(dest, filter="data")


def _drop_incomplete_bags(features_dir, other_dirs):
    """Remove bags whose files are missing in any of the label folders."""
    for fname in os.listdir(features_dir):
        if all((Path(d) / fname).exists() for d in other_dirs):
            continue
        print(f"[clean] removing {fname} (missing in some folder)")
        for d in [features_dir, *other_dirs]:
            (Path(d) / fname).unlink(missing_ok=True)


def prepare_dataset(name, data_dir=None, features=None, remove_archives=False):
    """Download + extract a dataset. Returns the root folder expected by torchmil.

    Args:
        name: 'camelyon16', 'panda' or 'rsna'.
        data_dir: where the compressed files are stored (default: /content, as in Colab).
        features: feature extractor name (default: UNI for WSIs, resnet50 for RSNA).
        remove_archives: delete the .tar.gz files after extraction.
    """
    spec = DATASETS[name]
    data_dir = Path(data_dir or DEFAULT_DATA_DIR)
    root = data_dir / spec["root_subdir"]
    base = root / spec["base"]
    features = features or spec["features"]
    url = HF_URL.format(repo=spec["repo"])
    prefix = f"{spec['base']}/" if spec["base"] else ""
    data_dir.mkdir(parents=True, exist_ok=True)

    # (remote path, local folder where it must be extracted)
    archives = [(f"{prefix}{d}.tar.gz", base / d) for d in spec["label_dirs"]]
    archives.append((f"{prefix}features/features_{features}.tar.gz",
                     base / "features" / f"features_{features}"))

    for remote, target in archives:
        archive = data_dir / Path(remote).name
        if target.exists() and any(target.iterdir()):
            print(f"[skip] {target} already extracted")
            continue
        _download(url + remote, archive)
        _extract(archive, target)
        if remove_archives:
            archive.unlink()

    # splits.csv is downloaded to data_dir; for RSNA it must live inside `root`.
    splits = data_dir / "splits.csv"
    if not (root / "splits.csv").exists():
        _download(url + "splits.csv", splits)
        if splits.resolve() != (root / "splits.csv").resolve():
            shutil.move(str(splits), root / "splits.csv")

    _drop_incomplete_bags(base / "features" / f"features_{features}",
                          [base / d for d in spec["label_dirs"]])
    return root


def load_datasets(name, root, features=None, val_size=0.15, seed=42, load_at_init=True):
    """Build train/val/test datasets. Validation is a stratified split of train."""
    spec = DATASETS[name]
    kwargs = dict(root=str(root), features=features or spec["features"],
                  bag_keys=["X", "Y", "y_inst"], load_at_init=load_at_init)
    full_train = spec["cls"](partition="train", **kwargs)
    test = spec["cls"](partition="test", **kwargs)

    labels = np.array([full_train[i]["Y"].item() for i in range(len(full_train))])
    train_idx, val_idx = train_test_split(
        np.arange(len(full_train)), test_size=val_size, random_state=seed, stratify=labels
    )
    train, val = Subset(full_train, train_idx.tolist()), Subset(full_train, val_idx.tolist())
    print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)} bags")
    return train, val, test


def make_loaders(train, val, test, batch_size=32, num_workers=0):
    """DataLoaders that pad bags and return a `mask` (torchmil collate_fn)."""
    def loader(ds, shuffle):
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          collate_fn=collate_fn, num_workers=num_workers)
    return loader(train, True), loader(val, False), loader(test, False)
