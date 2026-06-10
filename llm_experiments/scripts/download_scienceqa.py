#!/usr/bin/env python3
"""Download ScienceQA images via HuggingFace mirror (faster than S3 in CN)."""

import json
import os
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "llm_experiments" / "data" / "ScienceQA"
IMAGES_DIR = DATA_DIR / "images"

SPLIT_MAP = {
    "train": "train",
    "validation": "val",
    "test": "test",
}


def main() -> None:
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    for proxy in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "all_proxy"):
        os.environ.pop(proxy, None)

    splits = json.loads((DATA_DIR / "pid_splits.json").read_text())
    problems = json.loads((DATA_DIR / "problems.json").read_text())

    stats = {"saved": 0, "skipped": 0, "no_image": 0}

    for hf_split, local_split in SPLIT_MAP.items():
        pids = splits[local_split]
        ds = load_dataset("derek-thomas/ScienceQA", split=hf_split)
        assert len(ds) == len(pids), f"{hf_split}: HF={len(ds)} vs local={len(pids)}"

        for idx, pid in enumerate(tqdm(pids, desc=local_split)):
            problem = problems[str(pid)]
            if not problem.get("image"):
                stats["no_image"] += 1
                continue

            out_dir = IMAGES_DIR / local_split / str(pid)
            out_path = out_dir / problem["image"]
            if out_path.exists():
                stats["skipped"] += 1
                continue

            image = ds[idx]["image"]
            if image is None:
                stats["no_image"] += 1
                continue

            out_dir.mkdir(parents=True, exist_ok=True)
            image.save(out_path)
            stats["saved"] += 1

    readme = {
        "dataset": "derek-thomas/ScienceQA",
        "source": "https://huggingface.co/datasets/derek-thomas/ScienceQA",
        "official_repo": "https://github.com/lupantech/ScienceQA",
        "splits": {k: len(v) for k, v in splits.items() if k in ("train", "val", "test")},
        "image_dir": "images/{train,val,test}/{pid}/",
        "text_files": ["problems.json", "pid_splits.json"],
        **stats,
    }
    (DATA_DIR / "README.json").write_text(json.dumps(readme, indent=2) + "\n")

    print("Done:", stats)
    print("README:", DATA_DIR / "README.json")


if __name__ == "__main__":
    main()
