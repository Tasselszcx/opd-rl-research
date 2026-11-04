"""Prepare isolated JSONL splits for G-OPD without downloading any data.

This module is dependency-free until reading or writing Parquet. Reference
solutions are teacher-only metadata and never enter the student prompt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import unicodedata
from typing import Any

SPLITS = ("train", "meta_val", "test")
DATA_SOURCE = "meta_teaching_math"
SCHEMA_VERSION = 1


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_question(question: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", question)).strip().casefold()


def question_hash(question: str) -> str:
    return hashlib.sha256(normalize_question(question).encode("utf-8")).hexdigest()


def load_jsonl(path: str | Path, split: str) -> list[dict[str, Any]]:
    if split not in SPLITS:
        raise ValueError(f"Unknown split: {split}")
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            location = f"{path}:{line_number}"
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{location}: invalid JSON: {exc.msg}") from exc
            if not isinstance(raw, dict):
                raise ValueError(f"{location}: expected a JSON object")
            question = raw.get("question", raw.get("prompt"))
            if not isinstance(question, str) or not question.strip():
                raise ValueError(f"{location}: question (or prompt) must be a nonempty string")
            answer = raw.get("answer")
            if isinstance(answer, bool) or not isinstance(answer, (str, int, float)):
                raise ValueError(f"{location}: answer must be a string or number")
            if isinstance(answer, float) and not math.isfinite(answer):
                raise ValueError(f"{location}: answer must be finite")
            if not str(answer).strip():
                raise ValueError(f"{location}: answer cannot be empty")
            raw_id = raw.get("id")
            if raw_id is not None and (isinstance(raw_id, bool) or not isinstance(raw_id, (str, int))):
                raise ValueError(f"{location}: id must be a nonempty string or integer")
            sample_id = str(raw_id) if raw_id is not None else question_hash(question)
            if not sample_id.strip():
                raise ValueError(f"{location}: id cannot be empty")
            row = {"id": sample_id, "question": question, "answer": str(answer)}
            if split == "train":
                reference = raw.get("reference_solution")
                if not isinstance(reference, str) or not reference.strip():
                    raise ValueError(f"{location}: train requires nonempty reference_solution")
                row["reference_solution"] = reference
            # Discard reference_solution and all other input metadata on held-out data.
            rows.append(row)
    if not rows:
        raise ValueError(f"{path}: {split} split is empty")
    return rows


def validate_splits(splits: dict[str, list[dict[str, Any]]]) -> None:
    seen_ids: dict[str, str] = {}
    seen_questions: dict[str, str] = {}
    if set(splits) != set(SPLITS):
        raise ValueError(f"Expected exactly these splits: {SPLITS}")
    for split in SPLITS:
        if not splits[split]:
            raise ValueError(f"{split} split is empty")
        for row in splits[split]:
            sample_id = str(row["id"])
            q_hash = question_hash(row["question"])
            if sample_id in seen_ids:
                raise ValueError(f"Duplicate sample id {sample_id!r}: {seen_ids[sample_id]} and {split}")
            if q_hash in seen_questions:
                raise ValueError(f"Duplicate normalized question: {seen_questions[q_hash]} and {split}")
            seen_ids[sample_id] = split
            seen_questions[q_hash] = split


def build_rows(records: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    if split not in SPLITS:
        raise ValueError(f"Unknown split: {split}")
    result = []
    for index, record in enumerate(records):
        extra = {"sample_id": str(record["id"]), "split": split, "index": index}
        if split == "train":
            extra["reference_solution"] = record["reference_solution"]
        result.append({
            "data_source": DATA_SOURCE,
            "prompt": [{"role": "user", "content": record["question"]}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": str(record["answer"])},
            "extra_info": extra,
        })
    return result


def _arrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("Parquet support requires pyarrow; install the documented MVP environment first.") from exc
    return pa, pq


def prepare(train_jsonl: str | Path, meta_val_jsonl: str | Path,
            test_jsonl: str | Path, output_dir: str | Path, *, demo_only: bool = False) -> dict:
    sources = dict(zip(SPLITS, map(Path, (train_jsonl, meta_val_jsonl, test_jsonl))))
    records = {split: load_jsonl(source, split) for split, source in sources.items()}
    validate_splits(records)
    pa, pq = _arrow()
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION, "data_source": DATA_SOURCE,
        "demo_only": bool(demo_only), "splits": {},
    }
    for split in SPLITS:
        path = destination / f"{split}.parquet"
        temporary = path.with_suffix(".parquet.tmp")
        pq.write_table(pa.Table.from_pylist(build_rows(records[split], split)), temporary)
        temporary.replace(path)
        manifest["splits"][split] = {
            "source": str(sources[split].resolve()), "source_sha256": sha256_file(sources[split]),
            "path": path.name, "sha256": sha256_file(path), "count": len(records[split]),
            "question_hashes": [question_hash(row["question"]) for row in records[split]],
            "sample_ids": [row["id"] for row in records[split]],
        }
    manifest_path = destination / "manifest.json"
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(manifest_path)
    return manifest


def validate_manifest(manifest_path: str | Path, *, check_parquet: bool = True) -> dict:
    """Check file integrity, actual schemas and split isolation; raise on failure."""
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("data_source") != DATA_SOURCE:
        raise ValueError("Unsupported data manifest schema or data_source")
    if set(manifest.get("splits", {})) != set(SPLITS):
        raise ValueError("Manifest must contain train, meta_val and test")
    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()
    seen_paths: set[Path] = set()
    _, pq = _arrow() if check_parquet else (None, None)
    for split in SPLITS:
        item = manifest["splits"][split]
        path = (manifest_path.parent / item["path"]).resolve()
        if path in seen_paths:
            raise ValueError(f"{split}: two splits alias the same Parquet path: {path}")
        seen_paths.add(path)
        if not path.is_file():
            raise ValueError(f"{split}: Parquet file does not exist: {path}")
        if sha256_file(path) != item["sha256"]:
            raise ValueError(f"{split}: file SHA256 does not match manifest")
        count, ids, hashes = item["count"], item["sample_ids"], item["question_hashes"]
        if not isinstance(count, int) or count <= 0 or len(ids) != count or len(hashes) != count:
            raise ValueError(f"{split}: invalid manifest count")
        if len(set(ids)) != count or seen_ids.intersection(ids):
            raise ValueError(f"{split}: duplicate sample ids in manifest")
        if len(set(hashes)) != count or seen_hashes.intersection(hashes):
            raise ValueError(f"{split}: overlapping normalized questions in manifest")
        seen_ids.update(ids)
        seen_hashes.update(hashes)
        if check_parquet:
            table = pq.read_table(path)
            if table.num_rows != count:
                raise ValueError(f"{split}: Parquet row count mismatch")
            required = {"data_source", "prompt", "ability", "reward_model", "extra_info"}
            if set(table.column_names) != required:
                raise ValueError(f"{split}: unexpected Parquet columns")
            for index, row in enumerate(table.to_pylist()):
                prompt, reward, extra = row["prompt"], row["reward_model"], row["extra_info"]
                if not isinstance(prompt, list) or len(prompt) != 1 or prompt[0].get("role") != "user":
                    raise ValueError(f"{split}[{index}]: prompt must contain only one user question")
                question = prompt[0].get("content")
                if not isinstance(question, str) or question_hash(question) != hashes[index]:
                    raise ValueError(f"{split}[{index}]: question does not match manifest")
                if row["data_source"] != DATA_SOURCE or reward.get("style") != "rule":
                    raise ValueError(f"{split}[{index}]: incorrect reward schema")
                if not isinstance(reward.get("ground_truth"), str) or not reward["ground_truth"].strip():
                    raise ValueError(f"{split}[{index}]: ground_truth must be a nonempty string")
                if extra.get("sample_id") != ids[index] or extra.get("split") != split or extra.get("index") != index:
                    raise ValueError(f"{split}[{index}]: incorrect extra_info identity")
                expected_extra = {"sample_id", "split", "index"}
                if split == "train":
                    expected_extra.add("reference_solution")
                    if not isinstance(extra.get("reference_solution"), str) or not extra["reference_solution"].strip():
                        raise ValueError(f"train[{index}]: missing teacher reference_solution")
                if set(extra) != expected_extra:
                    raise ValueError(f"{split}[{index}]: unexpected metadata; held-out teacher information is forbidden")
    return manifest


def generate_demo_jsonl(output_dir: str | Path, *, train_count: int = 128,
                        meta_val_count: int = 32, test_count: int = 32) -> dict[str, Path]:
    """Deterministic, disjoint arithmetic fixtures. Only for plumbing smoke tests."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    paths = {}
    offset = 0
    for split, count in zip(SPLITS, (train_count, meta_val_count, test_count)):
        if count < 1:
            raise ValueError("All demo splits need at least one sample")
        path = destination / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for index in range(count):
                a, b = offset + index + 1, 17 + (offset + index) % 13
                answer = a + b
                row = {
                    "id": f"demo-{split}-{index:05d}",
                    "question": f"Calculate {a} + {b}. Put the final answer in \\boxed{{}}.",
                    "answer": str(answer),
                }
                if split == "train":
                    row["reference_solution"] = f"Adding {a} and {b} gives {answer}. Therefore \\boxed{{{answer}}}."
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        paths[split] = path
        offset += count
    (destination / "DEMO_ONLY.txt").write_text(
        "Synthetic arithmetic fixtures for pipeline smoke tests only. These are not research benchmarks.\n",
        encoding="utf-8",
    )
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train")
    parser.add_argument("--meta-val")
    parser.add_argument("--test")
    parser.add_argument("--output", required=True)
    parser.add_argument("--demo", action="store_true", help="Generate deterministic arithmetic smoke-test data")
    args = parser.parse_args(argv)
    if args.demo:
        if any((args.train, args.meta_val, args.test)):
            parser.error("--demo cannot be combined with input split paths")
        paths = generate_demo_jsonl(Path(args.output) / "raw_demo")
    else:
        if not all((args.train, args.meta_val, args.test)):
            parser.error("--train, --meta-val and --test are required unless --demo is used")
        paths = dict(zip(SPLITS, (args.train, args.meta_val, args.test)))
    manifest = prepare(paths["train"], paths["meta_val"], paths["test"], args.output, demo_only=args.demo)
    validate_manifest(Path(args.output) / "manifest.json")
    print(json.dumps({"manifest": str((Path(args.output) / "manifest.json").resolve()),
                      "counts": {key: value["count"] for key, value in manifest["splits"].items()},
                      "demo_only": manifest["demo_only"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
