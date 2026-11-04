import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from meta_teaching.data import (
    build_rows, generate_demo_jsonl, load_jsonl, prepare, sha256_file,
    validate_manifest, validate_splits,
)
from mvp_tools.math_reward import compute_numeric_demo_score, compute_score

HAS_ARROW = importlib.util.find_spec("pyarrow") is not None


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.paths = generate_demo_jsonl(self.root / "raw", train_count=4, meta_val_count=2, test_count=2)

    def tearDown(self):
        self.temp.cleanup()

    def records(self):
        return {split: load_jsonl(path, split) for split, path in self.paths.items()}

    def test_teacher_reference_never_enters_student_prompt(self):
        records = self.records()
        row = build_rows(records["train"], "train")[0]
        self.assertEqual(row["prompt"], [{"role": "user", "content": records["train"][0]["question"]}])
        self.assertNotIn(records["train"][0]["reference_solution"], json.dumps(row["prompt"]))
        self.assertEqual(row["extra_info"]["reference_solution"], records["train"][0]["reference_solution"])
        self.assertEqual(row["reward_model"], {"style": "rule", "ground_truth": records["train"][0]["answer"]})
        self.assertEqual(row["data_source"], "meta_teaching_math")
        self.assertEqual(set(row), {"prompt", "data_source", "ability", "extra_info", "reward_model"})

    def test_held_out_reference_is_discarded(self):
        self.paths["meta_val"].write_text(json.dumps({"question": "What is 1 + 2?", "answer": 3,
                                                   "reference_solution": "PRIVATE TEACHER INFO"}) + "\n")
        records = load_jsonl(self.paths["meta_val"], "meta_val")
        self.assertNotIn("reference_solution", records[0])
        rows = build_rows(records, "meta_val")
        self.assertNotIn("reference_solution", rows[0]["extra_info"])
        self.assertNotIn("PRIVATE TEACHER INFO", json.dumps(rows))

    def test_duplicate_normalized_question_is_rejected(self):
        records = self.records()
        records["meta_val"][0]["question"] = "  " + records["train"][0]["question"].upper().replace(" ", "\n ")
        with self.assertRaisesRegex(ValueError, "normalized question"):
            validate_splits(records)

    def test_duplicate_id_is_rejected(self):
        records = self.records()
        records["test"][0]["id"] = records["train"][0]["id"]
        with self.assertRaisesRegex(ValueError, "sample id"):
            validate_splits(records)

    def test_demo_is_reproducible_and_disjoint(self):
        before = {key: path.read_bytes() for key, path in self.paths.items()}
        generate_demo_jsonl(self.root / "raw", train_count=4, meta_val_count=2, test_count=2)
        self.assertEqual(before, {key: path.read_bytes() for key, path in self.paths.items()})
        validate_splits(self.records())

    def test_train_requires_reference_and_numeric_answer_is_string(self):
        self.paths["train"].write_text(json.dumps({"prompt": "p", "answer": 3}) + "\n")
        with self.assertRaisesRegex(ValueError, "reference_solution"):
            load_jsonl(self.paths["train"], "train")
        row = {"prompt": "p", "answer": 3, "reference_solution": "r"}
        self.paths["train"].write_text(json.dumps(row) + "\n")
        result = load_jsonl(self.paths["train"], "train")[0]
        self.assertEqual(result["answer"], "3")
        self.assertEqual(len(result["id"]), 64)
        self.assertEqual(result["id"], load_jsonl(self.paths["train"], "train")[0]["id"])

    def test_manifest_overlap_rejected_without_arrow(self):
        destination = self.root / "prepared"
        destination.mkdir()
        manifest = {"schema_version": 1, "data_source": "meta_teaching_math", "splits": {}}
        for split, records in self.records().items():
            path = destination / f"{split}.parquet"
            path.write_bytes(b"fake parquet: only testing file hashes and manifest")
            manifest["splits"][split] = {"path": path.name, "sha256": sha256_file(path),
                                         "count": 1, "sample_ids": [split], "question_hashes": ["same"]}
        path = destination / "manifest.json"
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "overlapping"):
            validate_manifest(path, check_parquet=False)

    @unittest.skipUnless(HAS_ARROW, "pyarrow required for Parquet integration")
    def test_parquet_roundtrip_and_hash_tampering(self):
        destination = self.root / "prepared"
        manifest = prepare(self.paths["train"], self.paths["meta_val"], self.paths["test"], destination, demo_only=True)
        checked = validate_manifest(destination / "manifest.json")
        self.assertEqual(manifest, checked)
        self.assertEqual(manifest["splits"]["train"]["count"], 4)
        self.assertTrue(manifest["demo_only"])
        (destination / "test.parquet").write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            validate_manifest(destination / "manifest.json")

    @unittest.skipUnless(HAS_ARROW, "pyarrow required for Parquet integration")
    def test_split_alias_is_rejected(self):
        destination = self.root / "prepared"
        manifest = prepare(self.paths["train"], self.paths["meta_val"], self.paths["test"], destination)
        manifest["splits"]["test"]["path"] = manifest["splits"]["meta_val"]["path"]
        manifest["splits"]["test"]["sha256"] = manifest["splits"]["meta_val"]["sha256"]
        (destination / "manifest.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "alias the same"):
            validate_manifest(destination / "manifest.json", check_parquet=False)

    @unittest.skipUnless(HAS_ARROW, "pyarrow required for Parquet integration")
    def test_actual_held_out_schema_is_checked(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        destination = self.root / "prepared"
        manifest = prepare(self.paths["train"], self.paths["meta_val"], self.paths["test"], destination)
        path = destination / "test.parquet"
        rows = pq.read_table(path).to_pylist()
        for row in rows:
            row["extra_info"]["reference_solution"] = "leaked"
        pq.write_table(pa.Table.from_pylist(rows), path)
        manifest["splits"]["test"]["sha256"] = sha256_file(path)
        (destination / "manifest.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "held-out teacher information"):
            validate_manifest(destination / "manifest.json")


class RewardTests(unittest.TestCase):
    def test_numeric_demo_requires_boxed_final_answer(self):
        self.assertEqual(compute_numeric_demo_score("x", "work 3, final \\boxed{4}", "4")["score"], 1.0)
        self.assertEqual(compute_numeric_demo_score("x", "4", "4")["score"], 0.0)
        self.assertEqual(compute_numeric_demo_score("x", "\\boxed{nan}", "NaN")["score"], 0.0)
        self.assertEqual(compute_numeric_demo_score("x", "\\boxed{3} \\boxed{4}", "3")["score"], 0.0)

    def test_production_never_silently_falls_back(self):
        with patch("mvp_tools.math_reward.importlib.import_module", side_effect=ImportError("missing")):
            with self.assertRaisesRegex(RuntimeError, "never selected automatically"):
                compute_score("x", "\\boxed{4}", "4")


if __name__ == "__main__":
    unittest.main()
