import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from meta_teaching.preflight import compare_tokenizers, inspect_local_model, run_preflight


def make_model(root, *, added_id=2, added_content="<eos>", vocab=None, merges=None):
    root.mkdir(parents=True)
    (root / "config.json").write_text(json.dumps({"model_type": "same_family", "vocab_size": 8}))
    (root / "tokenizer.json").write_text(json.dumps({
        "model": {"type": "BPE", "vocab": vocab or {"a": 0, "b": 1}, "merges": merges or []},
        "added_tokens": [{"id": added_id, "content": added_content, "special": True}],
        "normalizer": None, "pre_tokenizer": {"type": "Whitespace"},
    }))
    (root / "tokenizer_config.json").write_text(json.dumps({"chat_template": "{{ messages }}"}))
    (root / "model.safetensors").write_bytes(b"mock: never loaded by metadata preflight")
    return root


class ModelChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_identical_tokenizers_compatible(self):
        student = make_model(self.root / "student")
        teacher = make_model(self.root / "teacher")
        report = compare_tokenizers(student, teacher)
        self.assertEqual(report["student"]["tokenizer_sha256"], report["teacher"]["tokenizer_sha256"])

    def test_same_family_but_different_ids_rejected(self):
        student = make_model(self.root / "student")
        teacher = make_model(self.root / "teacher", vocab={"a": 1, "b": 0})
        with self.assertRaisesRegex(ValueError, "semantics differ"):
            compare_tokenizers(student, teacher)

    def test_added_token_mismatch_rejected(self):
        student = make_model(self.root / "student")
        teacher = make_model(self.root / "teacher", added_content="<other>")
        with self.assertRaisesRegex(ValueError, "semantics differ"):
            compare_tokenizers(student, teacher)

    def test_missing_shard_rejected(self):
        student = make_model(self.root / "student")
        (student / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"x": "missing.safetensors"}}))
        with self.assertRaisesRegex(ValueError, "Missing model shards"):
            inspect_local_model(student)

    def test_missing_path_rejected(self):
        with self.assertRaisesRegex(ValueError, "does not exist"):
            inspect_local_model(self.root / "missing")

    def test_absent_template_rejected(self):
        student = make_model(self.root / "student")
        (student / "tokenizer_config.json").unlink()
        with self.assertRaisesRegex(ValueError, "chat template"):
            inspect_local_model(student)

    def test_hf_ids_not_reported_verified(self):
        config = {"models": {"student": "org/student", "teacher": "org/teacher"},
                  "data": {"manifest": "does-not-exist.json"}}
        with patch("meta_teaching.preflight._package_versions", return_value={}):
            report = run_preflight(config, self.root)
        checks = {check["name"]: check for check in report["checks"]}
        self.assertEqual(checks["model.student"]["status"], "warn")
        self.assertIn("not_locally_verified", checks["model.student"]["detail"])
        self.assertEqual(checks["data"]["status"], "fail")
        self.assertFalse(report["ok"])

    def test_numeric_reward_rejected_for_real_data(self):
        student = make_model(self.root / "student")
        config = {"models": {"student": str(student), "teacher": str(student)},
                  "data": {"manifest": "manifest.json", "reward": "numeric_demo"}}
        with patch("meta_teaching.preflight.validate_manifest", return_value={"demo_only": False}), \
             patch("meta_teaching.preflight._package_versions", return_value={}):
            report = run_preflight(config, self.root)
        reward_check = next(item for item in report["checks"] if item["name"] == "reward")
        self.assertEqual(reward_check["status"], "fail")

    def test_cpu_preflight_never_probes_cuda(self):
        with patch("meta_teaching.preflight._gpu_checks", side_effect=AssertionError("unexpected CUDA probe")), \
             patch("meta_teaching.preflight._package_versions", return_value={}):
            run_preflight({}, self.root, require_gpu=False)


if __name__ == "__main__":
    unittest.main()
