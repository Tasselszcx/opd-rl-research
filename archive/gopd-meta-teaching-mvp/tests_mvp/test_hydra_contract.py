"""Exercise generated argv against the real pinned Hydra and dataclass configs.

No Ray cluster, model download, GPU context or model construction is performed.
"""
import copy
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "verl"))

from meta_teaching.backend import PhaseSpec, build_phase_command
from meta_teaching.config import DEFAULTS, hydra_value, validate_config

HAS_HYDRA = importlib.util.find_spec("hydra") is not None
HAS_LOCAL_TOKENIZERS = all(importlib.util.find_spec(name) is not None for name in ("transformers", "tokenizers"))
HAS_TRAINING_CONFIG = all(importlib.util.find_spec(name) is not None for name in ("torch", "transformers", "ray", "tensordict", "numpy"))
if HAS_HYDRA:
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    # The hooks are intentionally dependency-light. Import their actual source
    # directly, avoiding verl.__init__ (DataProto -> torch/ray/tensordict).
    _hooks_spec = importlib.util.spec_from_file_location(
        "mvp_hydra_contract_hooks", ROOT / "verl/verl/trainer/ppo/meta_teaching_hooks.py")
    _hooks = importlib.util.module_from_spec(_hooks_spec)
    _hooks_spec.loader.exec_module(_hooks)
    assert_same_tokenizer = _hooks.assert_same_tokenizer
    validate_phase_config = _hooks.validate_phase_config


@unittest.skipUnless(HAS_HYDRA, "hydra-core required for real configuration composition")
class HydraContractTests(unittest.TestCase):
    def setUp(self):
        self.config = copy.deepcopy(DEFAULTS)
        self.config["models"] = {"student": "/tmp/student", "teacher": "/tmp/teacher"}
        validate_config(self.config)

    def spec(self, phase="warmup"):
        return PhaseSpec(
            phase_id=f"round0/{phase}", phase=phase, arm_id="full", output_dir=f"/tmp/run/{phase}",
            student_model="/tmp/student", teacher_model="/tmp/teacher" if phase == "warmup" else None,
            resume_checkpoint=None, train_path="/tmp/train.parquet", validation_path="/tmp/meta_val.parquet",
            steps=5, seed=42, exposure="full",
        )

    def composed(self, spec):
        argv = build_phase_command(self.config, spec, repository_root=ROOT)
        self.assertEqual(argv[1:3], ["-m", "verl.trainer.main_ppo"])
        with initialize_config_dir(config_dir=str(ROOT / "verl/verl/trainer/config"), version_base=None):
            config = compose(config_name="ppo_trainer", overrides=argv[3:])
            OmegaConf.resolve(config)
        validate_phase_config(config)
        return config

    def test_all_phases_compose_with_real_backend_guards(self):
        for phase in ("warmup", "rl", "eval"):
            with self.subTest(phase=phase):
                config = self.composed(self.spec(phase))
                self.assertEqual(config.meta_teaching.phase, phase)
                self.assertEqual(config.actor_rollout_ref.model.lora_rank, 0)
                self.assertFalse(config.actor_rollout_ref.rollout.multi_turn.enable)
                self.assertEqual(config.actor_rollout_ref.rollout.mode, "sync")
                self.assertEqual(config.actor_rollout_ref.actor.optim.lr_scheduler_type, "constant")
                self.assertEqual(config.trainer.val_only, phase == "eval")

    def test_independent_rl_and_eval_have_no_teacher_request(self):
        for phase in ("rl", "eval"):
            config = self.composed(self.spec(phase))
            self.assertIsNone(config.actor_rollout_ref.ref.model.path)
            self.assertIsNone(config.actor_rollout_ref.ref.model.base_model_path)
            self.assertIsNone(config.actor_rollout_ref.model.base_model_path)
            self.assertFalse(config.actor_rollout_ref.actor.use_kl_loss)
            self.assertFalse(config.algorithm.use_kl_in_reward)
            self.assertFalse(config.actor_rollout_ref.actor.policy_loss.only_reverse_kl_advantages)
            self.assertFalse(config.algorithm.use_ref_solution_distillation)
            self.assertIsNone(config.algorithm.critique_vllm_url)
            self.assertFalse(config.reward_model.enable)

    def test_warmup_single_rollout_has_reverse_kl_supervision(self):
        config = self.composed(self.spec())
        self.assertEqual(config.actor_rollout_ref.rollout.n, 1)
        self.assertEqual(config.algorithm.adv_estimator, "grpo")
        self.assertTrue(config.actor_rollout_ref.actor.policy_loss.only_reverse_kl_advantages)
        self.assertTrue(config.actor_rollout_ref.actor.use_kl_loss)
        self.assertTrue(config.actor_rollout_ref.actor.use_rollout_log_probs)
        self.assertEqual(config.actor_rollout_ref.actor.policy_loss.lambda_vals, 1)

    def test_teacher_in_rl_spec_rejected_before_launch(self):
        spec = replace(self.spec("rl"), teacher_model="/tmp/teacher")
        with self.assertRaisesRegex(ValueError, "must not receive"):
            build_phase_command(self.config, spec)

    def test_spaces_punctuation_and_unicode_roundtrip(self):
        path = '/tmp/space, [list] (=value):中文 "double" and \'single\' $cash'
        spec = replace(self.spec(), student_model=path, teacher_model=path, output_dir=path,
                       train_path=path + "/train.parquet", validation_path=path + "/val.parquet")
        config = self.composed(spec)
        self.assertEqual(config.actor_rollout_ref.model.path, path)
        self.assertEqual(config.actor_rollout_ref.ref.model.path, path)
        self.assertEqual(config.meta_teaching.output_dir, path)
        self.assertEqual(list(config.data.train_files), [spec.train_path])
        self.assertEqual(list(config.data.val_files), [spec.validation_path])

    def test_backslashes_roundtrip(self):
        for suffix in [r"back\slash", "trailing\\", 'before\\"quote', 'double\\\\backslash']:
            with self.subTest(suffix=suffix):
                path = "/tmp/" + suffix
                spec = replace(self.spec(), student_model=path, train_path=path)
                config = self.composed(spec)
                self.assertEqual(config.actor_rollout_ref.model.path, path)
                self.assertEqual(config.data.train_files[0], path)

    def test_interpolation_or_control_chars_fail_explicitly(self):
        for path in ["/tmp/${unexpanded}", "/tmp/new\nline", "/tmp/tab\tname"]:
            with self.subTest(path=path):
                with self.assertRaisesRegex(ValueError, "interpolation expressions or control characters"):
                    hydra_value(path)

    @unittest.skipUnless(HAS_TRAINING_CONFIG, "torch and transformers needed for real upstream dataclasses")
    def test_real_dataclass_instantiation_and_upstream_validation(self):
        from verl.utils.config import omega_conf_to_dataclass, validate_config as validate_upstream
        for phase in ("warmup", "rl", "eval"):
            with self.subTest(phase=phase):
                config = self.composed(self.spec(phase))
                actor = omega_conf_to_dataclass(config.actor_rollout_ref.actor)
                rollout = omega_conf_to_dataclass(config.actor_rollout_ref.rollout)
                algorithm = omega_conf_to_dataclass(config.algorithm)
                self.assertEqual(rollout.n, 1 if phase == "warmup" else 4)
                # Legacy FSDP normalizes actor mini-batches with config.rollout.n;
                # ActorConfig.rollout_n is a newer roles/actor path, unused here.
                local_mini = actor.ppo_mini_batch_size * rollout.n // config.trainer.n_gpus_per_node
                self.assertEqual(local_mini % actor.ppo_micro_batch_size_per_gpu, 0)
                self.assertEqual(algorithm.adv_estimator, "grpo")
                validate_upstream(config, use_reference_policy=phase == "warmup", use_critic=False)

    @unittest.skipUnless(HAS_LOCAL_TOKENIZERS, "transformers/tokenizers needed for local tokenizer roundtrip")
    def test_real_local_tokenizer_roundtrip_matches_preflight(self):
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import AutoTokenizer, PreTrainedTokenizerFast
        from meta_teaching.preflight import compare_tokenizers
        backend = Tokenizer(WordLevel({"[UNK]": 0, "[EOS]": 1, "hello": 2, "world": 3}, unk_token="[UNK]"))
        backend.pre_tokenizer = Whitespace()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", eos_token="[EOS]", pad_token="[EOS]")
        tokenizer.chat_template = "{% for message in messages %}{{ message['content'] }}{% endfor %}"
        with tempfile.TemporaryDirectory() as temporary:
            student, teacher = Path(temporary) / "student", Path(temporary) / "teacher"
            for folder in (student, teacher):
                tokenizer.save_pretrained(folder)
                (folder / "config.json").write_text(json.dumps({"model_type": "qwen2", "vocab_size": 4}))
                (folder / "model.safetensors").write_bytes(b"metadata-only fixture")
            compare_tokenizers(student, teacher)
            student_tok = AutoTokenizer.from_pretrained(student, local_files_only=True)
            teacher_tok = AutoTokenizer.from_pretrained(teacher, local_files_only=True)
            assert_same_tokenizer(student_tok, teacher_tok)
            data = json.loads((teacher / "tokenizer.json").read_text())
            data["model"]["vocab"]["hello"], data["model"]["vocab"]["world"] = 3, 2
            (teacher / "tokenizer.json").write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "semantics differ"):
                compare_tokenizers(student, teacher)
            modified = AutoTokenizer.from_pretrained(teacher, local_files_only=True)
            with self.assertRaisesRegex(ValueError, "identical student/teacher token vocabularies"):
                assert_same_tokenizer(student_tok, modified)

    def test_backend_rejects_same_family_different_vocab(self):
        class Tokenizer:
            bos_token_id = 0
            eos_token_id = 1
            pad_token_id = 1
            chat_template = "same"
            def __init__(self, vocab):
                self.vocab = vocab
            def get_vocab(self):
                return self.vocab
        assert_same_tokenizer(Tokenizer({"a": 0, "b": 1}), Tokenizer({"a": 0, "b": 1}))
        with self.assertRaisesRegex(ValueError, "identical student/teacher token vocabularies"):
            assert_same_tokenizer(Tokenizer({"a": 0, "b": 1}), Tokenizer({"a": 1, "b": 0}))


if __name__ == "__main__":
    unittest.main()
