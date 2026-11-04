"""Real local Parquet -> RLHFDataset -> collate -> DataProto -> teacher hook.

No Ray initialization, GPU operation, model download, or fake dataset loader is
used. Only response IDs are supplied as a fixture in place of model generation.
"""
import copy
import json
from pathlib import Path
import sys

import pytest

# install-cpu intentionally omits this training-side dependency stack.
torch = pytest.importorskip("torch", reason="Real dataset integration needs the training dependency stack")
np = pytest.importorskip("numpy")
pytest.importorskip("pyarrow")
datasets = pytest.importorskip("datasets")
pytest.importorskip("ray")
pytest.importorskip("tensordict")
pytest.importorskip("accelerate")
pytest.importorskip("peft")
pytest.importorskip("transformers")
pytest.importorskip("tokenizers")
from omegaconf import OmegaConf
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import AutoTokenizer, PreTrainedTokenizerFast

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "verl"))
from verl import DataProto
from verl.trainer.ppo.meta_teaching_hooks import prepare_teacher_inputs
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
from meta_teaching.data import prepare, validate_manifest


@pytest.fixture
def local_pipeline(tmp_path, monkeypatch):
    # Local files only. Confine Arrow/HF cache artifacts to this test directory.
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    monkeypatch.setattr(datasets.config, "HF_DATASETS_OFFLINE", True)
    monkeypatch.setattr(datasets.config, "HF_DATASETS_CACHE", tmp_path / "hf_datasets_cache")
    monkeypatch.setattr(datasets.config, "HF_MODULES_CACHE", tmp_path / "hf_modules_cache")
    vocabulary = {
        "[UNK]": 0, "[PAD]": 1, "[EOS]": 2, "user": 3, "assistant": 4,
        "Compute": 5, "2": 6, "3": 7, "4": 8, "5": 9, "6": 10,
        "7": 11, "8": 12, "+": 13, ".": 14, "?": 15,
        "SECRETREF_A": 16, "SECRETREF_B": 17, "Reasoning": 18,
    }
    core = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    core.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=core, unk_token="[UNK]", pad_token="[PAD]", eos_token="[EOS]",
    )
    tokenizer.chat_template = (
        "{% for message in messages %}{{ message['role'] }} {{ message['content'] }} "
        "{% endfor %}{% if add_generation_prompt %}assistant{% endif %}"
    )
    tokenizer_path = tmp_path / "tiny_local_tokenizer"
    tokenizer.save_pretrained(tokenizer_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    rows = {
        "train": [
            {"id": "train-a", "question": "Compute 2 + 2.", "answer": "4", "reference_solution": "SECRETREF_A Reasoning 2 + 2 4."},
            {"id": "train-b", "question": "Compute 2 + 3.", "answer": "5", "reference_solution": "SECRETREF_B Reasoning 2 + 3 5."},
        ],
        "meta_val": [{"id": "meta-a", "question": "Compute 3 + 3.", "answer": "6", "reference_solution": "FORBIDDEN_META_REFERENCE"}],
        "test": [{"id": "test-a", "question": "Compute 4 + 4.", "answer": "8", "reference_solution": "FORBIDDEN_TEST_REFERENCE"}],
    }
    paths = {}
    for name, records in rows.items():
        path = tmp_path / f"{name}.jsonl"
        path.write_text("".join(json.dumps(record) + "\n" for record in records))
        paths[name] = path
    prepared = tmp_path / "prepared"
    prepare(paths["train"], paths["meta_val"], paths["test"], prepared)
    manifest = validate_manifest(prepared / "manifest.json")
    config = OmegaConf.create({
        "cache_dir": str(tmp_path / "verl_cache"), "prompt_key": "prompt", "max_prompt_length": 24,
        "return_raw_chat": True, "filter_overlong_prompts": True, "filter_overlong_prompts_workers": 1,
        "truncation": "error", "shuffle": False, "seed": 42,
        "apply_chat_template_kwargs": {"enable_thinking": False},
    })
    result = {}
    for name in ("train", "meta_val", "test"):
        path = prepared / manifest["splits"][name]["path"]
        actual_dataset = RLHFDataset(str(path), tokenizer, config)
        result[name] = DataProto.from_single_dict(collate_fn([actual_dataset[index] for index in range(len(actual_dataset))]))
    return tokenizer, result, rows


def with_response_fixture(batch, tokenizer):
    """Create sampled-response-shaped tensors without invoking a model."""
    count = len(batch)
    answers = [tokenizer.convert_tokens_to_ids(item["ground_truth"]) for item in batch.non_tensor_batch["reward_model"]]
    responses = torch.tensor([[answer, tokenizer.eos_token_id, tokenizer.pad_token_id] for answer in answers], dtype=torch.long)
    attention = torch.tensor([[1, 1, 0] if index == 0 else [1, 0, 0] for index in range(count)], dtype=torch.long)
    batch.batch["responses"] = responses
    batch.batch["input_ids"] = torch.cat((batch.batch["input_ids"], responses), dim=-1)
    batch.batch["attention_mask"] = torch.cat((batch.batch["attention_mask"], attention), dim=-1)
    batch.batch["position_ids"] = batch.batch["attention_mask"].cumsum(-1).sub(1).clamp_min(0)
    # Deliberately differs from attention: teacher must retain these context tokens.
    batch.batch["response_mask"] = torch.zeros_like(attention)
    return batch, responses, attention


def test_actual_training_dataset_preserves_metadata_without_student_leak(local_pipeline):
    tokenizer, batches, source = local_pipeline
    batch = batches["train"]
    assert batch.batch["input_ids"].shape == (2, 24)
    assert len(batch.non_tensor_batch["raw_prompt"]) == 2
    for index, record in enumerate(source["train"]):
        assert list(batch.non_tensor_batch["raw_prompt"][index]) == [{"role": "user", "content": record["question"]}]
        assert batch.non_tensor_batch["extra_info"][index]["reference_solution"] == record["reference_solution"]
        assert batch.non_tensor_batch["reward_model"][index] == {"style": "rule", "ground_truth": record["answer"]}
        assert batch.non_tensor_batch["data_source"][index] == "meta_teaching_math"
    reference_ids = {tokenizer.convert_tokens_to_ids("SECRETREF_A"), tokenizer.convert_tokens_to_ids("SECRETREF_B")}
    assert not reference_ids.intersection(batch.batch["input_ids"].flatten().tolist())
    assert "SECRETREF" not in tokenizer.batch_decode(batch.batch["input_ids"])[0]


def test_actual_batch_reaches_teacher_hook_with_exact_response_alignment(local_pipeline):
    tokenizer, batches, source = local_pipeline
    batch, responses, response_attention = with_response_fixture(batches["train"], tokenizer)
    original_student_ids = batch.batch["input_ids"].clone()
    original_student_attention = batch.batch["attention_mask"].clone()
    original_raw = copy.deepcopy(batch.non_tensor_batch["raw_prompt"])
    prepare_teacher_inputs(batch, tokenizer, {"exposure": "full", "teacher_max_prompt_length": 256})
    assert torch.equal(batch.batch["input_ids"], original_student_ids)
    assert torch.equal(batch.batch["attention_mask"], original_student_attention)
    assert batch.non_tensor_batch["raw_prompt"].tolist() == original_raw.tolist()
    teacher_ids = batch.batch["ref_input_ids"]
    assert teacher_ids.shape[0] == 2
    assert teacher_ids.shape == batch.batch["ref_attention_mask"].shape == batch.batch["ref_position_ids"].shape
    assert torch.equal(teacher_ids[:, -3:], responses)
    assert torch.equal(batch.batch["ref_attention_mask"][:, -3:], response_attention)
    for index, marker in enumerate(("SECRETREF_A", "SECRETREF_B")):
        assert tokenizer.convert_tokens_to_ids(marker) in teacher_ids[index, :-3].tolist()
    assert batch.batch["ref_attention_mask"][:, -3:].sum().item() > batch.batch["response_mask"].sum().item()


@pytest.mark.parametrize("split", ["meta_val", "test"])
def test_actual_held_out_batches_have_no_teacher_reference(local_pipeline, split):
    tokenizer, batches, source = local_pipeline
    batch = batches[split]
    assert all("reference_solution" not in extra for extra in batch.non_tensor_batch["extra_info"])
    assert "FORBIDDEN" not in repr(batch.non_tensor_batch)
    assert batch.non_tensor_batch["reward_model"][0]["ground_truth"] == source[split][0]["answer"]
    batch, _, _ = with_response_fixture(batch, tokenizer)
    with pytest.raises(ValueError, match="reference_solution"):
        prepare_teacher_inputs(batch, tokenizer, {"exposure": "full", "teacher_max_prompt_length": 256})
