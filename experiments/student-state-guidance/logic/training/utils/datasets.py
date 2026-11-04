import json
import os
from typing import Literal, Optional

import numpy as np
import verl.utils.torch_functional as verl_F
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from verl.utils.model import compute_position_id_with_mask

from reasoning_gym.coaching.experiment import Experiment
from reasoning_gym.dataset import ProceduralDataset


class ReasoningGymDataset(Dataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        procedural_dataset: Optional[ProceduralDataset] = None,
        experiment: Optional[Experiment] = None,
        developer_prompt: Optional[str] = None,
        developer_role: str = "system",
        max_prompt_length: int = 2048,
        truncation: str = "error",  ##  ['left', 'right', 'error']
        llama_nemotron_compat: bool = False,
        teacher_chat_template_path: Optional[str] = None,
        enable_thinking: Optional[bool] = None,
        cot_suffix: Optional[str] = None,
    ):
        assert procedural_dataset or experiment, "One of `procedural_dataset` or `experiment` must be provided"
        assert (
            procedural_dataset is None or experiment is None
        ), "Only one of `procedural_dataset` or `experiment` may be provided"

        self.tokenizer = tokenizer
        self.data = procedural_dataset or experiment.composite
        self.experiment = experiment
        self.developer_prompt = developer_prompt
        self.developer_role = developer_role
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation

        # Llama+Nemotron-style prompt adjustments. When llama_nemotron_compat is
        # True we set the system message to "detailed thinking on" and move any
        # pre-existing system content (task instructions) into the first user
        # turn, since Nemotron reserves the system slot for the reasoning
        # toggle. teacher_chat_template_path optionally overrides the
        # tokenizer's chat_template (e.g. to drop the Llama-3.2-Instruct date
        # stamp by borrowing Nemotron's template).
        # Both default to None, i.e. upstream behaviour: the kwarg is not passed
        # and nothing is appended. The paper's runs set enable_thinking=False
        # (Qwen3 non-thinking mode) and cot_suffix="Let's think step by step.\n".
        # The suffix is not cosmetic: without it the teacher tends to emit the
        # answer directly on these tasks rather than reasoning to it, and the
        # reasoning is what on-policy distillation transfers.
        self.enable_thinking = enable_thinking
        self.cot_suffix = cot_suffix
        self.llama_nemotron_compat = llama_nemotron_compat
        self.teacher_chat_template_path = teacher_chat_template_path
        self.override_chat_template = None
        if teacher_chat_template_path:
            cfg_path = os.path.join(teacher_chat_template_path, "tokenizer_config.json")
            with open(cfg_path) as f:
                self.override_chat_template = json.load(f).get("chat_template")
            assert self.override_chat_template is not None, f"No chat_template found in {cfg_path}"

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        row_dict = self.data[index].copy()
        q = row_dict["question"]

        chat = []
        if self.developer_prompt is not None:
            chat.append({"role": self.developer_role, "content": self.developer_prompt})
        chat.append({"role": "user", "content": q})

        if self.llama_nemotron_compat:
            # Nemotron post-training reserves the system slot exclusively for the
            # reasoning toggle ("detailed thinking on/off"); task instructions
            # belong in the user turn, so move any existing system content into
            # the first user message to stay on-distribution.
            task_instr = ""
            if len(chat) > 0 and chat[0].get("role") == "system":
                task_instr = chat[0]["content"]
                chat = chat[1:]
            chat = [{"role": "system", "content": "detailed thinking on"}] + chat
            if task_instr:
                for i, m in enumerate(chat):
                    if m.get("role") == "user":
                        chat[i] = {**m, "content": task_instr.rstrip() + "\n\n" + m["content"]}
                        break

        apply_kwargs = dict(tokenize=False, add_generation_prompt=True)
        if self.enable_thinking is not None:
            apply_kwargs["enable_thinking"] = self.enable_thinking
        if self.override_chat_template is not None:
            apply_kwargs["chat_template"] = self.override_chat_template
        prompt = self.tokenizer.apply_chat_template(chat, **apply_kwargs)
        # Nemotron drives reasoning through its system slot instead, so the
        # suffix is skipped there.
        if self.cot_suffix and not self.llama_nemotron_compat:
            prompt += self.cot_suffix
        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
            prompt=prompt,
            tokenizer=self.tokenizer,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        position_ids = compute_position_id_with_mask(attention_mask)

        row_dict["data_source"] = "reasoning_gym"
        row_dict["input_ids"] = input_ids[0]
        row_dict["attention_mask"] = attention_mask[0]
        row_dict["position_ids"] = position_ids[0]
        row_dict["raw_prompt_ids"] = self.tokenizer.encode(prompt, add_special_tokens=False)
        row_dict["raw_prompt"] = chat
        row_dict["index"] = index
        return row_dict

    def update_experiment_difficulty(self, dataset_name: str, method: Literal["increment", "decrement"]):
        """Update the difficulty of the underlying dataset."""
        if self.experiment is None:
            raise ValueError("Cannot update difficulty: dataset is not a CurriculumExperiment")
        if method not in ["increment", "decrement"]:
            raise ValueError("Invalid method: must be 'increment' or 'decrement'")
        self.experiment.score_board.clear(dataset_name)
        self.experiment.update_difficulty(dataset_name, method)
        self.data = self.experiment.composite
        return True

    def aggregate(self, last_n: Optional[int] = None):
        """Aggregate scores from the underlying experiment"""
        if self.experiment is None:
            raise ValueError("Cannot aggregate scores: dataset is not a CurriculumExperiment")

        results = self.experiment.score_board.aggregate(last_n=last_n)
        output_results = {}

        for key, value in results.items():
            output_results[key] = {}
            scores = value.scores
            first_key = list(scores.keys())[0]
            output_results[key]["results"] = np.mean(scores[first_key])
            output_results[key]["total_samples"] = value.total_scores
        return output_results


def make_dataset(
    tokenizer,
    data_source: Experiment | ProceduralDataset,
    developer_prompt: str,
    max_prompt_length: int = 2048,
    llama_nemotron_compat: bool = False,
    teacher_chat_template_path: Optional[str] = None,
    enable_thinking: Optional[bool] = None,
    cot_suffix: Optional[str] = None,
) -> ReasoningGymDataset:
    """
    Create ReasoningGymDataset object using either a ProceduralDataset or Experiment as the underlying data source.
    """
    common_kwargs = dict(
        tokenizer=tokenizer,
        developer_prompt=developer_prompt,
        developer_role="system",
        max_prompt_length=max_prompt_length,
        truncation="error",
        llama_nemotron_compat=llama_nemotron_compat,
        teacher_chat_template_path=teacher_chat_template_path,
        enable_thinking=enable_thinking,
        cot_suffix=cot_suffix,
    )
    if isinstance(data_source, Experiment):
        return ReasoningGymDataset(experiment=data_source, **common_kwargs)
    else:
        return ReasoningGymDataset(procedural_dataset=data_source, **common_kwargs)
