"""Train an LLM using GRPO over Reasoning Gym procedural dataset(s)."""

from dataclasses import replace
import os

import hydra
import ray
from omegaconf import OmegaConf
from trainers import RayGRPOTrainer
from utils import ReasoningGymDataset, make_dataset

import reasoning_gym
import reasoning_gym.utils
from reasoning_gym.coaching.curriculum_config import CurriculumAttributeConfig, CurriculumExperimentConfig
from reasoning_gym.coaching.experiment import CurriculumExperiment
from reasoning_gym.composite import CompositeDataset, DatasetSpec


def prepare_datasets(config, tokenizer) -> tuple[ReasoningGymDataset, ReasoningGymDataset]:
    """Prepare training and validation datasets."""
    dataset_size = config.reasoning_gym.dataset_size
    developer_prompt_setting = config.reasoning_gym.developer_prompt
    developer_prompt = reasoning_gym.utils.SYSTEM_PROMPTS[developer_prompt_setting]

    if config.data.get('g3_frozen_task'):
        from state_guidance.data import FrozenTaskDataset
        assert not config.curriculum.enabled
        train_data_source = FrozenTaskDataset(config.data.frozen_train_file)
        val_data_source = FrozenTaskDataset(config.data.frozen_validation_file,
            max_examples=config.reasoning_gym.get('val_dataset_size', 512))
    elif config.data.get('frozen_train_file'):
        from guidance.frozen_dataset import FrozenKnightsKnavesDataset
        assert not config.curriculum.enabled
        train_data_source = FrozenKnightsKnavesDataset(config.data.frozen_train_file)
        val_data_source = FrozenKnightsKnavesDataset(config.data.frozen_validation_file,
            max_examples=config.reasoning_gym.get('val_dataset_size', 128))
    elif config.curriculum.enabled:
        curricula = config.curriculum.curricula
        curriculum_config = CurriculumExperimentConfig(
            curricula={
                curriculum_name: CurriculumAttributeConfig(**curriculum_config)
                for curriculum_name, curriculum_config in curricula.items()
            }
        )

        train_data_source = CurriculumExperiment(
            name=config.trainer.experiment_name, config=curriculum_config, size=dataset_size, seed=1
        )
        val_data_source = CompositeDataset(config=replace(train_data_source.composite.config, seed=2))
    else:
        dataset_specs = [
            DatasetSpec(
                name=name,
                weight=ds.weight,
                config=OmegaConf.to_container(ds.config, resolve=True) if "config" in ds else {},
            )
            for name, ds in config.reasoning_gym.datasets.items()
        ]
        train_data_source = reasoning_gym.create_dataset("composite", seed=1, size=dataset_size, datasets=dataset_specs)
        # Upstream sizes the validation split like the training split; the paper
        # uses a much smaller fixed one so validation is affordable every few steps.
        val_dataset_size = config.reasoning_gym.get("val_dataset_size", dataset_size)
        val_data_source = reasoning_gym.create_dataset(
            "composite", seed=2, size=val_dataset_size, datasets=dataset_specs)
    llama_nemotron_compat = bool(config.data.get("llama_nemotron_compat", False))
    teacher_chat_template_path = config.data.get("teacher_chat_template_path", None)
    enable_thinking = config.data.get("enable_thinking", None)
    cot_suffix = config.data.get("cot_suffix", None)

    train_dataset = make_dataset(
        tokenizer,
        train_data_source,
        developer_prompt,
        max_prompt_length=config.data.max_prompt_length,
        llama_nemotron_compat=llama_nemotron_compat,
        teacher_chat_template_path=teacher_chat_template_path,
        enable_thinking=enable_thinking,
        cot_suffix=cot_suffix,
    )
    val_dataset = make_dataset(
        tokenizer,
        val_data_source,
        developer_prompt,
        max_prompt_length=config.data.max_prompt_length,
        llama_nemotron_compat=llama_nemotron_compat,
        teacher_chat_template_path=teacher_chat_template_path,
        enable_thinking=enable_thinking,
        cot_suffix=cot_suffix,
    )
    return train_dataset, val_dataset


@ray.remote
def main_task(config):
    if config.data.get('g3_frozen_task'):
        from state_guidance.runtime_paths import enter_task_directory
        enter_task_directory(config)
    from pprint import pprint

    from verl.utils import hf_tokenizer
    from verl.utils.fs import copy_local_path_from_hdfs

    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    # download the checkpoint from hdfs
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

    # instantiate tokenizer
    tokenizer = hf_tokenizer(local_path)

    # define worker classes
    if config.actor_rollout_ref.actor.strategy == "fsdp":
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.single_controller.ray import RayWorkerGroup
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker

        ray_worker_group_cls = RayWorkerGroup
    elif config.actor_rollout_ref.actor.strategy == "megatron":
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker

        ray_worker_group_cls = NVMegatronRayWorkerGroup
    else:
        raise NotImplementedError

    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker),
    }

    global_pool_id = "global_pool"
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
        Role.RefPolicy: global_pool_id,
    }

    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

    train_dataset, val_dataset = prepare_datasets(config, tokenizer)

    trainer = RayGRPOTrainer(
        config=config,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        role_worker_mapping=role_worker_mapping,
        resource_pool_manager=resource_pool_manager,
        ray_worker_group_cls=ray_worker_group_cls,
        max_output_length=config.data.max_response_length,
    )
    trainer.init_workers()
    trainer.fit()


@hydra.main(config_path="configs", config_name="llama3.1_1b_grpo", version_base=None)
def main(config):
    if config.data.get('g3_frozen_task'):
        from state_guidance.runtime_paths import enter_driver_directory
        enter_driver_directory(config, __file__)
    if not ray.is_initialized():
        # this is for local ray cluster
        ray_options = {}
        if os.environ.get("REPRO_RAY_ISOLATE") == "1":
            # Concurrent replicas must not attach to another job's Ray cluster.
            ray_options = {
                "address": "local",
                "_temp_dir": os.environ["RAY_TMPDIR"],
                "num_gpus": int(os.environ["REPRO_RAY_NUM_GPUS"]),
                "num_cpus": int(os.environ["REPRO_RAY_NUM_CPUS"]),
                "include_dashboard": False,
            }
            if os.environ.get("REPRO_RAY_PLASMA_DIR"):
                # Each lane uses a private local mapping, not the container-wide /dev/shm.
                # Keep the existing one-GiB object-store allowance and sampling unchanged.
                ray_options["_plasma_directory"] = os.environ["REPRO_RAY_PLASMA_DIR"]
                ray_options["object_store_memory"] = 1073741824
        ray.init(runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN"}},
                 **ray_options)
    ray.get(main_task.remote(config))


if __name__ == "__main__":
    main()
