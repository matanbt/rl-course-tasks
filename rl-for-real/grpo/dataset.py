"""Dataset wrapper and preparation for reasoning-gym."""

import random
from typing import Optional

from torch.utils.data import Dataset

import reasoning_gym
from reasoning_gym.composite import DatasetSpec
from reasoning_gym.dataset import ProceduralDataset

from .config import DatasetConfig
from .prompts import SYSTEM_PROMPTS

# Appended to some training questions of the tasks in DatasetConfig.train_format_hint_tasks. Never added to eval prompts.
FORMAT_HINT = ("Use the required format even though the question asks for a short answer: "
               "reason inside <think> </think>, then give only the final answer inside <answer> </answer>.")


class ReasoningGymDataset(Dataset):
    """
    Dataset wrapper for reasoning-gym procedural datasets.
    
    Formats prompts with chat templates for use with GRPOTrainer.
    """
    
    def __init__(
        self,
        tokenizer,
        procedural_dataset: ProceduralDataset,
        developer_prompt: Optional[str] = None,
        developer_role: Optional[str] = None,
        format_hint_tasks: Optional[list[str]] = None,
        format_hint_ratio: float = 0.0,
    ):
        self.tokenizer = tokenizer
        self.data = procedural_dataset
        self.developer_prompt = developer_prompt
        self.developer_role = developer_role
        self.format_hint_tasks = set(format_hint_tasks or [])
        self.format_hint_ratio = format_hint_ratio

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        item = self.data[idx]
        question = item["question"]
        # Some questions ask for a bare answer ("Respond only with the word ..."), so the model skips the <answer> tags,
        # every rollout scores 0 and the group gives no learning signal. A fixed share of those tasks' prompts (chosen
        # by index) get FORMAT_HINT; the rest stay exactly as in eval, where the behavior has to carry over.
        task = item.get("metadata", {}).get("source_dataset")
        if task in self.format_hint_tasks and random.Random(idx).random() < self.format_hint_ratio:
            question = f"{question}\n\n{FORMAT_HINT}"

        chat = []
        if self.developer_role is not None:
            chat.append({"role": self.developer_role, "content": self.developer_prompt})
        chat.append({"role": "user", "content": question})

        prompt = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        return {"prompt": prompt, "item": item}


def prepare_datasets(
    config: DatasetConfig,
    tokenizer,
) -> tuple[ReasoningGymDataset, ReasoningGymDataset]:
    """Prepare training and evaluation datasets from config."""
    developer_prompt = SYSTEM_PROMPTS[config.developer_prompt]

    # Training dataset specs
    train_dataset_specs = [
        DatasetSpec(
            name=name,
            weight=ds_config.weight,
            config=ds_config.config,
        )
        for name, ds_config in config.datasets.items()
    ]
    
    # Eval dataset specs: use eval_datasets if provided, otherwise use same as training
    if config.eval_datasets:
        eval_dataset_specs = [
            DatasetSpec(
                name=name,
                weight=ds_config.weight,
                config=ds_config.config,
            )
            for name, ds_config in config.eval_datasets.items()
        ]
    else:
        eval_dataset_specs = train_dataset_specs
    
    train_data = reasoning_gym.create_dataset(
        "composite", seed=1, size=config.dataset_size, datasets=train_dataset_specs
    )
    eval_data = reasoning_gym.create_dataset(
        "composite", seed=2, size=config.eval_num_samples, datasets=eval_dataset_specs
    )
    
    train_dataset = ReasoningGymDataset(
        tokenizer=tokenizer,
        procedural_dataset=train_data,
        developer_prompt=developer_prompt,
        developer_role=config.developer_role,
        format_hint_tasks=config.train_format_hint_tasks,
        format_hint_ratio=config.train_format_hint_ratio,
    )
    eval_dataset = ReasoningGymDataset(
        tokenizer=tokenizer,
        procedural_dataset=eval_data,
        developer_prompt=developer_prompt,
        developer_role=config.developer_role,
    )
    
    return train_dataset, eval_dataset
