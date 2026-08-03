"""
@article{zhang2025guidedbench,
  title={GuidedBench: Equipping Jailbreak Evaluation with Guidelines},
  author={Zhang, Ruixuan and others},
  journal={arXiv preprint arXiv:2502.16903},
  year={2025}
}
"""
from dataclasses import dataclass

import pandas as pd

from ..types import Conversation
from .prompt_dataset import PromptDataset


@dataclass
class GuidedBenchConfig:
    name: str = "guidedbench"
    path: str = "${root_dir}/../GuidedBench/Dataset/GuidedBench.csv"
    seed: int = 0
    idx: list[int] | int | str | None = None
    shuffle: bool = False
    multirun: bool = False


@PromptDataset.register("guidedbench")
class GuidedBenchDataset(PromptDataset):
    def __init__(self, config: GuidedBenchConfig):
        super().__init__(config)
        df = pd.read_csv(config.path)
        self.idx, self.config_idx = self._select_idx(config, len(df))
        df = df.iloc[self.idx].reset_index(drop=True)
        self.messages = df["question"]
        self.targets = df["target"]

    def __len__(self):
        return len(self.messages)

    def __getitem__(self, idx: int) -> Conversation:
        return [
            {"role": "user", "content": self.messages.iloc[idx]},
            {"role": "assistant", "content": self.targets.iloc[idx]},
        ]
