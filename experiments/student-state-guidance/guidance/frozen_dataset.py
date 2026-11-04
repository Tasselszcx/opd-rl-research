"""Read the pre-registered K&K questions without regenerating or adapting them."""
import json
from pathlib import Path
from reasoning_gym.dataset import ProceduralDataset
from reasoning_gym.logic.knights_knaves import KnightsKnavesConfig, KnightsKnavesDataset

class FrozenKnightsKnavesDataset(ProceduralDataset):
    def __init__(self, path, max_examples=None):
        self.rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
        if max_examples is not None:
            self.rows = self.rows[:max_examples]
        config = KnightsKnavesConfig(n_people=6, depth_constraint=2, width_constraint=2,
                                     size=len(self.rows), seed=1)
        super().__init__(config, seed=1, size=len(self.rows))
        self.scorer = KnightsKnavesDataset(config)
    def __getitem__(self, index):
        return self.rows[index].copy()
    def score_answer(self, answer, entry):
        return self.scorer.score_answer(answer, entry)
