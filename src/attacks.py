from __future__ import annotations

import os

from pyrit.models import Seed, AttackResult
from pyrit.prompt_target import OpenAIChatTarget
from pyrit.executor.attack import (
    AttackAdversarialConfig,
    AttackScoringConfig,
    RedTeamingAttack,
    CrescendoAttack,
    TAPAttack,
)
from pyrit.executor.attack.multi_turn.tree_of_attacks import TAPAttackScoringConfig
from pyrit.score import (
    FloatScaleThresholdScorer,
    SelfAskScaleScorer,
)


__all__ = ["RTA", "Crescendo", "TAP"]


DEFAULT_MODEL_NAME = "meta-llama/llama-3.1-8b-instruct"
DEFAULT_SCORING_THRESHOLD = 0.8
DEFAULT_MAX_TURNS = 10
DEFAULT_CRESCENDO_MAX_BACKTRACKS = 2
DEFAULT_TAP_TREE_WIDTH = 3
DEFAULT_TAP_TREE_DEPTH = 8
DEFAULT_TAP_BRANCHING_FACTOR = 2

API_KEY = os.environ["OPENROUTER_API_KEY"]
CHAT_ENDPOINT = os.environ["OPENROUTER_CHAT_ENDPOINT"]

scoring_scale_arguments_path = SelfAskScaleScorer.ScalePaths.TASK_ACHIEVED_SCALE.value
scoring_system_prompt_path = SelfAskScaleScorer.SystemPaths.RED_TEAMER_SYSTEM_PROMPT.value


class BaseAttack:

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg

        self.adversarial_bot = self._make_chat_bot(cfg.get("adversarial"))
        self.scoring_bot = self._make_chat_bot(cfg.get("scoring"))
        self.objective_bot = self._make_chat_bot(cfg.get("objective"))

        self.max_turns = cfg.get("max_turns", DEFAULT_MAX_TURNS)
        self.scoring_threshold = cfg.get("scoring_threshold", DEFAULT_SCORING_THRESHOLD)

        self.adversarial_config = self._make_adversarial_config()
        self.scoring_config = self._make_scoring_config()
        self.attack = self._make_attack()

    def _make_chat_bot(self, cfg: dict | None = None) -> OpenAIChatTarget:
        model_name = cfg.get("model_name", DEFAULT_MODEL_NAME)
        temperature = cfg.get("temperature")
        top_p = cfg.get("top_p")
        max_completion_tokens = cfg.get("max_completion_tokens")
        
        return OpenAIChatTarget(
            api_key=API_KEY,
            endpoint=CHAT_ENDPOINT,
            model_name=model_name,
            temperature=temperature,
            max_completion_tokens=max_completion_tokens,
            top_p=top_p,
        )
    
    def _make_adversarial_config(self) -> AttackAdversarialConfig:
        return AttackAdversarialConfig(
            target=self.adversarial_bot
        )

    def _make_scoring_config(self) -> AttackScoringConfig:
        return AttackScoringConfig(
            objective_scorer=FloatScaleThresholdScorer(
                scorer=SelfAskScaleScorer(
                    chat_target=self.scoring_bot,
                    scale_arguments_path=scoring_scale_arguments_path,
                    system_prompt_path=scoring_system_prompt_path,
                ),
                threshold=self.scoring_threshold,
            ),
        )

    def _make_attack(self):
        raise NotImplementedError()

    async def execute(self, seed: Seed) -> AttackResult:
        result = await self.attack.execute_async(objective=seed.value)
        return result


class RTA(BaseAttack):
    def __init__(self, cfg: dict) -> None:
        super().__init__(cfg)
        
    def _make_attack(self):
        return RedTeamingAttack(
            objective_target=self.objective_bot,
            attack_adversarial_config=self.adversarial_config,
            attack_scoring_config=self.scoring_config,
            max_turns=self.max_turns,
        )


class Crescendo(BaseAttack):
    def __init__(self, cfg: dict) -> None:
        self.max_backtracks = cfg.get("max_backtracks", DEFAULT_CRESCENDO_MAX_BACKTRACKS)
        super().__init__(cfg)

    def _make_attack(self):
        return CrescendoAttack(
            objective_target=self.objective_bot,
            attack_adversarial_config=self.adversarial_config,
            attack_scoring_config=self.scoring_config,
            max_turns=self.max_turns,
            max_backtracks=self.max_backtracks,
        )


class TAP(BaseAttack):
    def __init__(self, cfg: dict) -> None:
        self.tree_width = cfg.get("tree_width", DEFAULT_TAP_TREE_WIDTH)
        self.tree_depth = cfg.get("tree_depth", DEFAULT_TAP_TREE_DEPTH)
        self.branching_factor = cfg.get("branching_factor", DEFAULT_TAP_BRANCHING_FACTOR)
        super().__init__(cfg)

    def _make_scoring_config(self) -> TAPAttackScoringConfig:
        return TAPAttackScoringConfig(
            objective_scorer=FloatScaleThresholdScorer(
                scorer=SelfAskScaleScorer(
                    chat_target=self.scoring_bot,
                    scale_arguments_path=scoring_scale_arguments_path,
                    system_prompt_path=scoring_system_prompt_path,
                ),
                threshold=self.scoring_threshold,
            ),
        )

    def _make_attack(self):
        return TAPAttack(
            objective_target=self.objective_bot,
            attack_adversarial_config=self.adversarial_config,
            attack_scoring_config=self.scoring_config,
            tree_width=self.tree_width,
            tree_depth=self.tree_depth,
            branching_factor=self.branching_factor,
        )
