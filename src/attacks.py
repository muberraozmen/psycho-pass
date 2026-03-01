import os
from typing import Any

from pyrit.models import Seed, AttackResult
from pyrit.prompt_target import OpenAIChatTarget
from pyrit.executor.attack import (
    AttackAdversarialConfig,
    AttackScoringConfig,
    RedTeamingAttack,
    CrescendoAttack,
)
from pyrit.score import (
    FloatScaleThresholdScorer,
    SelfAskScaleScorer,
)


__all__ = ["RTA", "Crescendo"]


# API type -> (env var for key, env var for endpoint)
_API_ENV = {
    "ollama": ("OLLAMA_CHAT_KEY", "OLLAMA_CHAT_ENDPOINT"),
    "together": ("TOGETHER_CHAT_KEY", "TOGETHER_CHAT_ENDPOINT"),
    "openrouter": ("OPENROUTER_CHAT_KEY", "OPENROUTER_CHAT_ENDPOINT"),
}


def make_chat_bot(cfg: dict[str, Any]) -> OpenAIChatTarget:
    """Build an OpenAIChatTarget from config. Required keys: api_type, model_name."""
    model_name = cfg.get("model_name")
    if not model_name:
        raise ValueError("config must set 'model_name'")
    temperature = cfg.get("temperature", 0.7)
    top_p = cfg.get("top_p", 0.9)
    max_completion_tokens = cfg.get("max_completion_tokens", 1024)
    
    api_type = cfg.get("api_type")
    env_vars = _API_ENV.get(api_type)
    if env_vars is None:
        raise ValueError(f"Unsupported API type: {api_type}")

    key_var, endpoint_var = env_vars
    return OpenAIChatTarget(
        api_key=os.environ[key_var],
        endpoint=os.environ[endpoint_var],
        model_name=model_name,
        temperature=temperature,
        top_p=top_p,
        max_completion_tokens=max_completion_tokens,
    )


class BaseAttack:
    """Base attack class. Builds adversarial, scoring, and objective bots from config."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self.adversarial_bot = make_chat_bot(cfg.get("adversarial", {}))
        self.scoring_bot = make_chat_bot(cfg.get("scoring", {}))
        self.objective_bot = make_chat_bot(cfg.get("objective", {}))

        self.max_turns = cfg.get("max_turns", 10)
        self.scoring_threshold = cfg.get("scoring_threshold", 0.8)

        self.adversarial_config = self._make_adversarial_config()
        self.scoring_config = self._make_scoring_config()
        self.attack = self._make_attack()

    def _make_adversarial_config(self) -> AttackAdversarialConfig:
        return AttackAdversarialConfig(
            target=self.adversarial_bot
        )

    def _make_scoring_config(self) -> AttackScoringConfig:
        return AttackScoringConfig(
            objective_scorer=FloatScaleThresholdScorer(
                scorer=SelfAskScaleScorer(
                    chat_target=self.scoring_bot,
                    scale_arguments_path=SelfAskScaleScorer.ScalePaths.TASK_ACHIEVED_SCALE.value,
                    system_prompt_path=SelfAskScaleScorer.SystemPaths.RED_TEAMER_SYSTEM_PROMPT.value,
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
    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__(cfg)
        
    def _make_attack(self):
        return RedTeamingAttack(
            objective_target=self.objective_bot,
            attack_adversarial_config=self.adversarial_config,
            attack_scoring_config=self.scoring_config,
            max_turns=self.max_turns,
        )


class Crescendo(BaseAttack):
    def __init__(self, cfg: dict[str, Any]) -> None:
        self.max_backtracks = cfg.get("max_backtracks", 4)
        super().__init__(cfg)
    
    def _make_attack(self):
        return CrescendoAttack(
            objective_target=self.objective_bot,
            attack_adversarial_config=self.adversarial_config,
            attack_scoring_config=self.scoring_config,
            max_turns=self.max_turns,
            max_backtracks=self.max_backtracks,
        )
