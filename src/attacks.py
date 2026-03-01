import os
from pyrit.models import Seed, AttackResult
from pyrit.prompt_target import OpenAIChatTarget
from pyrit.executor.attack import (
    AttackAdversarialConfig,
    AttackScoringConfig,
    RedTeamingAttack,
    RTASystemPromptPaths,
)
from pyrit.score import (
    SelfAskTrueFalseScorer, 
    TrueFalseQuestion, 
    TrueFalseQuestionPaths
)


__all__ = ["RTA"]


def build_chat_bot(cfg: dict) -> OpenAIChatTarget:    
    if cfg.get("type") == "ollama":
        return OpenAIChatTarget(
            api_key= os.environ["OLLAMA_CHAT_KEY"],
            endpoint=os.environ["OLLAMA_CHAT_ENDPOINT"],
            model_name=cfg.get("model_name"),
            temperature=cfg.get("temperature", 0.5),
        )
    elif cfg.get("type") == "togetherai":
        return OpenAIChatTarget(
            api_key=os.environ["TOGETHER_CHAT_KEY"],
            endpoint=os.environ["TOGETHER_CHAT_ENDPOINT"],
            model_name=cfg.get("model_name"),
            temperature=cfg.get("temperature", 0.5),
        )
    else:
        raise ValueError(f"Unsupported type: {cfg.get('type')}")


class BaseAttack:
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
    
    async def run(self, seed: Seed) -> AttackResult:
        raise NotImplementedError()


class RTA(BaseAttack):
    """ Red Teaming Attack (RTA) """
    def __init__(self, cfg: dict) -> None:
        super().__init__(cfg)

        # Configuration
        self.max_turns = cfg.get("max_turns", 3)
        
        # Build Chat Bots
        self.adversarial_bot = build_chat_bot(cfg.get("adversarial", {}))
        self.scoring_bot = build_chat_bot(cfg.get("scoring", {}))
        self.objective_bot = build_chat_bot(cfg.get("objective", {}))

        # Prompts
        self.adversarial_prompt = cfg.get("adversarial", {}).get("prompt")
        self.scoring_prompt = cfg.get("scoring", {}).get("prompt")

    async def run(self, seed: Seed) -> AttackResult:
        objective = seed.value

        # Adversarial Configuration - Attacker
        if self.adversarial_prompt:
            seed_prompt =  self.adversarial_prompt.format(objective=objective)
            adversarial_config = AttackAdversarialConfig(
                target=self.adversarial_bot, 
                seed_prompt=seed_prompt
            )
        else:
            adversarial_config = AttackAdversarialConfig(
                target=self.adversarial_bot, 
                system_prompt_path=RTASystemPromptPaths.TEXT_GENERATION.value
            )
        
        # Scoring Configuration - Judge
        if self.scoring_prompt:
            true_desc = self.scoring_prompt.get("true_description", "").format(objective=objective)
            false_desc = self.scoring_prompt.get("false_description", "").format(objective=objective)
            scoring_config = AttackScoringConfig(
                objective_scorer=SelfAskTrueFalseScorer(
                    chat_target=self.scoring_bot,
                    true_false_question=TrueFalseQuestion(
                        true_description=true_desc,
                        false_description=false_desc
                    )
                )
            )
        else:
            scoring_config = AttackScoringConfig(
                objective_scorer=SelfAskTrueFalseScorer(
                    chat_target=self.scoring_bot,
                    true_false_question_path=TrueFalseQuestionPaths.TASK_ACHIEVED.value
                )
            )
            
        # Assemble Attack
        attack = RedTeamingAttack(
            objective_target=self.objective_bot,
            attack_adversarial_config=adversarial_config,
            attack_scoring_config=scoring_config,
            max_turns=self.max_turns
        )

        # Execute Attack
        result = await attack.execute_async(
            objective=objective, 
            memory_labels={
                "data_type": seed.data_type if seed.data_type else "text",
                "harm_categories": seed.harm_categories if seed.harm_categories else None,
            }
        )

        return result