import os
from pyrit.models import Seed
from pyrit.prompt_target import OpenAIChatTarget
from pyrit.executor.attack import (
    AttackAdversarialConfig,
    AttackScoringConfig,
    RedTeamingAttack,
    RTASystemPromptPaths,
)
from pyrit.executor.attack import AttackScoringConfig
from pyrit.score import SelfAskTrueFalseScorer, TrueFalseQuestionPaths

from typing import Union


__all__ = ["run_attack", "build_red_teaming_attack"]


async def run_attack(attack: Union[RedTeamingAttack], seed: Seed) -> None:
    return await attack.execute_async(
        objective=seed.value, 
        memory_labels={
            "data_type": seed.data_type if seed.data_type else "text",
            "harm_categories": seed.harm_categories if seed.harm_categories else None,
        }
    )


def build_chat_model(cfg: dict) -> OpenAIChatTarget:    
    if cfg.get("type") == "ollama":
        return OpenAIChatTarget(
            api_key= os.environ["OLLAMA_CHAT_KEY"],
            endpoint=os.environ["OLLAMA_CHAT_ENDPOINT"],
            model_name=cfg.get("model_name"),
            temperature=cfg.get("temperature", 0.5),
        )
    elif cfg.get("type") == "openai":
        return OpenAIChatTarget(
            api_key=os.environ["OPENAI_CHAT_KEY"],
            endpoint=os.environ["OPENAI_CHAT_ENDPOINT"],
            model_name=cfg.get("model_name"),
            temperature=cfg.get("temperature", 0.5),
        )
    elif cfg.get("type") == "anthropic":
        return OpenAIChatTarget(
            api_key=os.environ["ANTHROPIC_CHAT_KEY"],
            endpoint=os.environ["ANTHROPIC_CHAT_ENDPOINT"],
            model_name=cfg.get("model_name"),
            temperature=cfg.get("temperature", 0.5),
        )
    elif cfg.get("type") == "gemini":
        return OpenAIChatTarget(
            api_key=os.environ["GEMINI_CHAT_KEY"],
            endpoint=os.environ["GEMINI_CHAT_ENDPOINT"],
            model_name=cfg.get("model_name"),
            temperature=cfg.get("temperature", 0.5),
        )
    else:
        raise ValueError(f"Unsupported model type: {cfg.get('type')}")


def build_red_teaming_attack(cfg: dict) -> RedTeamingAttack:    
    """
    Make a red teaming attack configuration.
    """

    # Adversarial configuration 
    adversarial_model = build_chat_model(cfg.get("adversarial"))
    adversarial_config = AttackAdversarialConfig(
        target=adversarial_model, 
        system_prompt_path= RTASystemPromptPaths.TEXT_GENERATION.value, 
    )
    '''
    NOTE: source of RTASystemPromptPaths.TEXT_GENERATION.value
    https://github.com/Azure/PyRIT/blob/main/pyrit/datasets/executors/red_teaming/text_generation.yaml
    '''

    # Scoring configuration 
    scoring_model =  build_chat_model(cfg.get("scoring"))
    scoring_config = AttackScoringConfig(
        objective_scorer=SelfAskTrueFalseScorer(
            chat_target=scoring_model,
            # NOTE: This will come in next release of PyRIT.
            # true_false_question_path=TrueFalseQuestionPaths.TASK_ACHIEVED.value 
            true_false_question_path="src/task_achieved.yaml"
        ),
    )
    '''
    NOTE: common source of TrueFalseQuestionPaths.QUESTION_ANSWERING.value
    https://github.com/Azure/PyRIT/blob/main/pyrit/datasets/score/true_false_question/question_answering.yaml
    Alternative: TrueFalseQuestionPaths.TASK_ACHIEVED.value
    https://github.com/Azure/PyRIT/blob/main/pyrit/datasets/score/true_false_question/task_achieved.yaml
    '''
    
    return RedTeamingAttack(
        objective_target=build_chat_model(cfg.get("objective")),
        attack_adversarial_config=adversarial_config,
        attack_scoring_config=scoring_config,
        max_turns=cfg.get("max_turns", 3))

