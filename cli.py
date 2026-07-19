import argparse
import asyncio
from dotenv import load_dotenv

load_dotenv(".env")

from src.utils import make_experiment
from src.factory import AttackFactory, EncoderFactory, FeaturesFactory, ClassifierFactory


def generation():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, help="Name of the experiment run")
    parser.add_argument("--max_concurrency", type=int, default=10, help="Max concurrent attacks")

    parser.add_argument(
        "--seeds.dataset_names",
        nargs="+",
        required=True,
        metavar="NAME",
        help="One or more dataset names (repeatable tokens, e.g. adv_bench harmbench)",
    )
    parser.add_argument("--seeds.num_samples", type=int, default=1, help="Number of samples")
    
    parser.add_argument("--attack.type", type=str, default="crescendo", help="Attack type (crescendo|tap)")
    parser.add_argument("--attack.max_turns", type=int, default=8, help="Max turns (crescendo)")
    parser.add_argument("--attack.max_backtracks", type=int, default=2, help="Max backtracks (crescendo)")
    parser.add_argument("--attack.scoring_threshold", type=float, default=0.8, help="Scoring threshold")
    parser.add_argument("--attack.tree_width", type=int, default=3, help="Number of parallel branches (tap)")
    parser.add_argument("--attack.tree_depth", type=int, default=8, help="Max depth / turn budget (tap)")
    parser.add_argument("--attack.branching_factor", type=int, default=2, help="Child branches per node (tap)")
    
    parser.add_argument("--attack.adversarial.model_name", type=str, default="meta-llama/llama-3.1-8b-instruct", help="Adversarial model name")
    parser.add_argument("--attack.adversarial.temperature", type=float, default=1.0, help="Adversarial temperature")
    parser.add_argument("--attack.adversarial.top_p", type=float, default=1.0, help="Adversarial top p")
    parser.add_argument("--attack.adversarial.max_completion_tokens", type=int, default=1024, help="Adversarial max completion tokens")
    
    parser.add_argument("--attack.scoring.model_name", type=str, default="openai/gpt-oss-120b", help="Scoring model name")
    parser.add_argument("--attack.scoring.temperature", type=float, default=0.1, help="Scoring temperature")
    parser.add_argument("--attack.scoring.top_p", type=float, default=1.0, help="Scoring top p")
    parser.add_argument("--attack.scoring.max_completion_tokens", type=int, default=1024, help="Scoring max completion tokens")
    
    parser.add_argument("--attack.objective.model_name", type=str, default="meta-llama/llama-3.1-8b-instruct", help="Objective model name")
    parser.add_argument("--attack.objective.temperature", type=float, default=1.0, help="Objective temperature")
    parser.add_argument("--attack.objective.top_p", type=float, default=1.0, help="Objective top p")
    parser.add_argument("--attack.objective.max_completion_tokens", type=int, default=1024, help="Objective max completion tokens")

    parser.add_argument("--encoders.lexical.max_features", type=int, default=4096, help="Lexical max features")
    parser.add_argument("--encoders.lexical.stop_words", type=str, default="english", help="Lexical stop words")
    parser.add_argument("--encoders.lexical.max_df", type=float, default=0.95, help="Lexical max df")
    
    parser.add_argument("--encoders.semantic.model_name", type=str, default="qwen/qwen3-embedding-8b", help="Semantic model name")
    parser.add_argument("--encoders.semantic.max_context_tokens", type=int, default=32000, help="Semantic max context tokens") 
    parser.add_argument("--encoders.semantic.batch_size", type=int, default=128, help="Semantic batch size")

    args = parser.parse_args()
    
    cfg, experiment_dir, logger = make_experiment(args)

    attack_factory = AttackFactory(cfg, experiment_dir, logger)
    encoder_factory = EncoderFactory(cfg, experiment_dir, logger)

    asyncio.run(attack_factory.execute())
    encoder_factory.execute()


def analysis():
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, help="Name of the experiment run")
    parser.add_argument("--seed", type=int, help="Random seed for the train/test split and classifiers")

    parser.add_argument("--features.trim_to_first_n_turns", type=int, help="Trim to first n turns")
    parser.add_argument("--features.trim_the_last_n_turns", type=int, help="Trim the last n turns")

    parser.add_argument("--features.use_embeddings", nargs="+", type=str, help="Embeddings to use")
    parser.add_argument("--features.use_roles", nargs="+", type=str, help="Roles to use")
    parser.add_argument("--features.use_features", nargs="+", type=str, help="Features to use")
    
    parser.add_argument("--classifiers.logistic_regression.C", type=float, help="C value for logistic regression")

    parser.add_argument("--classifiers.gradient_boosting.learning_rate", type=float, help="Learning rate for gradient boosting")
    parser.add_argument("--classifiers.gradient_boosting.max_depth", type=int, help="Max depth for gradient boosting")
    parser.add_argument("--classifiers.gradient_boosting.l2_regularization", type=float, help="L2 regularization for gradient boosting")

    parser.add_argument("--baselines.llama_guard", action='store_true')

    parser.add_argument("--dataset_dirs", nargs="+", type=str, help="Paths to the dataset directories")
    
    args = parser.parse_args()

    cfg, experiment_dir, logger = make_experiment(args)

    dataset_dirs = [Path(dataset_dir) for dataset_dir in args.dataset_dirs]

    features_factory = FeaturesFactory(cfg, experiment_dir, logger)
    classifier_factory = ClassifierFactory(cfg, experiment_dir, logger)

    features_factory.execute(dataset_dirs)
    classifier_factory.execute()
    
