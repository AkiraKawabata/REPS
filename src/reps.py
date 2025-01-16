# src/reps.py

import argparse
import logging
import warnings
from pathlib import Path
from typing import List, Dict

from models.model import GenerationModel
from utils.config import load_config
from utils.io import load_jsonl, save_jsonl
from prompts import Task, Dataset, PromptManager

logger = logging.getLogger(__name__)


class SelfEvaluator:
    """
    Pairwise tournament evaluator for Chain-of-Thought explanations.
    Selects the 'best' explanation based on pairwise self-evaluation prompts.
    """

    def __init__(self, model: GenerationModel, dataset_name: str, config: Dict):
        """
        Args:
            model: the generation model used for self-evaluation
            dataset_name: e.g. "arc", "drop", "strategy_qa"
            config: dictionary that contains self_eval settings (N, S, etc.)
        """
        self.model = model
        self.config = config
        try:
            self.dataset_enum = Dataset[dataset_name.upper()]
        except KeyError:
            warnings.warn(
                f"Unsupported dataset: '{dataset_name}'. "
                "Falling back to Dataset.ARC for prompt usage."
            )
            self.dataset_enum = Dataset.ARC

        self.N = config["self_eval"]["N"]           # Number of correct solutions to consider in the tournament
        self.S = config["self_eval"]["S"]  # Number of pairwise comparisons (majority vote)
        self.prompt_manager = PromptManager()

    def evaluate_pair(self, kwargs: Dict, exp1: str, exp2: str) -> int:
        """
        Compare two explanations (exp1, exp2) using self-evaluation prompts.
        We run it S times and pick the majority.
        
        Returns:
            0 if exp1 wins, 1 if exp2 wins
        """
        prompt_text = self.prompt_manager.format_prompt(
            task=Task.SELF_EVALUATION,
            dataset=self.dataset_enum,
            **kwargs,
            explanation1=exp1,
            explanation2=exp2
        )
        responses = []
        for _ in range(self.S):
            out = self.model.generate_raw_text(
                prompt_text=prompt_text,
                num_iterations=1,
                batch_size=1
            )
            if out:
                responses.append(out[0])

        votes = [0, 0]
        for resp in responses:
            text = resp.get("eval_result", "").lower()
            if "jmh" in text:
                votes[0] += 1
            elif "bhy" in text:
                votes[1] += 1

        logger.debug(f"Vote result => exp1={votes[0]}, exp2={votes[1]}")
        return 0 if votes[0] > votes[1] else 1

    def run_tournament(self, item: Dict) -> Dict:
        """
        Collect correct explanations (those with final answers matching gold_answer),
        keep up to N of them, and run a single-elimination tournament to pick the best.
        """
        question = item["question"]
        choices = item.get("choices")
        if choices:
            item["choices"] = self.prompt_manager._format_choices(choices)

        # Filter correct explanations
        correct_exps = []
        if isinstance(item["gold_answer"], list):
            gold_answers_lower = [a.strip().lower() for a in item["gold_answer"]]
            for exp in item["generated_explanations_and_answers"]:
                if exp["answer"].strip().lower() in gold_answers_lower:
                    correct_exps.append(exp)
        else:
            gold_lower = item["gold_answer"].strip().lower()
            for exp in item["generated_explanations_and_answers"]:
                if exp["answer"].strip().lower() == gold_lower:
                    correct_exps.append(exp)

        current_exps = correct_exps[: self.N]
        round_num = 1

        # Single-elimination tournament
        while len(current_exps) > 1:
            logger.info(f"Tournament round {round_num}, {len(current_exps)} explanations remain.")
            next_round = []
            for i in range(0, len(current_exps), 2):
                if i + 1 >= len(current_exps):
                    # If odd number, last one automatically advances
                    next_round.append(current_exps[i])
                    continue
                winner_idx = self.evaluate_pair(
                    item,
                    exp1=current_exps[i]["explanation"],
                    exp2=current_exps[i + 1]["explanation"]
                )
                next_round.append(current_exps[i + winner_idx])
            current_exps = next_round
            round_num += 1

        best_exp = current_exps[0] if current_exps else None
        return {
            "id": item["id"],
            "passage": item.get("passage"),
            "question": item["question"],
            "choices": item.get("choices"),
            "gold_answer": item["gold_answer"],
            "best_explanation": best_exp,
            "all_explanations": item["generated_explanations_and_answers"],
            "tournament_size": len(correct_exps[:self.N])
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True, help="CoT results file (JSONL)")
    parser.add_argument("--config_file", required=True, help="Config file path (YAML)")
    parser.add_argument("--dataset", default="none", help="Dataset name or 'none'")
    args = parser.parse_args()

    config = load_config(args.config_file)

    model = GenerationModel(
        model_name=config["model"]["name"],
        model_config=config["model"],
        cache_dir=config["model"].get("cache_dir"),
        sampling_params=config["sampling_params"],
        dataset_name=args.dataset
    )

    evaluator = SelfEvaluator(model, args.dataset, config)

    input_path = Path(args.input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"File not found: {input_path}")

    results = load_jsonl(input_path)
    logger.info(f"Loaded {len(results)} items from {input_path}")

    evaluated_results = []
    for i, item in enumerate(results, start=1):
        logger.info(f"Evaluating item {i}/{len(results)} (id={item['id']})")
        res = evaluator.run_tournament(item)
        evaluated_results.append(res)

    out_path = Path(f"output/{input_path.stem}_evaluated_results.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_jsonl(evaluated_results, out_path)
    logger.info(f"Saved self-evaluation results to {out_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
