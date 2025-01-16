from typing import List, Dict, Optional
import argparse
import json
from pathlib import Path
from models.model import GenerationModel
from utils.config import load_config
from datasets import load_dataset


def generate_explanations(
    model: GenerationModel,
    questions: List[Dict],
    num_iterations: int = 128,
    batch_size: int = 8
) -> List[Dict]:
    """
    Generate Chain-of-Thought explanations for a list of questions.
    Returns a list of dicts containing the question, gold answer,
    and the generated explanations.
    """
    results = []
    for question in questions:
        explanations = model.generate(
            passage=question.get("passage"),
            question=question["question"],
            choices=question.get("choices"),
            num_iterations=num_iterations,
            batch_size=batch_size
        )
        results.append({
            "id": question["id"],
            "passage": question.get("passage", ""),
            "question": question["question"],
            "gold_answer": question["gold_answer"],
            "choices": question.get("choices", ""),
            "generated_explanations_and_answers": explanations
        })
    return results


def load_qa_dataset(dataset_name: str, config: Dict) -> List[Dict]:
    """
    Load and format QA datasets from huggingface or local files, depending on the dataset_name.
    Supported: arc, drop, strategy_qa
    """
    if dataset_name == "arc":
        dataset = load_dataset("allenai/ai2_arc", name="ARC-Challenge", split="train")
        few_shot_ids = set(config.get("cot_generation", {}).get("few_shot_samples_ids", []))

        formatted_data = []
        for item in dataset:
            if item["id"] in few_shot_ids:
                continue
            formatted_data.append({
                "id": item["id"],
                "question": item["question"],
                "gold_answer": item["answerKey"],
                "choices": dict(
                    label=item["choices"]["label"],
                    text=item["choices"]["text"]
                )
            })

    elif dataset_name == "drop":
        dataset = load_dataset("ucinlp/drop", split="train")
        few_shot_ids = set(config.get("cot_generation", {}).get("few_shot_samples_ids", []))

        formatted_data = []
        for item in dataset:
            if item["query_id"] in few_shot_ids:
                continue
            formatted_data.append({
                "id": item["query_id"],
                "question": item["question"],
                "passage": item["passage"],
                "gold_answer": item["answers_spans"]["spans"]
            })

    elif dataset_name == "strategy_qa":
        dataset_file = config.get("cot_generation", {}).get("strategyqa_file", "data/strategyqa/train.json")
        if not Path(dataset_file).exists():
            raise ValueError(f"StrategyQA file not found: {dataset_file}")

        few_shot_ids = set(config.get("cot_generation", {}).get("few_shot_samples_ids", []))
        formatted_data = []
        with open(dataset_file, "r", encoding="utf-8") as f:
            items = json.load(f)
            for item in items:
                if item["qid"] in few_shot_ids:
                    continue
                gold_answer_str = "yes" if item["answer"] else "no"
                formatted_data.append({
                    "id": item["qid"],
                    "question": item["question"],
                    "gold_answer": gold_answer_str
                })
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return formatted_data


def save_results(results: List[Dict], output_path: str):
    """
    Save the CoT generation results to a JSONL file.
    Each line is a JSON object representing a question and the generated explanations.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Dataset name (arc, drop, strategy_qa)")
    parser.add_argument("--config_file", required=True, help="Path to config file")
    parser.add_argument("--max_samples", type=int, default=10,
                        help="Number of samples to generate CoT for, for demonstration")
    args = parser.parse_args()

    config = load_config(args.config_file)

    dataset_config_path = Path(args.config_file).parent / "datasets" / f"{args.dataset}.yaml"
    dataset_config = load_config(dataset_config_path) if dataset_config_path.exists() else {}

    model = GenerationModel(
        model_name=config["model"]["name"],
        model_config=config["model"],
        cache_dir=config["model"]["cache_dir"],
        sampling_params=config["sampling_params"],
        dataset_name=args.dataset
    )

    questions = load_qa_dataset(args.dataset, dataset_config)[: args.max_samples]

    results = generate_explanations(
        model=model,
        questions=questions,
        num_iterations=config["cot_generation"]["num_iterations"],
        batch_size=config["cot_generation"]["batch_size"]
    )

    save_results(results, f"output/{args.dataset}_cot_results.jsonl")


if __name__ == "__main__":
    main()
