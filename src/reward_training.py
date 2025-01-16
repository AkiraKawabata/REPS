# src/reward_training.py

import argparse
import logging
import random
import json
from pathlib import Path
from typing import List, Dict

import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from trl import RewardTrainer, RewardConfig

from utils.config import load_config
from utils.io import load_jsonl

logger = logging.getLogger(__name__)


def format_choices(choices: Dict[str, List[str]]) -> str:
    """
    Format multiple-choice options as a multiline string like:
      A. optionA
      B. optionB
      ...
    """
    return "\n".join(
        f"{label}. {text}"
        for label, text in zip(choices["label"], choices["text"])
    )


def extract_answer(exp: Dict) -> str:
    """
    Extract the 'answer' string from an explanation dict.
    Lowercase the string for comparison.
    """
    return exp.get("answer", "").strip().lower()


def prepare_training_data(results: List[Dict]) -> Dict:
    """
    Build a dataset for reward model training by pairing:
      - chosen: the best (correct) explanation
      - rejected: a random incorrect explanation
    """
    training_data = {
        "question": [],
        "chosen": [],
        "rejected": []
    }

    for item in results:
        # Build question text (depending on dataset type)
        if "passage" in item and item["passage"]:
            # For tasks like DROP
            question_text = f"{item['passage']}\n{item['question']}"
        elif "choices" in item and item["choices"]:
            # For tasks like ARC
            question_text = f"{item['question']}\n{format_choices(item['choices'])}"
        else:
            # For tasks like StrategyQA
            question_text = item["question"]

        best_exp = item["best_explanation"]
        if best_exp is None:
            # If no winner explanation, skip this item
            continue

        # store question text
        training_data["question"].append(question_text)

        # Check correctness of best explanation
        gold_answer = item["gold_answer"].strip().lower()
        if extract_answer(best_exp) == gold_answer:
            training_data["chosen"].append(best_exp)
        else:
            # If the "best_explanation" is not actually correct, we might skip or do something else
            # but for simplicity, let's still treat it as chosen
            training_data["chosen"].append(best_exp)

        # gather incorrect
        incorrect_exps = []
        for exp in item["all_explanations"]:
            if exp is best_exp:
                continue
            if extract_answer(exp) != gold_answer:
                incorrect_exps.append(exp)

        # if we have any incorrect, pick one randomly
        if incorrect_exps:
            chosen_incorrect = random.choice(incorrect_exps)
            training_data["rejected"].append(chosen_incorrect)
        else:
            # If no incorrect explanation, fill a dummy or skip
            # We'll skip here
            training_data["rejected"].append({"explanation": "", "answer": ""})

    return training_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True, help="Evaluated results file (JSONL)")
    parser.add_argument("--config_file", required=True, help="Config file path (YAML)")
    args = parser.parse_args()

    config = load_config(args.config_file)

    # Load evaluated results
    results = load_jsonl(args.input_file)
    training_data = prepare_training_data(results)
    logger.info("Prepared training data with %d chosen and %d rejected",
                len(training_data["chosen"]), len(training_data["rejected"]))

    # Save chosen/rejected pairs as JSONL (optional)
    out_path = Path("output/reward_training_data.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for c in training_data["chosen"]:
            record = {"label": "chosen"}
            record.update(c)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        for r in training_data["rejected"]:
            record = {"label": "rejected"}
            record.update(r)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info(f"Saved training data to {out_path}")

    # Build pairwise dataset for RewardModel training
    dataset_records = []
    pair_len = min(len(training_data["chosen"]), len(training_data["rejected"]))
    for i in range(pair_len):
        question_text = training_data["question"][i]
        c = training_data["chosen"][i]
        r = training_data["rejected"][i]

        chosen_text = (
            f"Question:\n{question_text}\n"
            f"Explanation:{c.get('explanation', '')}\n"
            f"Answer: {c.get('answer', '')}"
        )
        rejected_text = (
            f"Question:\n{question_text}\n"
            f"Explanation:{r.get('explanation', '')}\n"
            f"Answer: {r.get('answer', '')}"
        )

        dataset_records.append({
            "chosen": chosen_text,
            "rejected": rejected_text
        })

    hf_dataset = Dataset.from_list(dataset_records)
    logger.info(f"Constructed HF dataset with {len(hf_dataset)} pairwise samples.")

    # Initialize tokenizer/model
    model_name = config["model"]["name"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=config["model"].get("trust_remote_code", True),
        cache_dir=config["model"].get("cache_dir", None)
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=1,
        trust_remote_code=config["model"].get("trust_remote_code", True),
        cache_dir=config["model"].get("cache_dir", None)
    )

    # Tokenize
    def tokenize_function(example):
        max_len = config["training"].get("max_length", 512)
        chosen_enc = tokenizer(
            example["chosen"],
            max_length=max_len,
            truncation=True,
            padding="max_length"
        )
        rejected_enc = tokenizer(
            example["rejected"],
            max_length=max_len,
            truncation=True,
            padding="max_length"
        )
        return {
            "input_ids_chosen": chosen_enc["input_ids"],
            "attention_mask_chosen": chosen_enc["attention_mask"],
            "input_ids_rejected": rejected_enc["input_ids"],
            "attention_mask_rejected": rejected_enc["attention_mask"]
        }

    tokenized_dataset = hf_dataset.map(tokenize_function, batched=False)

    # Prepare training args for RewardTrainer
    training_args = RewardConfig(
        per_device_train_batch_size=config["training"].get("per_device_train_batch_size", 8),
        num_train_epochs=config["training"].get("num_train_epochs", 1),
        learning_rate=config["training"].get("learning_rate", 3e-7),
        max_length=config["training"].get("max_length", 1024),
        output_dir=config["training"].get("output_dir", "output/reward_model"),
        logging_steps=config["training"].get("logging_steps", 10),
        evaluation_strategy=config["training"].get("evaluation_strategy", "no"),
    )

    trainer = RewardTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=tokenized_dataset
        # eval_dataset=...,  # if needed
    )
    trainer.train()

    # Save the trained model
    trainer.save_model(training_args.output_dir)
    logger.info("Finished training & saved model to %s", training_args.output_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
