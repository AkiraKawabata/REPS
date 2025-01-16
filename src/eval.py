import argparse
import torch
from dataclasses import dataclass
from typing import List, Dict, Optional
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from utils.io import load_jsonl


@dataclass
class RationaleExample:
    """
    Represents an evaluation example that includes:
      - question
      - optional passage
      - optional choices (for multiple-choice tasks)
      - gold_answer
      - multiple rationales
      - rationale_types describing correctness (e.g., "correct", "incorrect_incorrect", "incorrect_correct")
    """
    question: str
    passage: Optional[str]
    choices: Optional[List[Dict[str, str]]]
    gold_answer: str
    rationales: List[str]
    rationale_types: List[str]


class DatasetFormatter:
    """
    Formats dataset items into a unified structure for rationale evaluation.
    """

    @staticmethod
    def format_choices(choices: List[str]) -> str:
        """
        Convert a list of choice strings into a multiline string with labels A, B, C, etc.
        Example:
          ["dog", "cat"] => "A. dog\nB. cat"
        """
        return "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)])

    @staticmethod
    def format_prompt(example: Dict, explanation: str) -> str:
        """
        Build a prompt that includes passage, question, choices, the explanation,
        and the gold_answer for reference. 
        """
        prompt_parts = []

        # Add passage if present
        if example.get("passage"):
            prompt_parts.append(f"Passage:\n{example['passage']}")

        # Always add question
        prompt_parts.append(f"Question:\n{example['question']}")

        # Add choices if present
        if example.get("choices"):
            formatted_choices = DatasetFormatter.format_choices([c["text"] for c in example["choices"]])
            prompt_parts.append(f"Choices:\n{formatted_choices}")

        # Explanation and gold answer
        prompt_parts.append(f"Explanation: {explanation}")
        prompt_parts.append(f"Answer: {example['gold_answer']}")

        return "\n".join(prompt_parts)

    @classmethod
    def convert_to_evaluation_format(cls, dataset_name: str, example: Dict) -> RationaleExample:
        """
        Convert a dataset-specific record into a RationaleExample.
        This assumes each example has:
          - question, gold_answer, possibly passage, choices
          - lists: 'incorrect_rationales_for_incorrect_answer', 'incorrect_rationales_for_correct_answer'
          - a single 'correct_rationale'
        """
        base_info = {
            "question": example["question"],
            "passage": example.get("passage"),
            "choices": example.get("choices"),
            "gold_answer": example["gold_answer"]
        }

        # Collect rationales
        rationales = []
        rationale_types = []

        # Rationale for incorrect answers
        for item in example["incorrect_rationales_for_incorrect_answer"]:
            rationales.append(item["rationale"])
            rationale_types.append("incorrect_incorrect")

        # Rationale for correct answer but incorrect explanation
        for r in example["incorrect_rationales_for_correct_answer"]:
            rationales.append(r)
            rationale_types.append("incorrect_correct")

        # Correct rationale
        rationales.append(example["correct_rationale"])
        rationale_types.append("correct")

        return RationaleExample(
            question=base_info["question"],
            passage=base_info["passage"],
            choices=base_info["choices"],
            gold_answer=base_info["gold_answer"],
            rationales=rationales,
            rationale_types=rationale_types
        )


def collate_fn(batch):
    max_length = max(x["input_ids"].size(1) for x in batch)

    all_input_ids = []
    all_attention_mask = []
    all_rationale_types = []

    for item in batch:
        input_ids = item["input_ids"]
        attention_mask = item["attention_mask"]
        pad_token_id = item["pad_token_id"]

        # Current sequence length
        curr_len = input_ids.size(1)
        pad_len = max_length - curr_len

        if pad_len > 0:
            padding = torch.full(
                (input_ids.size(0), pad_len),
                pad_token_id,
                dtype=input_ids.dtype
            )
            attention_padding = torch.zeros(
                (attention_mask.size(0), pad_len),
                dtype=attention_mask.dtype
            )
            input_ids = torch.cat([input_ids, padding], dim=1)
            attention_mask = torch.cat([attention_mask, attention_padding], dim=1)

        all_input_ids.append(input_ids)
        all_attention_mask.append(attention_mask)
        all_rationale_types.append(item["rationale_types"])

    return {
        "input_ids": torch.cat(all_input_ids, dim=0),
        "attention_mask": torch.cat(all_attention_mask, dim=0),
        "rationale_types": sum(all_rationale_types, [])
    }


class RationaleDataset(Dataset):
    def __init__(self, examples: List[RationaleExample], tokenizer, max_length: int = 512):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token_id = tokenizer.pad_token_id

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        prompts = []

        # Build a prompt for each rationale
        for rationale in example.rationales:
            prompt_text = DatasetFormatter.format_prompt(
                {
                    "question": example.question,
                    "passage": example.passage,
                    "choices": example.choices,
                    "gold_answer": example.gold_answer
                },
                rationale
            )
            prompts.append(prompt_text)

        # Tokenize all rationale prompts
        encodings = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "rationale_types": example.rationale_types,
            "pad_token_id": self.pad_token_id
        }


class RationaleEvaluator:
    """
    A class to evaluate rationale correctness using a reward model.
    """
    def __init__(self, model_path: str, device: str = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding="left")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=1,
            torch_dtype=(torch.bfloat16 if torch.cuda.is_available() else torch.float32),
        ).to(device)

        # Ensure we have a pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def evaluate(self, dataset_name: str, eval_examples: List[Dict], batch_size: int = 8) -> Dict[str, float]:
        """
        Evaluate a list of examples (in raw dict form). Each example is converted
        to RationaleExample, then loaded into a dataset, then we compute the ratio
        of correct or acceptable rationales.
        """
        # Convert raw data to RationaleExample
        formatted_examples = [
            DatasetFormatter.convert_to_evaluation_format(dataset_name, ex)
            for ex in eval_examples
        ]
        dataset = RationaleDataset(formatted_examples, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

        stats = {
            "total": 0,
            "correct": 0,
            "correct_from_rationale": 0
        }

        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

                # logits shape: [batch_size *  (num_rationales_per_example), 1]
                # We group them by 5 (in the original example) => [batch_size, 5]
                logits = outputs.logits.squeeze(-1).view(-1, 5)

                for question_idx in range(logits.size(0)):
                    question_logits = logits[question_idx]
                    selected_rationale_idx = torch.argmax(question_logits).item()

                    # rationale_types are stored in a flattened manner
                    start_idx = question_idx * 5
                    end_idx = start_idx + 5
                    question_rationale_types = batch["rationale_types"][start_idx:end_idx]
                    selected_type = question_rationale_types[selected_rationale_idx]

                    stats["total"] += 1
                    if selected_type == "correct":
                        stats["correct"] += 1
                    if selected_type in ["correct", "incorrect_correct"]:
                        stats["correct_from_rationale"] += 1

        results = {
            "correct_ratio": stats["correct"] / stats["total"] if stats["total"] else 0.0,
            "correct_from_rationale_ratio": stats["correct_from_rationale"] / stats["total"] if stats["total"] else 0.0
        }
        return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate rationale selection model")
    parser.add_argument("--dataset", type=str, required=True, choices=["arc", "drop", "strategy_qa"],
                        help="Dataset name (arc, drop, or strategy_qa)")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the trained evaluation model")
    parser.add_argument("--eval_data_dir", type=str, default="data/eval_data",
                        help="Directory containing eval_<dataset>.jsonl")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for evaluation")
    args = parser.parse_args()

    # Load the data from a JSONL file
    eval_data_file = f"{args.eval_data_dir}/eval_{args.dataset}.jsonl"
    eval_data = load_jsonl(eval_data_file)

    evaluator = RationaleEvaluator(args.model_path)
    results = evaluator.evaluate(args.dataset, eval_data, batch_size=args.batch_size)

    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model_path}")
    print(f"Correct Rationale Selection Accuracy: {results['correct_ratio']:.4f}")
    print(f"Correct or Valid Rationale Selection Ratio: {results['correct_from_rationale_ratio']:.4f}")


if __name__ == "__main__":
    main()
