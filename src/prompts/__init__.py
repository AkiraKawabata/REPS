from typing import Dict, Optional, List
from pathlib import Path
from importlib import import_module
import logging
from enum import Enum, auto
from dataclasses import dataclass
from .base import BasePrompt

logger = logging.getLogger(__name__)

class Task(Enum):
    """Available tasks"""
    COT_GENERATION = "cot_generation"
    SELF_EVALUATION = "self_evaluation"

    @property
    def dir_name(self) -> str:
        """Directory name for the task"""
        return self.value

class Dataset(Enum):
    """Supported datasets"""
    ARC = "arc"
    DROP = "drop"
    STRATEGY_QA = "strategy_qa"

    @property
    def class_name(self) -> str:
        """Class name for the prompt"""
        return f"{self.name}Prompt"

@dataclass
class PromptConfig:
    """Configuration for prompt loading"""
    task: Task
    dataset: Dataset
    
    @property
    def module_path(self) -> str:
        """Import path for the prompt module"""
        return f"prompts.{self.task.dir_name}.{self.dataset.value}"
    
    @property
    def prompt_class_name(self) -> str:
        """Class name for the prompt"""
        return self.dataset.class_name
    
class PromptManager:
    """Manager for task-specific and dataset-specific prompts"""
    
    def __init__(self):
        self.prompts: Dict[Task, Dict[Dataset, BasePrompt]] = {
            task: {} for task in Task
        }
        self._load_prompts()
    
    def _load_prompts(self):
        """Load all available prompt templates"""
        for task in Task:
            for dataset in Dataset:
                config = PromptConfig(task=task, dataset=dataset)
                self._load_prompt(config)
    
    def _load_prompt(self, config: PromptConfig):
        """Load a specific prompt template"""
        try:
            module = import_module(config.module_path)
            prompt_class = getattr(module, config.prompt_class_name)
            self.prompts[config.task][config.dataset] = prompt_class()
            logger.info(f"Loaded {config.dataset.value} prompt for {config.task.value}")
        except (ImportError, AttributeError) as e:
            logger.warning(
                f"Could not load prompt for {config.dataset.value} in "
                f"{config.task.value}: {e}"
            )

    def _format_choices(self, choices: Dict[str, List[str]]) -> str:
        """Format multiple choice options"""
        return "\n".join(
            f"{label}. {text}"
            for label, text in zip(choices["label"], choices["text"])
        )
    
    def get_prompt(self, task: Task, dataset: Dataset) -> BasePrompt:
        """Get prompt for specified task and dataset"""
        try:
            return self.prompts[task][dataset]
        except KeyError:
            raise ValueError(
                f"No prompt found for dataset {dataset.value} in task {task.value}"
            )

    def format_prompt(self, task: Task, dataset: Dataset, **kwargs) -> str:
        """Format prompt with given arguments"""
        prompt = self.get_prompt(task, dataset)
        return prompt.format(**kwargs)