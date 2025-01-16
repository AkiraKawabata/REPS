from typing import List, Dict, Optional
from vllm import LLM, SamplingParams
import logging
import warnings

from prompts import PromptManager, Task, Dataset

logger = logging.getLogger(__name__)

class GenerationModel:
    """Chain-of-Thought model using vLLM for efficient generation"""

    def __init__(
        self,
        model_name: str,
        model_config: Dict,
        dataset_name: str,
        sampling_params: Dict[str, int],
        cache_dir: Optional[str] = None
    ):
        self.model_name = model_name
        self.config = model_config
        self.dataset_name = dataset_name
        self.cache_dir = cache_dir

        try:
            self.dataset = Dataset[dataset_name.upper()]
        except KeyError:
            warnings.warn(
                f"Unsupported dataset: '{dataset_name}'. "
                "PromptManager / dataset-specific logic may be limited."
            )
            self.dataset = None

        # Initialize prompt manager
        self.prompt_manager = PromptManager()

        # Initialize model (vLLM)
        self.llm = self._init_model()
        self.sampling_params = SamplingParams(**sampling_params)

    def _init_model(self) -> LLM:
        """Initialize the vLLM model with config parameters"""
        try:
            llm = LLM(
                model=self.model_name,
                dtype=self.config.get("dtype", "bfloat16"),
                download_dir=self.cache_dir,
                trust_remote_code=self.config.get("trust_remote_code", True),
                tensor_parallel_size=self.config.get("tensor_parallel_size", 1)
            )
            logger.info(f"Successfully initialized model: {self.model_name}")
            return llm
        except Exception as e:
            logger.error(f"Failed to initialize model: {str(e)}")
            raise

    def generate(
        self,
        question: str = "",
        choices: Optional[Dict] = None,
        passage: Optional[str] = None,
        num_iterations: int = 1,
        batch_size: int = 8,
        task: Optional[str] = None
    ) -> List[str]:
        """
        Generate responses using prompt templates. 
        
        Args:
            question: Input question text
            choices: Dictionary containing multiple choice options
            passage: Optional context passage
            num_iterations: Number of responses to generate
            batch_size: Batch size for generation
            task: Task type for prompt selection
            
        Returns:
            List of generated responses
        """
        try:
            task_enum = Task.COT_GENERATION if task is None else Task[task.upper()]

            # Generate prompts
            prompts = []
            for _ in range(num_iterations):
                format_args = {"question": question}
                if choices:
                    format_args["choices"] = self._format_choices(choices)
                if passage:
                    format_args["passage"] = passage

                if not self.dataset:
                    raise ValueError("Dataset is required for choice of prompt template.")
                dataset_for_prompt = self.dataset
                # breakpoint()
                prompt_text = self.prompt_manager.format_prompt(
                    task=task_enum,
                    dataset=dataset_for_prompt,
                    **format_args
                )
                prompts.append(prompt_text)

            return self._generate_batch(prompts, batch_size)

        except Exception as e:
            logger.error(f"Generation failed: {str(e)}")
            raise

    def generate_raw_text(
        self,
        prompt_text: str,
        num_iterations: int = 1,
        batch_size: int = 1
    ) -> List[Dict[str, str]]:
        """
        Generate responses using raw prompt text without template processing.
        
        Args:
            prompt_text: Complete prompt text to use directly
            num_iterations: Number of responses to generate
            batch_size: Batch size for generation
            
        Returns:
            List of dictionaries containing explanation and answer
        """
        try:
            prompts = [prompt_text] * num_iterations
            return self._generate_eval_batch(prompts, batch_size)
        except Exception as e:
            logger.error(f"Raw generation failed: {str(e)}")
            raise

    def _generate_batch(
        self,
        prompts: List[str],
        batch_size: int
    ) -> List[Dict[str, str]]:
        """
        Generate responses in batches.
        
        Args:
            prompts: List of prompt texts
            batch_size: Size of each generation batch
            
        Returns:
            List of parsed responses
        """
        all_responses = []
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i : i + batch_size]
            outputs = self.llm.generate(batch, self.sampling_params)
            responses = [output.outputs[0].text for output in outputs]
            parsed_responses = [self._parse_text(r) for r in responses]
            all_responses.extend(parsed_responses)

        logger.info(f"Successfully generated {len(all_responses)} responses")
        return all_responses
    
    def _generate_eval_batch(
        self,
        prompts: List[str],
        batch_size: int
    ) -> List[Dict[str, str]]:
        """
        Generate responses in batches.
        
        Args:
            prompts: List of prompt texts
            batch_size: Size of each generation batch
            
        Returns:
            List of parsed responses
        """
        all_responses = []
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i : i + batch_size]
            outputs = self.llm.generate(batch, self.sampling_params)
            responses = [output.outputs[0].text for output in outputs]
            parsed_responses = [self._parse_eval_text(r) for r in responses]
            all_responses.extend(parsed_responses)

        logger.info(f"Successfully generated {len(all_responses)} responses")
        return all_responses

    def _format_choices(self, choices: Dict[str, List[str]]) -> str:
        """Format multiple choice options"""
        return "\n".join(
            f"{label}. {text}"
            for label, text in zip(choices["label"], choices["text"])
        )

    def _parse_text(self, text: str) -> Dict[str, str]:
        """Parse generated text into explanation and answer components"""
        explanation_start = text.find("Explanation:")
        answer_start = text.find("Answer:")

        if explanation_start == -1 or answer_start == -1:
            return {"explanation": "", "answer": ""}

        explanation = text[explanation_start + len("Explanation:"):answer_start].strip()
        answer = self._extract_answer(text[answer_start + len("Answer:"):])
        return {"explanation": explanation, "answer": answer}
    
    def _parse_eval_text(self, text: str) -> Dict[str, str]:
        """ Parse generated text by splitting at the first newline character """
        explanation = text.split("\n")[0].strip()
        return {"eval_result": explanation}

    def _extract_answer(self, answer_text: str) -> str:
        lines = [line.strip() for line in answer_text.splitlines() if line.strip()]
        if not lines:
            return ""
        return lines[0]