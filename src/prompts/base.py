from abc import ABC
from typing import Dict, Optional

class BasePrompt(ABC):
    """Base class for all prompt templates"""
    
    def __init__(self):
        self.instruction: str = ""
        self.examples: str = ""
        self.template: str = ""
    
    def format(self, **kwargs) -> str:
        """Format the template with given arguments"""
        return self.template.format(**kwargs)
    
    def get_components(self) -> Dict[str, str]:
        """Get all prompt components"""
        return {
            "instruction": self.instruction,
            "examples": self.examples,
            "template": self.template
        }