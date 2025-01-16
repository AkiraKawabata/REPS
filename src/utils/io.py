import json
from pathlib import Path
from typing import Dict, List, Union

def load_jsonl(file_path: Union[str, Path]) -> List[Dict]:
    """Load JSONL file"""
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]
    
def save_jsonl(data: List[Dict], file_path: Union[str, Path]):
    """Save data to JSONL file"""
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")