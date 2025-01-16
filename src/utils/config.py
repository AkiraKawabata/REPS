from pathlib import Path
import yaml
from typing import Dict, Any

def load_config(config_path: str) -> Dict[str, Any]:
    """Load and merge configuration files
    
    Args:
        config_path: Path to main config file
        
    Returns:
        Dict containing merged configuration
    """
    config_path = Path(config_path)
    
    # Load main config
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    # Load dataset-specific config if exists
    if "dataset" in config:
        dataset_config_path = config_path.parent / "datasets" / f"{config['dataset']}.yaml"
        if dataset_config_path.exists():
            with open(dataset_config_path, "r", encoding="utf-8") as f:
                dataset_config = yaml.safe_load(f)
                config = deep_merge(config, dataset_config)
    
    return config

def deep_merge(base: Dict, update: Dict) -> Dict:
    """Recursively merge two dictionaries
    
    Args:
        base: Base dictionary
        update: Dictionary to merge into base
        
    Returns:
        Merged dictionary
    """
    merged = base.copy()
    
    for key, value in update.items():
        if (
            key in base and 
            isinstance(base[key], dict) and 
            isinstance(value, dict)
        ):
            merged[key] = deep_merge(base[key], value)
        else:
            merged[key] = value
            
    return merged