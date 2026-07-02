import os
import json

def get_best_model_config():
    """
    Finds and loads the best_model.json configuration file.
    Traverses up from the directory containing this file until it finds best_model.json,
    or falls back to checking the current working directory, then the parent directories.
    If not found, returns the default fallback.
    """
    # Start traversing up from the current file's directory
    start_dirs = [
        os.path.dirname(os.path.abspath(__file__)),
        os.getcwd()
    ]
    
    for start_dir in start_dirs:
        current_dir = start_dir
        while current_dir and current_dir != os.path.dirname(current_dir):
            config_path = os.path.join(current_dir, "best_model.json")
            if os.path.exists(config_path):
                try:
                    with open(config_path, "r", encoding="utf-8") as f:
                        config = json.load(f)
                        if "repo" in config and "revision" in config:
                            return config
                except Exception as e:
                    print(f"Warning: Failed to load best_model.json from {config_path}: {e}")
            current_dir = os.path.dirname(current_dir)
            
    # Fallback default values
    return {
        "repo": "charliemcvicker/asr-cherokee",
        "revision": "5464d15"
    }
