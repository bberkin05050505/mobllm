import yaml
import subprocess
from pathlib import Path

# Path to your config file and script
CONFIG_PATH = Path("conf/experiment/standard.yaml")
MAIN_SCRIPT = "main.py"
PYTHON_EXE_PATH = "" # FILL HERE

# The functions to sweep over
functions = ["nguyen/nguyen4", "nguyen/nguyen5", "nguyen/nguyen6", "nguyen/nguyen7", "nguyen/nguyen8"]

for fn in functions:
    # Load the current YAML
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)

    # Modify the function field in defaults
    for d in cfg["defaults"]:
        if isinstance(d, dict) and "function" in d:
            d["function"] = fn
            break

    # Save modified YAML back
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(cfg, f, sort_keys=False)

    print(f"\n▶ Running {MAIN_SCRIPT} with function: {fn}")
    subprocess.run([PYTHON_EXE_PATH, MAIN_SCRIPT])