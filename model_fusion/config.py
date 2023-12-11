import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DATA_DIR=Path(os.getenv("BASE_DATA_DIR", "./data/"))

NUM_WORKERS = int(os.getenv("NUM_WORKERS", 0))

WANDB_PROJECT_NAME = os.getenv("WANDB_PROJECT_NAME", "Model Fusion")
