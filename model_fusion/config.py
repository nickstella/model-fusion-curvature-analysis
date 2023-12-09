import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# TODO: Add our own config here

# Example, remove this later
DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
