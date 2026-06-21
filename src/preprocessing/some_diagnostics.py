import pandas as pd
from pathlib import Path
import sys

# Add src to path to import utilities
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.paths import DATA_PROCESSED
   
states = pd.read_parquet(DATA_PROCESSED / "game_states.parquet")
print(states.columns.tolist())
print(states.head())
print(states.dtypes)
print(f"Shape: {states.shape}")