"""Settings shared across src/, the notebooks and the figure scripts."""

from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "datasets"
MODEL_DIR = DATA_DIR.parent / "models"
LOG_DIR = DATA_DIR.parent / "logs"

N_STRIKES = 10

# maturity window -> (T_min, T_max, seeds for train/val/test)
WINDOWS = {"1d_1y": (1 / 365, 1.0, (42, 43, 44)),
           "1y_2y": (1.0, 2.0, (45, 46, 47))}

SIZES = (250, 500, 750, 1000)  # ANN hidden-layer widths
