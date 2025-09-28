from pathlib import Path

SEED = 42
N_NEIGHBORS = 5
N_REP_MFCM = 50

# Configurações de validação cruzada
N_FOLDS = 5
VAR_PERCENTAGES = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

# Configurações de diretórios
BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"

# Datasets a serem utilizados
DATASETS = [3]

