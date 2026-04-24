from __future__ import annotations

from src.config import ROOT_DIR, SEED
from src.evaluation import evaluate_all_stages
from src.plotting import generate_outputs
from src.preprocessing import preprocess_and_save
from src.training import train_all_stages
from src.utils import clean_generated_outputs, ensure_directories, remove_notebook_checkpoints, set_seed


def main() -> None:
    set_seed(SEED)
    ensure_directories()
    remove_notebook_checkpoints(ROOT_DIR)
    clean_generated_outputs()

    preprocess_and_save(seed=SEED)
    train_all_stages(seed=SEED)
    evaluate_all_stages(seed=SEED)
    generate_outputs()

    print("Pipeline completed successfully.")


if __name__ == "__main__":
    main()
