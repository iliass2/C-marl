from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"
CHECKPOINTS_DIR = ROOT_DIR / "checkpoints"
FIGURES_DIR = RESULTS_DIR / "final_figures"
NOTEBOOKS_DIR = ROOT_DIR / "notebooks"
DATASET_PATH = DATA_DIR / "KaggleV2-May-2016.csv"

SEED = 42
TEST_SIZE = 0.20

TRAIN_EPISODES = 30
MAX_STEPS_PER_EPISODE = 5_000
ALPHA = 0.10
GAMMA = 0.95
EPSILON_START = 1.0
EPSILON_DECAY = 0.97
EPSILON_MIN = 0.05

TARGET_COLUMN = "No_show"
MEDICAL_RISK_BONUS = 0.25
RESCHEDULE_BONUS = 0.25
HIGH_WAITING_GROUP_THRESHOLD = 2


@dataclass(frozen=True)
class StageConfig:
    name: str
    state_columns: tuple[str, ...]
    actions: tuple[int, ...]
    action_names: dict[int, str]
    dataset_filename: str
    train_filename: str
    test_filename: str
    rewards_filename: str
    checkpoint_filename: str

    @property
    def output_columns(self) -> list[str]:
        return [*self.state_columns, TARGET_COLUMN]


STAGE_CONFIGS: dict[str, StageConfig] = {
    "easy": StageConfig(
        name="easy",
        state_columns=("AgeGroup", "Gender", "SMS_received"),
        actions=(0, 1),
        action_names={
            0: "do_nothing",
            1: "send_reminder",
        },
        dataset_filename="easy_data.csv",
        train_filename="train_easy.csv",
        test_filename="test_easy.csv",
        rewards_filename="rewards_easy.csv",
        checkpoint_filename="checkpoint_easy.pkl",
    ),
    "medium": StageConfig(
        name="medium",
        state_columns=(
            "AgeGroup",
            "Gender",
            "SMS_received",
            "Scholarship",
            "Hypertension",
            "Diabetes",
            "Alcoholism",
            "Handicap",
            "MedicalRisk",
        ),
        actions=(0, 1, 2),
        action_names={
            0: "do_nothing",
            1: "send_reminder",
            2: "prioritize_patient",
        },
        dataset_filename="medium_data.csv",
        train_filename="train_medium.csv",
        test_filename="test_medium.csv",
        rewards_filename="rewards_medium.csv",
        checkpoint_filename="checkpoint_medium.pkl",
    ),
    "hard": StageConfig(
        name="hard",
        state_columns=(
            "AgeGroup",
            "Gender",
            "SMS_received",
            "Scholarship",
            "Hypertension",
            "Diabetes",
            "Alcoholism",
            "Handicap",
            "MedicalRisk",
            "WaitingGroup",
            "AppointmentWeekday",
            "Neighbourhood",
        ),
        actions=(0, 1, 2, 3),
        action_names={
            0: "do_nothing",
            1: "send_reminder",
            2: "prioritize_patient",
            3: "propose_rescheduling",
        },
        dataset_filename="hard_data.csv",
        train_filename="train_hard.csv",
        test_filename="test_hard.csv",
        rewards_filename="rewards_hard.csv",
        checkpoint_filename="checkpoint_hard.pkl",
    ),
}
