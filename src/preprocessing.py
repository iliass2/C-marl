from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from .config import DATASET_PATH, RESULTS_DIR, SEED, STAGE_CONFIGS, TARGET_COLUMN, TEST_SIZE
from .utils import ensure_directories, set_seed


def load_raw_dataset(path: Path = DATASET_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned = cleaned.rename(
        columns={
            "Hipertension": "Hypertension",
            "Handcap": "Handicap",
            "No-show": TARGET_COLUMN,
        }
    )

    cleaned["ScheduledDay"] = pd.to_datetime(cleaned["ScheduledDay"], errors="coerce")
    cleaned["AppointmentDay"] = pd.to_datetime(cleaned["AppointmentDay"], errors="coerce")
    cleaned = cleaned.dropna(subset=["ScheduledDay", "AppointmentDay"]).copy()

    cleaned["WaitingDays"] = (
        cleaned["AppointmentDay"].dt.normalize() - cleaned["ScheduledDay"].dt.normalize()
    ).dt.days
    cleaned["AppointmentWeekday"] = cleaned["AppointmentDay"].dt.dayofweek.astype(int)

    cleaned = cleaned.loc[(cleaned["Age"] >= 0) & (cleaned["WaitingDays"] >= 0)].copy()

    cleaned[TARGET_COLUMN] = cleaned[TARGET_COLUMN].map({"No": 0, "Yes": 1})
    cleaned = cleaned.dropna(subset=[TARGET_COLUMN]).copy()
    cleaned[TARGET_COLUMN] = cleaned[TARGET_COLUMN].astype(int)

    gender_encoder = LabelEncoder()
    neighbourhood_encoder = LabelEncoder()
    cleaned["Gender"] = gender_encoder.fit_transform(cleaned["Gender"].astype(str))
    cleaned["Neighbourhood"] = neighbourhood_encoder.fit_transform(
        cleaned["Neighbourhood"].astype(str)
    )

    cleaned["AgeGroup"] = pd.cut(
        cleaned["Age"],
        bins=[-1, 18, 40, 60, 120],
        labels=[0, 1, 2, 3],
        include_lowest=True,
    ).astype(int)

    cleaned["WaitingGroup"] = pd.cut(
        cleaned["WaitingDays"],
        bins=[-1, 7, 15, 30, np.inf],
        labels=[0, 1, 2, 3],
        include_lowest=True,
    ).astype(int)

    cleaned["MedicalRisk"] = (
        cleaned["Hypertension"]
        + cleaned["Diabetes"]
        + cleaned["Alcoholism"]
        + cleaned["Handicap"]
    )
    cleaned["MedicalRisk"] = (cleaned["MedicalRisk"] > 0).astype(int)

    return cleaned.reset_index(drop=True)


def _build_stage_dataset(cleaned_df: pd.DataFrame, stage_name: str) -> pd.DataFrame:
    stage_config = STAGE_CONFIGS[stage_name]
    return cleaned_df.loc[:, stage_config.output_columns].copy()


def split_clean_data(
    cleaned_df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    seed: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_index, test_index = train_test_split(
        cleaned_df.index,
        test_size=test_size,
        random_state=seed,
        stratify=cleaned_df[TARGET_COLUMN],
    )
    train_df = cleaned_df.loc[sorted(train_index)].reset_index(drop=True)
    test_df = cleaned_df.loc[sorted(test_index)].reset_index(drop=True)
    return train_df, test_df


def preprocess_and_save(seed: int = SEED) -> dict[str, object]:
    set_seed(seed)
    ensure_directories()

    raw_df = load_raw_dataset()
    cleaned_df = preprocess_dataframe(raw_df)
    train_clean, test_clean = split_clean_data(cleaned_df, seed=seed)

    saved_splits: dict[str, dict[str, pd.DataFrame]] = {}
    summary_rows: list[dict[str, object]] = []

    for stage_name, stage_config in STAGE_CONFIGS.items():
        full_df = _build_stage_dataset(cleaned_df, stage_name)
        train_df = _build_stage_dataset(train_clean, stage_name)
        test_df = _build_stage_dataset(test_clean, stage_name)

        full_df.to_csv(RESULTS_DIR / stage_config.dataset_filename, index=False)
        train_df.to_csv(RESULTS_DIR / stage_config.train_filename, index=False)
        test_df.to_csv(RESULTS_DIR / stage_config.test_filename, index=False)

        saved_splits[stage_name] = {
            "full": full_df,
            "train": train_df,
            "test": test_df,
        }
        summary_rows.append(
            {
                "Stage": stage_name.capitalize(),
                "State Columns": len(stage_config.state_columns),
                "Full Rows": len(full_df),
                "Train Rows": len(train_df),
                "Test Rows": len(test_df),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    return {
        "cleaned": cleaned_df,
        "train_clean": train_clean,
        "test_clean": test_clean,
        "stage_splits": saved_splits,
        "summary": summary_df,
    }


if __name__ == "__main__":
    results = preprocess_and_save()
    print(results["summary"].to_string(index=False))
