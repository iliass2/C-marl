from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .config import FIGURES_DIR, RESULTS_DIR, STAGE_CONFIGS
from .utils import ensure_directories


def _load_evaluation_outputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    comparison_df = pd.read_csv(RESULTS_DIR / "evaluation_comparison.csv")
    gain_df = pd.read_csv(RESULTS_DIR / "hard_gain_summary.csv")
    return comparison_df, gain_df


def _load_training_histories() -> dict[str, pd.DataFrame]:
    histories: dict[str, pd.DataFrame] = {}
    for stage_name, stage_config in STAGE_CONFIGS.items():
        histories[stage_name.capitalize()] = pd.read_csv(RESULTS_DIR / stage_config.rewards_filename)
    return histories


def _save_bar_plot(
    comparison_df: pd.DataFrame,
    *,
    column: str,
    filename: str,
    title: str,
    ylabel: str,
) -> None:
    plt.figure(figsize=(8, 5))
    plt.bar(comparison_df["Stage"], comparison_df[column])
    plt.title(title)
    plt.xlabel("Stage")
    plt.ylabel(ylabel)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=300)
    plt.close()


def _build_final_report_text(comparison_df: pd.DataFrame, gain_df: pd.DataFrame) -> str:
    indexed = comparison_df.set_index("Stage")
    hard_is_best_avg = indexed["Average Reward"].idxmax() == "Hard"
    hard_is_best_good = indexed["Good Decision Rate"].idxmax() == "Hard"
    hard_has_lowest_unnecessary = indexed["Unnecessary Intervention Rate"].idxmin() == "Hard"

    conclusion_lines = []
    if hard_is_best_avg and hard_is_best_good and hard_has_lowest_unnecessary:
        conclusion_lines.append(
            "Based on the held-out test metrics, Hard is the strongest stage across reward quality and intervention discipline."
        )
    elif hard_is_best_avg:
        conclusion_lines.append(
            "Hard reaches the highest average reward on the held-out test split, but it does not dominate every metric."
        )
        if not hard_has_lowest_unnecessary:
            best_stage = indexed["Unnecessary Intervention Rate"].idxmin()
            conclusion_lines.append(
                f"{best_stage} has the lowest unnecessary intervention rate, so Hard is not unambiguously best overall."
            )
        if not hard_is_best_good:
            best_stage = indexed["Good Decision Rate"].idxmax()
            conclusion_lines.append(
                f"{best_stage} has the highest good decision rate, which should also influence the interpretation."
            )
    else:
        best_stage = indexed["Average Reward"].idxmax()
        conclusion_lines.append(
            f"Hard does not have the highest average reward on the held-out test split; {best_stage} performs better on that metric."
        )

    lines = [
        "Project: Educational reinforcement-learning simulation for medical appointment no-show interventions",
        "",
        "Important note",
        "This repository is a pedagogical simulation. It is not a clinical system and must not be used for real patient care decisions.",
        "",
        "Method honesty",
        "The implementation trains one tabular Q-learning agent per stage.",
        "It is best described as a simplified educational approximation of cooperative decision support, not as a true interacting multi-agent RL system.",
        "",
        "Reproducibility",
        "The pipeline uses fixed seeds, relative paths, reusable Python modules, portable checkpoints, and an explicit train/test split.",
        "",
        "Reward design",
        "Base reward is normalized across stages: +1 for a useful intervention or correctly doing nothing, and -1 for a missed no-show or unnecessary intervention.",
        "Small bonuses are only used for targeted medium/hard actions such as prioritizing high-risk absent patients or rescheduling high-wait absent patients.",
        "",
        "Evaluation results",
        comparison_df.to_string(index=False),
        "",
        "Hard gain summary",
        gain_df.to_string(index=False),
        "",
        "Conclusion",
        *conclusion_lines,
    ]
    return "\n".join(lines)


def generate_outputs() -> dict[str, pd.DataFrame]:
    ensure_directories()
    comparison_df, gain_df = _load_evaluation_outputs()

    _save_bar_plot(
        comparison_df,
        column="Average Reward",
        filename="final_average_reward_by_stage.png",
        title="Average Reward by Stage",
        ylabel="Average Reward",
    )
    _save_bar_plot(
        comparison_df,
        column="Total Reward",
        filename="final_total_reward_by_stage.png",
        title="Total Reward by Stage",
        ylabel="Total Reward",
    )
    _save_bar_plot(
        comparison_df,
        column="Good Decision Rate",
        filename="final_good_decision_rate_by_stage.png",
        title="Good Decision Rate by Stage",
        ylabel="Good Decision Rate",
    )
    _save_bar_plot(
        comparison_df,
        column="Unnecessary Intervention Rate",
        filename="final_unnecessary_intervention_rate_by_stage.png",
        title="Unnecessary Intervention Rate by Stage",
        ylabel="Unnecessary Intervention Rate",
    )

    histories = _load_training_histories()
    plt.figure(figsize=(10, 6))
    for stage_name, history_df in histories.items():
        plt.plot(history_df["episode"], history_df["average_reward"], label=stage_name)
    plt.title("Training Rewards by Stage")
    plt.xlabel("Episode")
    plt.ylabel("Average Reward per Episode")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "final_training_rewards_by_stage.png", dpi=300)
    plt.close()

    report_text = _build_final_report_text(comparison_df, gain_df)
    (RESULTS_DIR / "final_report_synthesis.txt").write_text(report_text, encoding="utf-8")

    return {
        "comparison": comparison_df,
        "hard_gain": gain_df,
    }


if __name__ == "__main__":
    generate_outputs()
