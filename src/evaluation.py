from __future__ import annotations

import numpy as np
import pandas as pd

from .agent import load_checkpoint
from .config import CHECKPOINTS_DIR, RESULTS_DIR, SEED, STAGE_CONFIGS
from .environment import MedicalAppointmentEnv
from .utils import ensure_directories, set_seed


def _load_stage_test_data(stage_name: str) -> pd.DataFrame:
    stage_config = STAGE_CONFIGS[stage_name]
    return pd.read_csv(RESULTS_DIR / stage_config.test_filename)


def _choose_best_action(
    checkpoint: dict[str, object],
    state: tuple[int, ...],
) -> int:
    q_table = checkpoint["q_table"]
    actions = [int(action) for action in checkpoint["actions"]]
    if state not in q_table:
        return 0
    q_values = np.asarray(q_table[state], dtype=float)
    return actions[int(np.argmax(q_values))]


def evaluate_stage(stage_name: str, test_df: pd.DataFrame) -> dict[str, object]:
    stage_config = STAGE_CONFIGS[stage_name]
    checkpoint = load_checkpoint(CHECKPOINTS_DIR / stage_config.checkpoint_filename)

    env = MedicalAppointmentEnv(test_df, stage=stage_name, shuffle=False, seed=int(checkpoint["seed"]))
    state = env.reset()

    total_reward = 0.0
    total_steps = 0
    good_decisions = 0
    interventions = 0
    useful_interventions = 0
    unnecessary_interventions = 0

    while state is not None:
        action = _choose_best_action(checkpoint, state)
        next_state, reward, done, info = env.step(action)

        total_reward += reward
        total_steps += 1
        good_decisions += int(info["good_decision"])
        interventions += int(info["intervention"])
        useful_interventions += int(info["useful_intervention"])
        unnecessary_interventions += int(info["unnecessary_intervention"])

        state = next_state
        if done:
            break

    average_reward = total_reward / total_steps if total_steps else 0.0
    good_decision_rate = good_decisions / total_steps if total_steps else 0.0
    intervention_rate = interventions / total_steps if total_steps else 0.0
    useful_intervention_rate = useful_interventions / total_steps if total_steps else 0.0
    unnecessary_intervention_rate = unnecessary_interventions / total_steps if total_steps else 0.0

    return {
        "Stage": stage_name.capitalize(),
        "Total Reward": total_reward,
        "Average Reward": average_reward,
        "Total Steps": total_steps,
        "Good Decisions": good_decisions,
        "Good Decision Rate": good_decision_rate,
        "Interventions": interventions,
        "Intervention Rate": intervention_rate,
        "Useful Interventions": useful_interventions,
        "Useful Intervention Rate": useful_intervention_rate,
        "Unnecessary Interventions": unnecessary_interventions,
        "Unnecessary Intervention Rate": unnecessary_intervention_rate,
    }


def build_hard_gain_summary(comparison_df: pd.DataFrame) -> pd.DataFrame:
    indexed = comparison_df.set_index("Stage")
    metrics = [
        ("Average Reward", "higher_is_better"),
        ("Good Decision Rate", "higher_is_better"),
        ("Useful Intervention Rate", "higher_is_better"),
        ("Unnecessary Intervention Rate", "lower_is_better"),
    ]

    rows: list[dict[str, object]] = []
    for metric, direction in metrics:
        hard_value = float(indexed.loc["Hard", metric])
        easy_value = float(indexed.loc["Easy", metric])
        medium_value = float(indexed.loc["Medium", metric])
        rows.append(
            {
                "Metric": metric,
                "Direction": direction,
                "Hard": hard_value,
                "Easy": easy_value,
                "Medium": medium_value,
                "Hard_minus_Easy": hard_value - easy_value,
                "Hard_minus_Medium": hard_value - medium_value,
            }
        )
    return pd.DataFrame(rows)


def _build_summary_text(comparison_df: pd.DataFrame, gain_df: pd.DataFrame) -> str:
    indexed = comparison_df.set_index("Stage")
    best_average_reward_stage = indexed["Average Reward"].idxmax()
    best_good_decision_stage = indexed["Good Decision Rate"].idxmax()
    lowest_unnecessary_stage = indexed["Unnecessary Intervention Rate"].idxmin()

    lines = [
        "Evaluation summary for the educational appointment no-show simulation",
        "",
        comparison_df.to_string(index=False),
        "",
        "Hard gain summary",
        gain_df.to_string(index=False),
        "",
        f"Highest average reward stage: {best_average_reward_stage}",
        f"Highest good decision rate stage: {best_good_decision_stage}",
        f"Lowest unnecessary intervention rate stage: {lowest_unnecessary_stage}",
    ]

    if (
        best_average_reward_stage == "Hard"
        and best_good_decision_stage == "Hard"
        and lowest_unnecessary_stage == "Hard"
    ):
        lines.append("Hard is strongest across the tracked metrics on the held-out test split.")
    elif best_average_reward_stage == "Hard":
        lines.append(
            "Hard has the highest average reward on the held-out test split, but the other metrics must also be considered."
        )
    else:
        lines.append("Hard does not dominate the held-out test metrics in this fair evaluation.")

    return "\n".join(lines)


def evaluate_all_stages(seed: int = SEED) -> dict[str, pd.DataFrame]:
    set_seed(seed)
    ensure_directories()

    comparison_rows = []
    for stage_name in STAGE_CONFIGS:
        test_df = _load_stage_test_data(stage_name)
        comparison_rows.append(evaluate_stage(stage_name, test_df))

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(RESULTS_DIR / "evaluation_comparison.csv", index=False)

    gain_df = build_hard_gain_summary(comparison_df)
    gain_df.to_csv(RESULTS_DIR / "hard_gain_summary.csv", index=False)

    summary_text = _build_summary_text(comparison_df, gain_df)
    (RESULTS_DIR / "evaluation_summary.txt").write_text(summary_text, encoding="utf-8")

    return {
        "comparison": comparison_df,
        "hard_gain": gain_df,
    }


if __name__ == "__main__":
    evaluate_all_stages()
