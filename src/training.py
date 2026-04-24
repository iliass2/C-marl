from __future__ import annotations

import pandas as pd

from .agent import QLearningAgent
from .config import (
    ALPHA,
    CHECKPOINTS_DIR,
    EPSILON_DECAY,
    EPSILON_MIN,
    EPSILON_START,
    GAMMA,
    MAX_STEPS_PER_EPISODE,
    RESULTS_DIR,
    SEED,
    STAGE_CONFIGS,
    TRAIN_EPISODES,
)
from .environment import MedicalAppointmentEnv
from .utils import ensure_directories, set_seed


def _load_stage_training_data(stage_name: str) -> pd.DataFrame:
    stage_config = STAGE_CONFIGS[stage_name]
    return pd.read_csv(RESULTS_DIR / stage_config.train_filename)


def train_stage(
    stage_name: str,
    train_df: pd.DataFrame,
    *,
    episodes: int = TRAIN_EPISODES,
    max_steps: int = MAX_STEPS_PER_EPISODE,
    seed: int = SEED,
) -> tuple[dict[str, object], pd.DataFrame]:
    stage_config = STAGE_CONFIGS[stage_name]
    stage_seed = seed + list(STAGE_CONFIGS).index(stage_name)

    env = MedicalAppointmentEnv(
        train_df,
        stage=stage_name,
        shuffle=True,
        max_steps=max_steps,
        seed=stage_seed,
    )
    agent = QLearningAgent(
        stage=stage_name,
        actions=stage_config.actions,
        state_columns=stage_config.state_columns,
        alpha=ALPHA,
        gamma=GAMMA,
        epsilon=EPSILON_START,
        epsilon_decay=EPSILON_DECAY,
        epsilon_min=EPSILON_MIN,
        seed=stage_seed,
    )

    history_rows: list[dict[str, float | int]] = []
    for episode in range(1, episodes + 1):
        state = env.reset()
        total_reward = 0.0
        steps = 0

        while state is not None:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.update(state, action, reward, next_state, done)
            total_reward += reward
            steps += 1
            state = next_state
            if done:
                break

        average_reward = total_reward / steps if steps else 0.0
        history_rows.append(
            {
                "episode": episode,
                "total_reward": total_reward,
                "average_reward": average_reward,
                "epsilon": agent.epsilon,
            }
        )
        agent.decay_epsilon()

    history_df = pd.DataFrame(history_rows)
    checkpoint = agent.to_checkpoint_dict()
    return checkpoint, history_df


def train_all_stages(seed: int = SEED) -> dict[str, object]:
    set_seed(seed)
    ensure_directories()

    history_by_stage: dict[str, pd.DataFrame] = {}
    checkpoint_by_stage: dict[str, dict[str, object]] = {}

    for stage_name, stage_config in STAGE_CONFIGS.items():
        train_df = _load_stage_training_data(stage_name)
        checkpoint, history_df = train_stage(stage_name, train_df, seed=seed)

        history_df.to_csv(RESULTS_DIR / stage_config.rewards_filename, index=False)
        with (CHECKPOINTS_DIR / stage_config.checkpoint_filename).open("wb") as handle:
            import pickle

            pickle.dump(checkpoint, handle)

        history_by_stage[stage_name] = history_df
        checkpoint_by_stage[stage_name] = checkpoint

    return {
        "history": history_by_stage,
        "checkpoints": checkpoint_by_stage,
    }


if __name__ == "__main__":
    train_all_stages()
