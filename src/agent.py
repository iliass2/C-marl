from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np


class QLearningAgent:
    def __init__(
        self,
        *,
        stage: str,
        actions: tuple[int, ...],
        state_columns: tuple[str, ...],
        alpha: float,
        gamma: float,
        epsilon: float,
        epsilon_decay: float,
        epsilon_min: float,
        seed: int,
    ) -> None:
        self.stage = stage
        self.actions = tuple(actions)
        self.action_to_index = {action: index for index, action in enumerate(self.actions)}
        self.state_columns = tuple(state_columns)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.q_table: dict[tuple[int, ...], np.ndarray] = {}

    def _init_state(self, state: tuple[int, ...]) -> None:
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(self.actions), dtype=float)

    def choose_action(self, state: tuple[int, ...]) -> int:
        self._init_state(state)
        if self.rng.random() < self.epsilon:
            return int(self.rng.choice(self.actions))
        best_index = int(np.argmax(self.q_table[state]))
        return self.actions[best_index]

    def choose_greedy_action(self, state: tuple[int, ...]) -> int:
        if state not in self.q_table:
            return 0
        best_index = int(np.argmax(self.q_table[state]))
        return self.actions[best_index]

    def update(
        self,
        state: tuple[int, ...],
        action: int,
        reward: float,
        next_state: tuple[int, ...] | None,
        done: bool,
    ) -> None:
        self._init_state(state)
        action_index = self.action_to_index[action]
        current_q = self.q_table[state][action_index]

        if done or next_state is None:
            target_q = reward
        else:
            self._init_state(next_state)
            target_q = reward + self.gamma * float(np.max(self.q_table[next_state]))

        self.q_table[state][action_index] = current_q + self.alpha * (target_q - current_q)

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def to_checkpoint_dict(self) -> dict[str, object]:
        serialized_q_table = {
            tuple(map(int, state)): values.astype(float).tolist()
            for state, values in self.q_table.items()
        }
        return {
            "stage": self.stage,
            "q_table": serialized_q_table,
            "actions": list(self.actions),
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_decay": self.epsilon_decay,
            "epsilon_min": self.epsilon_min,
            "state_columns": list(self.state_columns),
            "seed": self.seed,
        }

    def save_checkpoint(self, path: Path) -> None:
        with path.open("wb") as handle:
            pickle.dump(self.to_checkpoint_dict(), handle)

    @classmethod
    def from_checkpoint(cls, checkpoint: dict[str, object]) -> "QLearningAgent":
        agent = cls(
            stage=str(checkpoint["stage"]),
            actions=tuple(int(action) for action in checkpoint["actions"]),
            state_columns=tuple(str(column) for column in checkpoint["state_columns"]),
            alpha=float(checkpoint["alpha"]),
            gamma=float(checkpoint["gamma"]),
            epsilon=float(checkpoint["epsilon"]),
            epsilon_decay=float(checkpoint["epsilon_decay"]),
            epsilon_min=float(checkpoint["epsilon_min"]),
            seed=int(checkpoint["seed"]),
        )
        agent.q_table = {
            tuple(int(value) for value in state): np.asarray(values, dtype=float)
            for state, values in dict(checkpoint["q_table"]).items()
        }
        return agent


def load_checkpoint(path: Path) -> dict[str, object]:
    with path.open("rb") as handle:
        checkpoint = pickle.load(handle)
    if not isinstance(checkpoint, dict) or "q_table" not in checkpoint:
        raise ValueError(f"Unsupported checkpoint format in {path}.")
    return checkpoint
