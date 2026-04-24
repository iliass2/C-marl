from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import (
    HIGH_WAITING_GROUP_THRESHOLD,
    MEDICAL_RISK_BONUS,
    RESCHEDULE_BONUS,
    STAGE_CONFIGS,
    TARGET_COLUMN,
)


@dataclass(frozen=True)
class DecisionInfo:
    action: int
    action_name: str
    intervention: bool
    useful_intervention: bool
    unnecessary_intervention: bool
    good_decision: bool
    targeted_bonus: float
    reward: float


class MedicalAppointmentEnv:
    def __init__(
        self,
        data: pd.DataFrame,
        stage: str,
        *,
        shuffle: bool = False,
        max_steps: int | None = None,
        seed: int | None = None,
    ) -> None:
        self.stage_config = STAGE_CONFIGS[stage.lower()]
        self.data = data.reset_index(drop=True).copy(deep=True)
        self.state_columns = list(self.stage_config.state_columns)
        self.actions = tuple(self.stage_config.actions)
        self.action_names = dict(self.stage_config.action_names)
        self.shuffle = shuffle
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)

        self.order = np.arange(len(self.data))
        self.current_position = 0

    def reset(self) -> tuple[int, ...] | None:
        self.current_position = 0
        self.order = np.arange(len(self.data))
        if self.shuffle:
            self.order = self.rng.permutation(len(self.data))
        if self.max_steps is not None:
            self.order = self.order[: min(self.max_steps, len(self.order))]
        return self.get_state()

    def is_done(self) -> bool:
        return self.current_position >= len(self.order)

    def current_patient(self) -> pd.Series:
        if self.is_done():
            raise RuntimeError("The episode is finished. Call reset() before stepping again.")
        row_index = int(self.order[self.current_position])
        return self.data.iloc[row_index]

    def get_state(self) -> tuple[int, ...] | None:
        if self.is_done():
            return None
        patient = self.current_patient()
        return tuple(int(patient[column]) for column in self.state_columns)

    def _validate_action(self, action: int) -> None:
        if action not in self.actions:
            raise ValueError(f"Invalid action {action}. Allowed actions: {self.actions}.")

    def evaluate_action(self, patient: pd.Series, action: int) -> DecisionInfo:
        self._validate_action(action)

        no_show = int(patient[TARGET_COLUMN])
        intervention = action != 0
        useful_intervention = no_show == 1 and intervention
        unnecessary_intervention = no_show == 0 and intervention
        good_decision = useful_intervention or (no_show == 0 and action == 0)

        reward = 1.0 if good_decision else -1.0
        targeted_bonus = 0.0

        if useful_intervention and action == 2 and "MedicalRisk" in patient.index:
            if int(patient["MedicalRisk"]) == 1:
                targeted_bonus += MEDICAL_RISK_BONUS

        if useful_intervention and action == 3 and "WaitingGroup" in patient.index:
            if int(patient["WaitingGroup"]) >= HIGH_WAITING_GROUP_THRESHOLD:
                targeted_bonus += RESCHEDULE_BONUS

        reward += targeted_bonus
        return DecisionInfo(
            action=action,
            action_name=self.action_names[action],
            intervention=intervention,
            useful_intervention=useful_intervention,
            unnecessary_intervention=unnecessary_intervention,
            good_decision=good_decision,
            targeted_bonus=targeted_bonus,
            reward=reward,
        )

    def step(self, action: int) -> tuple[tuple[int, ...] | None, float, bool, dict[str, object]]:
        patient = self.current_patient()
        info = self.evaluate_action(patient, action)

        self.current_position += 1
        done = self.is_done()
        next_state = None if done else self.get_state()

        return next_state, info.reward, done, {
            "action": info.action,
            "action_name": info.action_name,
            "intervention": info.intervention,
            "useful_intervention": info.useful_intervention,
            "unnecessary_intervention": info.unnecessary_intervention,
            "good_decision": info.good_decision,
            "targeted_bonus": info.targeted_bonus,
            "reward": info.reward,
        }

    def get_action_space(self) -> int:
        return len(self.actions)

    def get_action_names(self) -> dict[int, str]:
        return dict(self.action_names)

    def get_number_of_patients(self) -> int:
        return len(self.order)
