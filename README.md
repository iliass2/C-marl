# Medical Appointment No-Show RL Simulation

This repository is a pedagogical reinforcement-learning project built around the Kaggle dataset `Medical Appointment No Shows - KaggleV2-May-2016.csv`.

The goal is to simulate simple appointment-management interventions such as:

- do nothing
- send reminder
- prioritize patient
- propose rescheduling

The project compares three stages of increasing state complexity:

- `Easy`
- `Medium`
- `Hard`

## Important Disclaimer

This project is an educational simulation only.

- It is **not** a medical device.
- It is **not** a clinical decision-support system.
- It must **not** be used for real patient care decisions.

## Method Honesty

The implementation trains **one tabular Q-learning agent per stage**.

That means the repository is best described as a **simplified educational approximation** of cooperative decision support rather than a true interacting multi-agent RL system. The older “C-MARL” wording is kept only as project context; the actual code is single-agent tabular Q-learning.

## Dataset Source

- Kaggle: Medical Appointment No Shows
- File used here: `data/KaggleV2-May-2016.csv`

## Project Structure

```text
medical_c_marl_project/
├── data/
│   └── KaggleV2-May-2016.csv
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── preprocessing.py
│   ├── environment.py
│   ├── agent.py
│   ├── training.py
│   ├── evaluation.py
│   ├── plotting.py
│   └── utils.py
├── notebooks/
│   ├── 01_preprocessing.ipynb
│   ├── 02_environment.ipynb
│   ├── 03_agents_training.ipynb
│   ├── 04_evaluation_results_corrected.ipynb
│   └── 05_report_figures_corrected.ipynb
├── checkpoints/
├── results/
│   └── final_figures/
├── README.md
├── requirements.txt
└── run_pipeline.py
```

## Installation

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If your system Python was built without `venv`, create an environment with an equivalent tool and then run the same `pip install -r requirements.txt` command.

## Run The Full Pipeline

From the repository root:

```bash
python run_pipeline.py
```

This runs, in order:

1. preprocessing
2. train/test split generation
3. Q-learning training for Easy, Medium, and Hard
4. checkpoint saving
5. held-out test evaluation
6. figure generation
7. final text report generation

## Run The Notebooks

The notebooks are thin wrappers over the reusable Python modules in `src/`.

Run them in this order:

1. `notebooks/01_preprocessing.ipynb`
2. `notebooks/02_environment.ipynb`
3. `notebooks/03_agents_training.ipynb`
4. `notebooks/04_evaluation_results_corrected.ipynb`
5. `notebooks/05_report_figures_corrected.ipynb`

They do not depend on hidden notebook state and they do not require manual `cwd` changes.

## Stage Definitions

### Easy

Observable state:

- `AgeGroup`
- `Gender`
- `SMS_received`

Actions:

- `0`: do nothing
- `1`: send reminder

### Medium

Observable state:

- `AgeGroup`
- `Gender`
- `SMS_received`
- `Scholarship`
- `Hypertension`
- `Diabetes`
- `Alcoholism`
- `Handicap`
- `MedicalRisk`

Actions:

- `0`: do nothing
- `1`: send reminder
- `2`: prioritize patient

### Hard

Observable state:

- `AgeGroup`
- `Gender`
- `SMS_received`
- `Scholarship`
- `Hypertension`
- `Diabetes`
- `Alcoholism`
- `Handicap`
- `MedicalRisk`
- `WaitingGroup`
- `AppointmentWeekday`
- `Neighbourhood`

Actions:

- `0`: do nothing
- `1`: send reminder
- `2`: prioritize patient
- `3`: propose rescheduling

## Preprocessing

The preprocessing pipeline:

- loads `data/KaggleV2-May-2016.csv`
- normalizes key column names
- converts date columns
- creates `WaitingDays`
- removes invalid ages and negative waiting times
- maps `No_show` to binary `0/1`
- encodes `Gender` and `Neighbourhood`
- creates:
  - `AgeGroup`
  - `WaitingGroup`
  - `MedicalRisk`
- saves full, train, and test CSVs for each stage

## No Leakage Rule

`No_show` is **not** part of the observable state.

It is used only:

- inside the reward function
- during evaluation
- when computing metrics

## Reward Function

The base reward is normalized across all stages:

- absent patient + intervention: `+1`
- absent patient + no intervention: `-1`
- attending patient + no intervention: `+1`
- attending patient + intervention: `-1`

Small targeted bonuses are allowed but kept intentionally small:

- `+0.25` for prioritizing an absent medically risky patient
- `+0.25` for rescheduling an absent patient with high waiting-time risk

These bonuses do not dominate the main attendance objective.

## Q-Learning Logic

Each stage uses tabular Q-learning with:

- discrete tuple states
- epsilon-greedy exploration
- update rule:

```text
Q(s, a) = Q(s, a) + alpha * (reward + gamma * max_a' Q(s', a') - Q(s, a))
```

Portable checkpoints are saved as plain dictionaries containing:

- stage name
- Q-table
- actions
- alpha
- gamma
- epsilon
- state columns
- seed

## Evaluation Metrics

Evaluation uses the **test split only**.

The main outputs are:

- `Total Reward`
- `Average Reward`
- `Good Decision Rate`
- `Useful Intervention Rate`
- `Unnecessary Intervention Rate`
- `Hard gain over Easy`
- `Hard gain over Medium`

The final report is evidence-based. It does not hardcode that Hard must win.

## Reproducibility

The project uses:

- fixed global seed: `42`
- deterministic train/test split
- deterministic episode shuffling
- portable checkpoints
- relative paths via `pathlib.Path`

## Limitations

- This is still tabular Q-learning, not a true multi-agent RL system.
- The environment is a simplified row-by-row simulation.
- The reward is based on observed outcomes in the historical dataset, so it is a pedagogical proxy rather than a causal treatment model.
- The `Hard` stage may or may not outperform the simpler stages after fair evaluation.
- The project should not be interpreted as a deployable healthcare optimization tool.

## Ethical Note

Appointment no-show behavior reflects structural and social factors that are not fully captured in this dataset. The results here should be treated as a classroom-style RL exercise, not as policy guidance for real healthcare systems.
