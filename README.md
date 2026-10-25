# DATA3800-gruppe-7

## Members

 - Erik Skålhegg (erska3276@oslomet.no)
 - Emirhan Güven (emguv8653@oslomet.no)
 - Abdulkadir Koc (abkoc1049@oslomet.no)
 - Haakon Meyer Tischbein
 - Sondre Gullichsen Ulsaker

## About the project

#### What is this project?
* A data science project where the goal is to review an open dataset in light of scientific literature

#### What dataset was chosen?
[Traffic Crashes - Crashes](https://data.cityofchicago.org/Transportation/Traffic-Crashes-Crashes/85ca-t3if/data_preview) from the City of Chicago's Data Portal

#### What scientific literature was chosen?
* Analysis of driver injury severity levels at multiple locations using ordered probit models (https://doi.org/10.1016/j.jsr.2003.05.009)
* The statistical analysis of highway crash-injury severities: A review and assessment of methodological alternatives (https://doi.org/10.1016/j.aap.2011.03.025)

#### Main goals of this project:
* What factors make a traffic accident more or less likely to result in personal injury

## How to run the pipeline

Install uv first: https://docs.astral.sh/uv/getting-started/installation/. Then run these commands from the project root, in this order:

```bash
uv sync
uv run python data_fetch.py
uv run python run_cleaning.py
uv run python -m data3800_gruppe_7.modeling
uv run jupyter lab
```

1. `uv sync` installs the right Python version and the packages in `uv.lock`.
2. `data_fetch.py` downloads the raw data to `data/raw_dataset.csv`.
3. `run_cleaning.py` saves the cleaned data in `data/cleaned_crashes.csv`. The EDA notebook reads this file, so it has to exist before the notebook runs.
4. The modeling module trains four models and saves them in `output/models/`, together with their predictions on the test set. The evaluation notebook reads these files.
5. In Jupyter, run the notebooks in order: `02-eda.ipynb`, `03-eg-modeling.ipynb` and `04_evaluation.ipynb`.

The `data/` and `output/models/` folders are not on GitHub, so a new clone needs steps 2 to 4 before the notebooks will run.

## Pipeline

| Step | Files | Output |
| --- | --- | --- |
| Download | `data_fetch.py` | `data/raw_dataset.csv` |
| Cleaning | `src/data3800_gruppe_7/cleaning.py`, `run_cleaning.py` | `data/cleaned_crashes.csv`, `output/cleaning_summary.txt` |
| Exploratory analysis | `notebooks/02-eda.ipynb` | Tables and figures in the notebook |
| Modeling | `src/data3800_gruppe_7/modeling.py`, `notebooks/03-eg-modeling.ipynb` | `output/models/*.joblib`, `output/models/predictions.csv` |
| Evaluation | `notebooks/04_evaluation.ipynb` | Comparison of the models |

## Who did what

| Part | Responsible |
| --- | --- |
| Data cleaning | Abdulkadir Koc |
| Exploratory analysis | Erik Skålhegg |
| Modeling | Emirhan Güven |
| Evaluation | Haakon Meyer Tischbein |
| Integration and documentation | Sondre Gullichsen Ulsaker, with the whole group |

## Results

### Exploratory analysis

After cleaning, the data has 108,903 police-reported crashes from 2025, and 16.4% of them involved at least one injury. The exploratory analysis found these patterns:

- The injury share is 15.9% in daylight and 20.7% in darkness on lighted roads.
- By hour, the share is lowest in the morning (13.8% at 10:00) and between 18.5 and 20.8% from 21:00 to 04:00.
- A higher posted speed limit goes with more injuries: 8.8% at 20 mph or less and 22.1% at 50 mph or more.
- Wet roads have an injury share of 20.5% and icy roads 10.3%.
- When the conditions are recorded as UNKNOWN, only 5.5 to 7.8% of the crashes involve injuries.

These are associations in police-reported crashes and do not show that the conditions cause the injuries.

### Models

We train four models: a baseline that always predicts no injury, logistic regression, a decision tree and a random forest. The evaluation notebook compares them.

---

## Internal

### Prerequisites
Install uv: https://docs.astral.sh/uv/getting-started/installation/

### One-time setup
```bash
git clone <repo-url>
cd DATA3800-gruppe-7
uv sync                    # builds .venv from the lockfile, installs the correct Python if missing
uv run pre-commit install  # activates notebook output stripping on commit
```

### Every session
```bash
git pull
uv sync             # picks up dependencies others have added
uv run jupyter lab  # or select the .venv kernel in VS Code
```

### Adding a dependency
```bash
uv add <package>
```
Commit `pyproject.toml` and `uv.lock` together in the same commit.
If `uv.lock` gets a merge conflict, don't edit it by hand: accept either side, run `uv lock`, and commit the result.

### Notebooks
* Name notebooks `NN-initials-topic.ipynb`, e.g. `01-hmt-eda.ipynb`. One notebook, one owner.
* Reusable code (loading, cleaning, features) goes in `src/`, not in notebooks.
* Start every notebook with `%load_ext autoreload` and `%autoreload 2` so edits in `src/` are picked up without restarting the kernel.
* Outputs are stripped automatically on commit. If a commit is aborted because the hook modified files, `git add` the notebook and commit again.
