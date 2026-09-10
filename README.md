# DATA3800-gruppe-7

## Medlemmer

 - Erik Skålhegg (erska3276@oslomet.no)
 - Emirhan Güven (emguv8653@oslomet.no)
 - Abdulkadir Koc (abkoc1049@oslomet.no)

# DATA3800-gruppe-7

#### What is this project?
* A data science project where the goal is to review an open dataset in light of scientific literature

#### What dataset was chosen?
[Traffic Crashes - Crashes](https://data.cityofchicago.org/Transportation/Traffic-Crashes-Crashes/85ca-t3if/data_preview) from the City of Chicago's Data Portal

#### What scientific literature was chosen?
* TBD

#### Main goals of this project:
* What factors make a traffic accident more or less likely to result in personal injury

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
