EPAA
==============================

TAED2 Project: Meaningful insight extraction from git commit texts.

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

## Requirements

## Initial data collection
First of all we need to download the raw dataset from [The Technical Debt Dataset](https://github.com/clowee/The-Technical-Debt-Dataset/releases/tag/2.0) in `.db` format.

Then we can generate the `.csv` for all the tables in the database with:

```bash
python3 src/data/make_dataset.py path/to/td_V2.db path/to/output_folder
```

After this, in `output_folder` we will have the following `.csv` files:

    output_folder
    ├── GIT_COMMITS_CHANGES.csv
    ├── GIT_COMMITS.csv
    ├── JIRA_ISSUES.csv
    ├── PROJECTS.csv
    ├── REFACTORING_MINER.csv
    ├── SONAR_ANALYSIS.csv
    ├── SONAR_ISSUES.csv
    ├── SONAR_MEASURES.csv
    ├── SONAR_RULES.csv
    └── SZZ_FAULT_INDUCING_COMMITS.csv

## Data Preparation
Now that we have our raw data in a suitable format, we can preparate it. First of all we take the `.csv` and generate the cleaned dataset from `GIT_COMMITS.csv`, `SONAR_ANALYSIS.csv` and `SONAR_MEASURES.csv`:

```bash
python3 src/data/preparate_data.py path/to/input_folder path/to/output_folder
```

Where `input_folder` is the root folder where all the `.csv` are stored and `output_folder` is the root folder where the final dataset `predictionDB.csv` will be stored.

Once we have `predictionDB.csv` we need to generate the sentence embeddings, one for each commit message. This can be done with:

```bash
python3 src/data/commit_to_emb.py path/to/input_folder path/to/output_folder
```

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
