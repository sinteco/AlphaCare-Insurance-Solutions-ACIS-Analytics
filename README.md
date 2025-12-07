# AlphaCare Insurance Solutions (ACIS) Analytics Project

## Overview
This project involves analyzing historical insurance claim data for AlphaCare Insurance Solutions (ACIS). The objective is to optimize marketing strategies and discover "low-risk" targets for potential premium reductions.

## Structure
- `data/`: Contains raw and processed data.
- `notebooks/`: Jupyter notebooks for EDA and experimentation.
- `src/`: Source code for the project (Python package).
- `scripts/`: Standalone scripts for executing tasks.
- `tests/`: Unit tests.

## Setup
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run analysis:
   ```bash
   python scripts/perform_eda.py
   ```

## Data Version Control (DVC)
This project uses DVC to manage large datasets.
1. Install DVC: `pip install dvc`
2. Pull data from remote:
   ```bash
   dvc pull
   ```
   *Note: Ensure you have access to the configured local remote storage.*

## Statistical Hypothesis Testing
This project includes a module for statistical validation of risk drivers.
Run the hypothesis tests:
```bash
python scripts/test_hypotheses.py
```
This script evaluates:
- Risk differences across Provinces and ZipCodes.
- Margin differences across ZipCodes.
- Risk differences between Genders.

## Predictive Modeling
This project builds machine learning models to predict Claim Severity.
Train and evaluate models:
```bash
python scripts/train_models.py
```
Models implemented:
- Linear Regression
- Random Forest (Best Performance)
- XGBoost
Includes SHAP analysis for feature importance.
