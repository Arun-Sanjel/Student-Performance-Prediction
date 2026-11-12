# Student Performance Prediction Using Machine Learning

## Project Overview

This project predicts whether a student will pass or fail using machine learning techniques and the UCI Student Performance Dataset.

The project was developed for CSE 4309 — Machine Learning.

## Dataset

- Source: UCI Machine Learning Repository
- Dataset: Student Performance (Portuguese)
- Total records: 649
- Target: Pass (G3 >= 10) or Fail (G3 < 10)

## Machine Learning Algorithms

Three classification algorithms are implemented:

1. Logistic Regression
2. Decision Tree
3. Random Forest

The models are evaluated using accuracy, balanced accuracy, precision, recall, F1-score, and cross-validation.

## Project Files

- `eda.py` — Exploratory data analysis and visualizations
- `train.py` — Model training and evaluation
- `feature_importance.py` — Feature importance analysis
- `app.py` — Streamlit prediction interface
- `data/` — Dataset used by the project
- `models/` — Saved trained model
- `outputs/` — Generated graphs and evaluation results
- `requirements.txt` — Python dependencies

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/Student-Performance-Prediction.git
cd Student-Performance-Prediction
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Run the Application

```bash
python -m streamlit run app.py
```

Open the local URL displayed in the terminal.

## Run the ML Scripts

```bash
python eda.py
python train.py
python feature_importance.py
```

## Results

The Random Forest model with previous grades (G1 and G2) was selected based on five-fold cross-validation Fail F1-score.

- Cross-validation Fail F1: 0.785
- Test accuracy: 85.4%
- Test Fail recall: 75%
- Test Fail F1: 0.612

The project is intended as an educational machine learning demonstration, not a validated system for real-world academic decisions.

## Team

CSE 4309 Machine Learning Group Project.

Team members and contributions will be added.
