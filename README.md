# Incident Risk Model – Theme Park Safety Prediction

This project builds a professional-grade machine learning pipeline to predict the likelihood of a visitor experiencing an incident on a theme park ride, using historical incident data and engineered contextual features.

---

## Project Structure

`````
FINAL-ML-PROJECT/
├── data/
│ ├── raw/ # Original input data
│ │ └── incidents.csv
│ ├── processed/ # Cleaned and feature-enriched datasets
│ │ ├── incidents_clean.parquet
│ │ ├── incidents_enriched.parquet
│ │ ├── model_ready.parquet
│ │ └── model_ready_balanced.parquet
│ └── external/ # Metadata and supporting data sources
│ ├── rides_metadata_wikipedia.csv
│ ├── theme_park_locations.csv
│ └── weather_cache.parquet
│
├── notebooks/
│ ├── exploratory/ # EDA notebooks
│ │ ├── 01_EDA_incidents.ipynb
│ │ └── 02_EDA_enriched.ipynb
│ └── modeling/ # Modeling pipeline
│ ├── 01_prepare_model_data.ipynb
│ ├── 02_model_train.ipynb
│ ├── 03_model_baseline.ipynb
│ ├── 03_train_xgboost.ipynb
│ ├── 04_model_comparison.ipynb
│ ├── 05_model_error_analysis.ipynb
│ ├── 06_model_tuning.ipynb
│ ├── 07_export_model.ipynb
│ ├── 08_retrain_without_incident_count.ipynb
│ └── 09_retrain_balanced_model.ipynb
│
├── outputs/
│ ├── figures/ # Plots and visualizations
│ ├── models/ # Serialized models
│ │ ├── final_logistic_model.joblib
│ │ ├── final_logistic_model_no_incident_count.joblib
│ │ └── final_logistic_model_balanced.joblib
│ └── reports/ # Reports and summaries
│ └── nulls_enriched.csv
│
├── src/
│ ├── data_cleaning/ # Data cleaning logic
│ │ └── incidents.py
│ ├── enrichment/ # Feature engineering modules
│ │ ├── aggregate_features.py
│ │ ├── generate_negatives.py
│ │ ├── ride_metadata.py
│ │ ├── temporal_features.py
│ │ ├── visitor_profile.py
│ │ └── weather_enrichment.py
│ ├── io.py # I/O utilities for loading/saving data
│ ├── utils.py # Generic utility functions
│ └── test_cases.py # Optional test utilities
│
├── app.py # Streamlit web application
├── main.ipynb # Orchestrator notebook
├── requirements.txt # Python dependencies
└── README.md # Project overview (you are here)
`````



---

## Objective

Develop a predictive model that estimates the probability of a visitor experiencing an incident on a theme park ride, given:

- Visitor profile (age, gender, simulated medical condition, first visit)
- Ride characteristics (name, type, duration, historical incident rate)
- Temporal context (weekday, season, holiday)
- Simulated weather conditions

The final model is deployed via a Streamlit application that takes user inputs and outputs the estimated incident risk, with an explainable breakdown of the contributing factors.

---

## Pipeline Overview

1. **Data Cleaning** (`src/data_cleaning`)
   - Standardizes date formats and extracts structured values from `age_gender` and `description` columns.
   - Classifies incidents into six high-level types: `trauma`, `fall`, `motion`, `medical`, `pre_existing`, `other`.

2. **Feature Enrichment** (`src/enrichment`)
   - Adds features from:
     - Ride metadata (e.g., Wikipedia-derived)
     - Visitor profile synthesis
     - Temporal context (weekday, month, season, etc.)
     - Simulated weather conditions

3. **Negative Class Generation**
   - Creates synthetic "no-incident" cases to balance the dataset (`generate_negatives.py`).

4. **Modeling** (`notebooks/modeling`)
   - Baseline and advanced models (Logistic Regression, XGBoost)
   - Includes cross-validation, tuning, evaluation, and retraining for robustness

5. **Deployment**
   - Interactive web app built with `Streamlit` (`app.py`)
   - Accepts input parameters and returns risk probability with feature attribution

---

## Installation

Clone the repo and install dependencies:

```bash
git clone https://github.com/yourusername/incident-risk-model.git
cd incident-risk-model
pip install -r requirements.txt
````

##  Streamlit App

To run the app locally:

````
streamlit run app.py

````

##  Testing

To run the app locally:

````
streamlit run app.py

````

## Author
This project was developed as part of a final ML pipeline project by Germán Álvarez, applying industry best practices and structured workflows to a real-world predictive modeling use case.

##  License
MIT License – see LICENSE file for details.