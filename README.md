# AI-Driven Hybrid Intrusion Detection System

This repository contains the code for my master’s project on a hybrid intrusion detection system (IDS) that combines AI-based anomaly detection with a modular monitoring pipeline for host-level cyber defense.

## Project Overview

Traditional intrusion detection approaches often rely on signature-based detection, which can struggle with previously unseen or evolving attack behavior. This project explores a hybrid IDS design centered on anomaly detection using an autoencoder trained on cleaned CICIDS2017-based features.

The system is organized as a modular workflow with components for synthetic activity generation, live anomaly scoring, evaluation, and interactive visualization.

## Architecture

The project follows a feeder–watcher–dashboard design:

- **Feeder (`tools/feeder.py`)**
  Generates synthetic feature activity with normal behavior, bursts, ramps, noisy windows, drift, and attack windows.

- **Watcher (`tools/live_score.py`)**
  Loads the trained autoencoder model, reads live feature rows, computes reconstruction error, applies threshold-based anomaly detection, and writes live scores.

- **Dashboard (`tools/dashboard.py`)**
  Displays anomaly scores, alerts, threshold controls, and live monitoring output using Streamlit.

- **Evaluation (`tools/eval_and_plot.py`, `tools/metrics.py`)**
  Produces metrics and visual outputs such as time-series plots, ROC curve, and precision-recall curve.

## Main Features

- AI-driven anomaly detection using an autoencoder
- threshold-based real-time alerting
- synthetic attack and drift simulation for demos
- Streamlit dashboard for live monitoring
- archived run snapshots and post-run evaluation
- modular structure for future hybrid IDS or deception-based extensions

## Repository Structure

```text
AI driven hybrid IDS/
├── README.md
├── requirements.txt
├── .gitignore
├── models/
│   └── autoencoder_cicids17_cleaned.h5
├── tools/
│   ├── dashboard.py
│   ├── eval_and_plot.py
│   ├── feeder.py
│   ├── live_score.py
│   └── metrics.py
├── inputs/
├── outputs/
├── control/
├── artifacts/
└── screenshots/

## Technologies Used
 -Python
 -TensorFlow / Keras
 -NumPy
 -Pandas
 -Matplotlib
 -Streamlit
 -Altair

## How to Run

### 1. Clone the repository
git clone https://github.com/shanmukhchintalapati/ai-driven-hybrid-ids.git
cd ai-driven-hybrid-ids

### 2. Install dependencies
pip install -r requirements.txt

### 3. Start the feeder
python tools/feeder.py --write-header

### 4. Start the watcher
python tools/live_score.py --model models/autoencoder_cicids17_cleaned.h5 --thr 0.02

### 5. Start the dashboard
streamlit run tools/dashboard.py

## Outputs
During execution, the project writes runtime and evaluation artifacts such as:
 -inputs/live_features.csv
 -outputs/scores_live.csv
 -archived run snapshots in artifacts/
 -plots and evaluation outputs generated after a run

## Notes on Evaluation
The included evaluation scripts are intended for project demonstration and post-run inspection. Some metrics are computed using heuristic logic derived from reconstruction error and thresholded alerts. They are useful for explaining system behavior, but they should not be interpreted as a direct replacement for full benchmark evaluation against labeled ground-truth datasets.

## Research Context
This project was developed as part of my master’s work in Information Assurance. It focuses on anomaly-based intrusion detection and presents a modular implementation that can be extended toward more advanced hybrid IDS and deception-oriented cyber defense workflows.

## Author
Shanmukh Chintalapati
Master’s project in Information Assurance