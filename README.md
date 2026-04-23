# Motorsport Analytics - Tire Degradation Prediction

This mini-project predicts lap-time degradation over a race stint, inspired by Formula 1 strategy analytics.

## What it does

- Ingests race laps using FastF1
- Filters noisy laps (pit in/out, SC/VSC)
- Engineers tire degradation features
- Trains a regression model (Random Forest or Gradient Boosting)
- Evaluates with MSE and R2
- Visualizes actual vs predicted lap times in Streamlit

## Core features used

- `Compound` (Soft/Medium/Hard): baseline tire behavior
- `TyreLife`: tire age in laps
- `TrackTemp`: thermal conditions affecting degradation
- `RollingLapTime`: smoothing feature for traffic anomalies

## Local run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Docker run

```bash
docker build -t motorsport-analytics .
docker run --rm -p 8501:8501 motorsport-analytics
```

Open: http://localhost:8501

## Notes

- FastF1 cache is enabled by default (`./cache`) and configurable in the Streamlit sidebar.
- Driver should be a 3-letter code (e.g., `VER`, `NOR`, `LEC`).
