# PCDS Risk Prediction – Logistic Regression App

This project builds and deploys a logistic regression model to predict the risk of **Peripheral Capillary Delay Syndrome (PCDS)** — a fictional medical condition designed for this project.

### 🔬 Dataset
Synthetic dataset with 500 samples, generated using medically inspired features:
- `capillary_refill_time` (seconds)
- `oxygen_saturation` (%)
- `heart_rate` (bpm)
- `age` (years)
- `has_pcds` (0 = No, 1 = Yes)

### 🧠 Model
- Logistic Regression
- Standard Scaler for preprocessing
- Pipeline training
- Model serialized with `joblib`

### 🖥️ Streamlit App
Interactive web interface for users to input clinical parameters and predict PCDS risk with probability score.

### 🚀 How to Run
#### 1. Generate dataset
```bash
python generate_dataset.py
