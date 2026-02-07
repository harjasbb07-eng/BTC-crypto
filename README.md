# BTC-Crypto — AI Meta-Labelled Trading Strategy

An AI-powered crypto trading system designed to analyze Bitcoin hourly price data and decide which trades are worth taking — and which ones to avoid.

Built during a 48-hour hackathon after our team backend failed, this became the replacement end-to-end strategy — covering data processing, signal generation, ML filtering, and backtesting.

---

## 🚀 Project Highlights

### 🌿 Meta-Labelling AI Layer  
Filters bad trades instead of generating signals blindly.

### 📊 Walk-Forward Validation  
Trains on past years → tests on the next year (realistic simulation).

### ✔️ True Holdout Testing  
2022 data kept completely unseen until final evaluation.

### 📈 Equity Curve Visualization  
Strategy vs Buy-and-Hold comparison.

### 📉 Drawdown Analysis  
Measures worst loss periods & risk.

### 🧠 Feature Importance  
Shows which market factors influenced AI decisions most.

### ⚡ GPU-Accelerated XGBoost  
Fast training using CUDA.

---

## 🧠 Strategy Architecture

### 1️⃣ Primary Signal Layer

Trend-following signals generated using:

EMA Crossovers (20 vs 50)

---

### 2️⃣ Feature Engineering

Market context features:

RSI (Momentum)  
ATR Ratio (Volatility)  
Volume Ratio  
30-Day Volatility  
Hour of Day  
Day of Week

---

### 3️⃣ Meta-Labelling Model

Model: XGBoost Classifier

Purpose:

Predict whether a trade signal will be profitable after 24 hours.

Only high-probability trades are executed.

---

### 4️⃣ Backtesting Engine

Supports:

Long + Short positions  
Confidence-based exits  
Signal flip exits  
Trading fees included  
Equity tracking

---

## 📊 Validation Methodology

### Walk-Forward Testing

Train Years | Test Year  
2018 | 2019  
2018–2019 | 2020  
2018–2020 | 2021  

Simulates real deployment.

---

### True Out-of-Sample Test

Train → 2018–2021  
Test → 2022 (Unseen)

Ensures no data leakage.

---

## 🖥️ Live Demo

### 🔗 Run the project yourself:
https://colab.research.google.com/drive/1ps2r2VAzUHPUNpB6MDSU3NSQlOg8Xl15?usp=sharing

Inside the notebook you can:

Click Run All  
Train the ML model  
Execute backtests  
View performance charts  
Compare vs Buy-and-Hold  

No installation required — runs fully in browser.

###Streamlit version
https://btc-crypto-ezz.streamlit.app/

You can run the project here too but be warned it takes plenty time to run.

---

## 📂 Repository Links

### 🔗 Original Team Repo:
https://github.com/chandraxshu/OOC

### 🔗 My Implementation (Cleaned):
https://github.com/harjasbb07-eng/BTC-crypto

---

## 🛠️ Tech Stack

Python  
Pandas / NumPy  
XGBoost (GPU)  
Scikit-learn  
Matplotlib  
Google Colab

---

## 📈 Outputs Generated

The notebook produces:

Walk-forward yearly returns  
2022 holdout performance  
Equity curves  
Sharpe ratios  
Max drawdown  
Feature importance charts

---

## 🎯 Project Goal

To demonstrate how AI can enhance trading systems by:

Filtering low-quality trades  
Improving risk management  
Increasing strategy robustness  
Providing realistic backtesting

---

## ⚠️ Disclaimer

This project is for educational & research purposes only.  
It does not constitute financial advice or a production trading system.
