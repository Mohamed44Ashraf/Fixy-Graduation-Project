# Smart Service Marketplace & AI Analytics System

## 📌 Project Overview
This project is an intelligent service marketplace platform designed to connect customers with qualified technicians in a structured and efficient way. It integrates machine learning models for rating prediction and anomaly detection to improve trust, fairness, and service quality.

The system addresses real-world challenges such as:
- Difficulty in finding reliable technicians
- Unfair or biased rating systems
- Lack of transparency in service execution
- Fraudulent or inconsistent feedback patterns

---

## 🎯 Project Objectives
- Build a smart platform for service matching
- Improve fairness in rating systems
- Detect anomalies and fraudulent behavior
- Enhance decision-making using machine learning
- Simulate realistic service marketplace data

---

## 📊 Datasets

### 1. Rating Dataset
A synthetic dataset generated from scratch to simulate real-world service interactions between customers and technicians.

**Key Features:**
- Service types (plumbing, electrical, etc.)
- Response time and job duration
- Customer and technician ratings
- Satisfaction scores
- Final aggregated rating

---

### 2. Anomaly Detection Dataset
Designed to detect abnormal or fraudulent behavior in rating patterns and interactions.

**Key Features:**
- Customer and technician profiles
- Rating deviation
- Predicted rating
- Feedback consistency
- Anomaly label (IsAnomalous)

---

## ⚙️ Data Generation Approach
The datasets were fully generated using rule-based statistical methods to ensure realism and consistency:

- Gaussian and truncated distributions for realistic values
- Probabilistic modeling for user and technician behavior
- Hidden latent behavior simulation for anomaly creation
- Controlled noise injection to mimic real-world imperfections
- Balanced class distribution for model training

---

## 🤖 Machine Learning Models Used

| Model | Description |
|------|-------------|
| ANN | Neural network for complex pattern learning |
| LightGBM | Fast gradient boosting model |
| CatBoost | Handles categorical features efficiently |
| XGBoost | High-performance boosting algorithm |
| Voting Classifier | Ensemble of multiple models |

---

## 🎯 Model Selection Strategy
We selected models based on **Recall** to minimize missed anomalies and ensure high detection sensitivity.

> **Note:** Recall was prioritized to reduce missed anomaly cases.

---

## 🧠 Key Features of the System
- Smart technician matching system
- Dual-sided rating evaluation (customer & technician)
- Fraud and anomaly detection module
- Fairness-aware rating aggregation
- Secure service workflow simulation

---

## ⚠️ Challenges Addressed
- Job availability imbalance for technicians
- Difficulty in finding reliable workers
- Bias and unfairness in rating systems
- Missing or incomplete feedback data
- Fraudulent or inconsistent evaluations

---

## 🚀 Future Work
- Expand system to multiple countries
- Deploy as a full-scale production platform
- Improve real-time anomaly detection
- Enhance recommendation engine accuracy
- Integrate mobile application support

---

## 👨‍💻 Technologies Used
- Python
- NumPy / Pandas
- Scipy
- Scikit-learn
- Machine Learning Models (ANN, Boosting, Ensemble)

---

## 📌 Conclusion
This project demonstrates a complete end-to-end intelligent service marketplace system combining data generation, machine learning, and anomaly detection to simulate real-world challenges and provide scalable solutions.