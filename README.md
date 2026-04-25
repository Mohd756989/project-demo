# 📊 XGBoost Regressor with Evaluation Metrics

## 📌 Overview
This project demonstrates how to build a regression model using **XGBoost Regressor** and evaluate its performance using multiple standard metrics such as RMSE, MAE, and R².

It provides a simple and scalable pipeline for training and evaluating regression models on real-world datasets.

---

## 🚀 Features
- Train a regression model using XGBoost
- Evaluate performance using:
  - RMSE (Root Mean Squared Error)
  - MAE (Mean Absolute Error)
  - R² Score (Coefficient of Determination)
- Modular and clean code structure
- Easy to extend for real-world applications

---

## 🧠 Technologies Used
- Python 🐍
- XGBoost
- Scikit-learn
- NumPy
- Pandas

---

## 📂 Project Structure
project/
│── data/
│ └── dataset.csv
│── model/
│ └── xgboost_model.pkl
│── 
│ ├── main.py
│ ├── streamlit_app.py
│ 
│
└── README.md


---

## ⚙️ Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/xgboost-regressor.git
cd xgboost-regressor
#run the main file for model generation
python main.py 

#run the strealit app 
python streamlit_app.py