# 🏠 House Price Predictor

A **Machine Learning-powered web application** built with **Flask**, capable of predicting house prices based on key property features like area, bedrooms, bathrooms, location, quality, and garage size.

---

## 🚀 Project Overview

This project predicts house prices using trained ML models such as **Random Forest** and **XGBoost**.
It includes:

* Clean and responsive **Flask web app**
* Interactive **HTML interface** for input
* Pretrained ML model for instant predictions
* Fully customizable dataset and model tuning

---

## 🧠 Features

* 🔍 **Accurate Predictions** — Uses advanced ML models trained on housing data
* 🌐 **Flask Web Interface** — Lightweight and interactive UI
* 🎨 **Responsive Design** — Clean layout with transparent house background
* ⚙️ **Easily Extendable** — Add or modify features anytime
* 💾 **Pretrained Model** — No need to retrain every time

---

## 🧩 Tech Stack

| Category         | Technology                      |
| ---------------- | ------------------------------- |
| Backend          | Python, Flask                   |
| Machine Learning | Scikit-learn, XGBoost, CatBoost |
| Frontend         | HTML, CSS (Inline Styling)      |
| Data             | Pandas, NumPy                   |
| Deployment       | GitHub / Localhost              |

---

## 🧪 Sample Input Fields

| Feature         | Example Input |
| --------------- | ------------- |
| Area (sqft)     | 1500          |
| Bedrooms        | 3             |
| Bathrooms       | 2             |
| Location        | Downtown      |
| Quality         | 8             |
| Garage Size     | 300           |
| Garage Capacity | 2             |

---

## 📊 Model Performance

| Model         | RMSE   | R² Score |
| ------------- | ------ | -------- |
| Random Forest | 1.18e5 | 0.961    |
| XGBoost       | 1.16e5 | 0.963    |

✅ The best-performing model (XGBoost) is automatically saved as `best_model.joblib`.

---

## 🖼️ Preview

Here’s a sneak peek of the app UI:

```
🏡 [ House Price Predictor ]
--------------------------------
Enter:
 Area → 1500
 Bedrooms → 3
 Bathrooms → 2
 Location → Downtown
 Quality → 8
 Garage Size → 300
 Garage Capacity → 2
--------------------------------
Predicted Price: ₹ 4,250,000
```

---


