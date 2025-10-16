# Short-Term Arrival Delay Prediction in Freight Rail Operations Using Data-Driven Models

![Python](https://img.shields.io/badge/Python-3.11-blue) ![License](https://img.shields.io/badge/License-MIT-green)

---

## Project Overview

Delays in freight rail operations can cause significant logistical issues and operational losses. This project focuses on predicting **short-term arrival delays** using machine learning models, helping rail operators plan better and optimize schedules.  

Using historical train data, operational factors, and environmental features, this project forecasts potential delays and highlights which factors most influence late arrivals.

---

## Motivation

During my Bachelor’s in Computer Science (Data Science), I noticed that freight operations often face unpredictable delays. I wanted to explore how **data-driven models** can help predict these delays and improve efficiency. This project reflects my interest in applying machine learning to real-world operational challenges.

---

## Key Features

- Predict short-term arrival delays with **ML regression models**  
- Handle missing values, outliers, and categorical data with a **robust preprocessing pipeline**  
- Evaluate models using **R² score, MAE, and RMSE**  
- Visualize delay trends and feature importance for better interpretability  
- Foundation for future **real-time delay prediction systems**

---

## Dataset

- **Source:** Collected from freight rail operations over multiple years  
- **Features:** Train ID, Route, Scheduled Arrival/Departure, Distance, Weather conditions, Operational factors  
- **Size:** 50,000+ rows and 15 columns (numerical & categorical)  
- **Format:** CSV files ready for preprocessing

---

## Tools & Technologies

- Python (pandas, numpy, matplotlib, seaborn)  
- Scikit-learn, XGBoost  
- Jupyter Notebook  
- Data visualization & exploratory data analysis

---

## Models Used

- **Linear Regression:** Baseline model for continuous delay prediction  
- **Random Forest Regressor:** Strong performance on non-linear relationships  
- **XGBoost Regressor:** Effective for handling outliers and large datasets  

---

## Results

| Model                  | R² Score | MAE (min) | RMSE (min) |
|------------------------|----------|-----------|------------|
| Linear Regression       | 0.72     | 7.5       | 10.2       |
| Random Forest Regressor | 0.87     | 4.3       | 6.8        |
| XGBoost Regressor       | 0.88     | 4.1       | 6.5        |

> Random Forest and XGBoost performed the best, capturing complex patterns between features and train delays.  

---

## Visualizations / Screenshots
<img width="1366" height="768" alt="Picture1" src="https://github.com/user-attachments/assets/874dae30-238c-4ed4-a5fa-d503cf966f84" />

*(Add screenshots of plots here, e.g., Actual vs Predicted delays, Feature Importance charts, etc.)*

---

## Future Work

- Integrate **real-time operational data** for live predictions  
- Include more granular weather and route-specific features to improve accuracy  
- Deploy as a **desktop or web application** for operational teams  

---

## Author

**Abdul Furkhan** – I led the project and collaborated with peers as part of my Bachelor’s in CSE (Data Science).  
> This project demonstrates my skills in **data preprocessing, machine learning, and predictive analytics**, and shows my interest in solving real-world logistical challenges with data-driven solutions.

---

## License

This project is licensed under the **MIT License** – see the LICENSE file for details.
