# Heart Disease Prediction – Cleveland UCI Dataset

**University of Aveiro**

This project focuses on binary classification of heart disease presence using the Cleveland UCI Heart Disease dataset. We explore and compare the performance of various supervised learning models in predicting whether a patient has heart disease or not.

---

## Dataset

We use the Cleveland UCI Heart Disease dataset, which can be found on [Kaggle](https://www.kaggle.com/datasets/cherngs/heart-disease-cleveland-uci?fbclid=IwAR1Zs9rCsFKeC8-kOWTJF9sP5vB5ThC3pgcBjArKtHnt_uhLcXcv2petlS8).

The target variable is binary:
- **1**: Presence of heart disease
- **0**: Absence of heart disease

The dataset contains the following 13 features:

1. **age**: Age in years
2. **sex**: Sex (1 = male; 0 = female)
3. **cp**: Chest pain type:
   - 0: Typical angina
   - 1: Atypical angina
   - 2: Non-anginal pain
   - 3: Asymptomatic
4. **trestbps**: Resting blood pressure (in mm Hg on admission to the hospital)
5. **chol**: Serum cholesterol in mg/dl
6. **fbs**: Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
7. **restecg**: Resting electrocardiographic results:
   - 0: Normal
   - 1: Having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)
   - 2: Showing probable or definite left ventricular hypertrophy by Estes' criteria
8. **thalach**: Maximum heart rate achieved
9. **exang**: Exercise-induced angina (1 = yes; 0 = no)
10. **oldpeak**: ST depression induced by exercise relative to rest
11. **slope**: The slope of the peak exercise ST segment:
    - 0: Upsloping
    - 1: Flat
    - 2: Downsloping
12. **ca**: Number of major vessels (0-3) colored by fluoroscopy
13. **thal**: Thalassemia:
    - 0: Normal
    - 1: Fixed defect
    - 2: Reversible defect

---

## Models

We implemented and evaluated the following machine learning models:

- **Logistic Regression**
- **Support Vector Machine (SVM)**
- **Random Forest**
- **K-Nearest Neighbors (KNN)**
- **XGBoost**
- **Neural Network (MLPClassifier)**

---

This project was developed for the **Complements of Machine Learning** course at the **University of Aveiro**.
