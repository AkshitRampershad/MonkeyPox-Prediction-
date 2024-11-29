## Monkeypox Prediction Model for Rapid Disease Detection
## Project Overview
Monkeypox, a viral disease, has emerged as a significant health concern, often requiring timely diagnosis to prevent severe outcomes. This project leverages machine learning to develop a predictive model for rapid and accurate detection of monkeypox. Using a synthetic dataset based on studies published by the British Medical Association (BMJ), the project aims to create a reliable classification model for identifying positive and negative cases.

## Project Highlights
1. Objective: Build a machine learning model to predict Monkeypox infections with high precision, minimizing false positives.
2. Dataset: Synthetic dataset of 25,000 global patients with 10 features and a binary target variable (MonkeyPox).
3. Outcome: Logistic Regression achieved the highest precision score of 0.684, outperforming models like K-NN, Decision Trees, and Neural Networks.

## Dataset Description
The dataset contains 25,000 patient records with the following features:
Features:
Rectal Pain, Sore Throat, Penile Oedema, Oral Lesions, Solitary Lesion, Swollen Tonsils, HIV Infection, Sexually Transmitted Infection
Target Variable: MonkeyPox (Binary: 1 = Positive, 0 = Negative)

## Tools and Technologies
Languages: Python
Libraries:
Data Manipulation: Pandas, NumPy
Model Training: Scikit-learn
Visualization: Matplotlib
Machine Learning Models:
Logistic Regression (Best Model)
K-Nearest Neighbors (with and without k-fold)
Decision Tree
AdaBoost
Gradient Descent
Neural Networks
Data Preprocessing:
OrdinalEncoder, LabelEncoder
StandardScaler
GridSearchCV for Hyperparameter Tuning

## Project Workflow
1. Import Libraries: Load essential libraries for data processing and modeling.
2. Read Dataset: Explore and preprocess the dataset for training and testing.
3. Clean Data: Handle missing values and encode categorical features.
4. Split Dataset: Divide data into training and validation sets.
5. Train Models: Train multiple models and tune hyperparameters.
6. Evaluate Models: Use metrics like precision, recall, and F1 score to assess model performance.
7. Select Best Model: Optimize and deploy the best-performing model (Logistic Regression).

## Model Performance
Evaluation Metric: Precision (focus on minimizing false positives).
Best Model: Logistic Regression
Precision: 0.684
Key Advantage: Effective in reducing false positives, minimizing unnecessary treatments and associated costs.

## Business Implications
1. Early Disease Detection: Enables healthcare professionals to act swiftly, ensuring timely treatment.
2. Cost Efficiency: Reduces unnecessary tests and medical expenses caused by false positives.
3. Public Health Preparedness: Improves resource allocation and planning for outbreaks.
4. Enhanced Outcomes: Contributes to better management and patient care.

## Future Scope
Incorporate real-world datasets to validate the model further.
Explore ensemble techniques to improve precision and recall.
Optimize for imbalanced datasets using advanced sampling techniques.
Deploy the model as a web application or API for practical healthcare use.
