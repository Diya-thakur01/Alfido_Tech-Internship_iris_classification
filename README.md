# Iris Classification Project

This project focuses on building a classification model using the famous Iris dataset. Two machine learning algorithms are used: **Decision Tree** and **Logistic Regression**.

---

# Dataset

The dataset contains 150 samples of iris flowers, with the following features:

- `sepal_length`
- `sepal_width`
- `petal_length`
- `petal_width`
- `species` (target variable)

# Project Objectives

- Load and preprocess the Iris dataset
- Train and evaluate two classification models:
  - Decision Tree Classifier
  - Logistic Regression
- Visualize results using a **confusion matrix**
- Save output plots for analysis

# Requirements

Install the necessary Python packages with:

```bash
pip install -r requirements.txt
requirements.txt includes:
pandas

numpy

matplotlib

seaborn

scikit-learn

 How to Run the Project
Clone the repository or download the project folder:

bash

git clone https://github.com/Diya-thakur01/Alfido_Tech-Internship_iris_classification.git

Navigate into the folder:
bash
cd iris_classification

Run the script:
bash
python iris_classification.py
Output metrics and confusion matrix plots will be displayed and saved in the outputs/ folder.

 Results
The models are evaluated using:

Accuracy Score

Classification Report (Precision, Recall, F1-Score)

Confusion Matrix

 Folder Structure

iris_classification/
├── iris_classification.py
├── requirements.txt
├── README.md
├── outputs/
│   ├── decision_tree_confusion_matrix.png
│   └── logistic_regression_confusion_matrix.png
