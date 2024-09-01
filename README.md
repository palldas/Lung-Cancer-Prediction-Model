# Lung Cancer Prediction Models

## Overview

This project aims to predict lung cancer using both regression and classification models, specifically Linear Regression and K-Nearest Neighbors (KNN). The dataset used for this analysis is the "Lung Cancer Survey" dataset, which contains various features related to lung cancer risk factors and symptoms. The goal is to analyze these features to understand their impact on lung cancer prediction and to develop reliable models for early detection.

## Dataset

The "Lung Cancer Survey" dataset includes a mix of categorical and numerical variables, offering a comprehensive basis for both regression and classification tasks. The target variable for classification is whether the individual has lung cancer (YES or NO). The dataset has 309 entries, each representing a respondent with various recorded attributes related to lung cancer risk factors.

### Features

- **Gender**: M (male), F (female)
- **Age**: Age of the patient
- **Smoking**: YES=2, NO=1
- **Yellow Fingers**: YES=2, NO=1
- **Anxiety**: YES=2, NO=1
- **Peer Pressure**: YES=2, NO=1
- **Chronic Disease**: YES=2, NO=1
- **Fatigue**: YES=2, NO=1
- **Allergy**: YES=2, NO=1
- **Wheezing**: YES=2, NO=1
- **Alcohol Consuming**: YES=2, NO=1
- **Coughing**: YES=2, NO=1
- **Shortness of Breath**: YES=2, NO=1
- **Swallowing Difficulty**: YES=2, NO=1
- **Chest Pain**: YES=2, NO=1
- **Lung Cancer**: YES, NO

## Requirements

To run this project, you need to have the following Python libraries installed:

- pandas
- scikit-learn
- matplotlib

You can install the required libraries using pip:

```bash
pip install pandas scikit-learn matplotlib
```

## Project Structure

- **lung_cancer_prediction.py**: The main Python script containing all the code for data preprocessing, model training, evaluation, and visualization.
- **survey lung cancer.csv**: The dataset file used for analysis, sourced from Kaggle.
- **README.md**: This file, providing an overview of the project and instructions for setup and usage.
- **Lung Cancer Prediction Report.pdf**: A more detailed report of the prediction models.

## Running the Code

### 1. Clone the Repository

Ensure you clone the repository or download the project files to your local machine. Make sure all files, including `survey lung cancer.csv`, are in the same directory.

### 2. Run the Script

Execute the `lung_cancer_prediction.py` script to perform the entire analysis. This script includes:

- **Data Preprocessing**: Encoding categorical variables and scaling numerical features using scikit-learn's `ColumnTransformer`.
- **Linear Regression**: Training and evaluation using RMSE and MAE, with a loadings plot generated to visualize feature coefficients.
- **K-Nearest Neighbors Regression**: Hyperparameter tuning using `GridSearchCV` to find the optimal number of neighbors (k), followed by model training and evaluation.
- **K-Nearest Neighbors Classification**: Similar to regression, with additional evaluation metrics such as accuracy, precision, recall, F1-score, sensitivity, and specificity.

### 3. Analyze the Results

After running the script, review the printed output and generated plots to understand the performance of each model. The evaluation metrics will help you compare the models and draw conclusions about their effectiveness in predicting lung cancer.

## Results

This project provides a comprehensive comparison of Linear Regression and K-Nearest Neighbors models for both regression and classification tasks. The results are evaluated using various metrics such as RMSE, MAE, accuracy, precision, recall, F1-score, sensitivity, and specificity.

### Key Findings

- **Linear Regression**: Demonstrates better consistency in predictions with a lower RMSE, making it suitable for applications where stable and reliable predictions are necessary.
- **K-Nearest Neighbors Regression**: Offers more precise predictions with a lower MAE, indicating its effectiveness in closely predicting outcomes.
- **K-Nearest Neighbors Classification**: Achieves high accuracy and recall, making it a strong model for identifying positive lung cancer cases. However, the specificity indicates a higher rate of false positives, suggesting room for improvement in distinguishing negative cases.

## Conclusion

This project demonstrates the application of regression and classification models to predict lung cancer using a survey dataset. While both models show strong performance, their strengths vary: Linear Regression is more consistent, while KNN models are more precise. The classification model's high recall makes it effective for early detection, although its specificity needs improvement to reduce false positives. These insights can help in refining predictive tools for early lung cancer diagnosis and intervention.
