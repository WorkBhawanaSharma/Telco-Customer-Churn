# Churn Prediction Project

This project is aimed at predicting customer churn for a telecom company using various features such as customer demographics, account information, and service usage patterns. The goal is to build a model that accurately predicts whether a customer will churn (leave) based on historical data.

## Project Overview

The project involves loading and preprocessing a dataset, performing exploratory data analysis (EDA), and building a machine learning model to predict customer churn. A Random Forest Classifier is used to make the predictions, and model performance is evaluated using accuracy, precision, recall, F1-score, and confusion matrix.

## Dataset

The dataset contains the following key columns:
- **customerID**: Unique identifier for each customer.
- **gender**: Gender of the customer.
- **SeniorCitizen**: Whether the customer is a senior citizen (1 or 0).
- **Partner**: Whether the customer has a partner.
- **Dependents**: Whether the customer has dependents.
- **tenure**: Number of months the customer has been with the company.
- **PhoneService**: Whether the customer has phone service.
- **MultipleLines**: Whether the customer has multiple phone lines.
- **InternetService**: Type of internet service the customer subscribes to.
- **OnlineSecurity**: Whether the customer has online security.
- **TechSupport**: Whether the customer has tech support.
- **Contract**: Type of contract the customer has (month-to-month, one year, two years).
- **PaymentMethod**: Payment method used by the customer.
- **MonthlyCharges**: Monthly charges for the customer.
- **TotalCharges**: Total charges incurred by the customer.
- **Churn**: Whether the customer has churned (target variable).

## Steps Involved

1. **Loading the Dataset**: The dataset is loaded using `pandas` and initial data checks are performed.
2. **Data Exploration**: Exploratory Data Analysis (EDA) is conducted to understand trends and relationships within the data.
3. **Data Cleaning & Preprocessing**: Missing values are handled, numerical features are normalized, and categorical variables are encoded.
4. **Model Development**: A Random Forest Classifier is trained using the preprocessed data to predict customer churn.
5. **Model Evaluation**: The model's performance is evaluated using various metrics like accuracy, precision, recall, F1-score, and a confusion matrix.
6. **Insights & Recommendations**: Based on the feature importances, insights and recommendations are provided to improve customer retention.

## Insights & Recommendations

Based on feature importances, the following insights and recommendations are made:

1. **TotalCharges (Importance: 0.189992)**
   - **Insight**: The total amount a customer has spent with the company is the most significant factor in predicting churn. Customers with lower total charges are more likely to churn.
   - **Recommendation**: Focus on retaining long-term customers by offering loyalty rewards, discounts, or special offers to increase their total spending. Target low-spending customers with promotions to increase their lifetime value.

2. **MonthlyCharges (Importance: 0.177867)**
   - **Insight**: Monthly charges are a strong predictor of churn. Higher monthly charges could be a deterrent for customers if they perceive a lack of value.
   - **Recommendation**: Evaluate pricing structure to ensure it aligns with customer expectations. Offering affordable plans or value-added services could reduce churn, especially in competitive markets.

3. **Tenure (Importance: 0.157428)**
   - **Insight**: Customers with shorter tenures are more likely to churn. New customers may not be as committed or satisfied with their service.
   - **Recommendation**: Improve onboarding experience for new customers to increase their tenure. Offer introductory benefits or personalized onboarding. Consider retention incentives for customers nearing their first year.

4. **Contract (Importance: 0.077357)**
   - **Insight**: The type of contract (e.g., month-to-month, one year, two years) affects churn. Customers with shorter contracts are more likely to leave.
   - **Recommendation**: Encourage customers to commit to longer-term contracts. Offer discounts or incentives to those opting for long-term contracts to reduce churn.

5. **PaymentMethod (Importance: 0.050379)**
   - **Insight**: Payment method influences churn. Certain payment methods may correlate with higher churn due to payment issues or inconvenience.
   - **Recommendation**: Review the payment process for convenience. Offer multiple payment options and easy reminders. Consider discounts for customers using certain payment methods (e.g., auto-debit) to improve retention.

## Requirements

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
