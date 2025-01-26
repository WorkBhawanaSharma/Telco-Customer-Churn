#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Step 1: Importing Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report


# In[8]:


# Step 2: Loading the Dataset
data = pd.read_csv(r'C:\Users\kdbha\Desktop\Pyython\WA_Fn-UseC_-Telco-Customer-Churn.csv')


# In[9]:


# Step 3: Data Exploration
# Checking first few rows of the dataset
print("Dataset Head:")
print(data.head())


# In[10]:


# info about the dataset
print("\nDataset Info:")
print(data.info())


# In[11]:


# Checking missing values
print("\nMissing Values:")
print(data.isnull().sum())


# In[12]:


# Checking summary statistics for numerical columns
print("\nSummary Statistics:")
print(data.describe())


# In[13]:


# Analyzing churn distribution
sns.countplot(x='Churn', data=data)
plt.title("Churn Distribution")
plt.show()


# In[14]:


# Analyzing trends- Monthly Charges vs Churn
sns.boxplot(x='Churn', y='MonthlyCharges', data=data)
plt.title("Monthly Charges vs Churn")
plt.show()


# In[15]:


# Analyzing trends- Contract type vs Churn
sns.countplot(x='Contract', hue='Churn', data=data)
plt.title("Churn by Contract Type")
plt.show()


# In[16]:


# Step 4: Data Cleaning & Preprocessing
#missing or inconsistent data
# Convert 'TotalCharges' to numeric and handle any invalid entries
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
data.fillna({'TotalCharges': data['TotalCharges'].median()}, inplace=True)


# In[17]:


# Normalize numerical features (MonthlyCharges and TotalCharges)
scaler = StandardScaler()
data[['MonthlyCharges', 'TotalCharges']] = scaler.fit_transform(data[['MonthlyCharges', 'TotalCharges']])


# In[19]:


# Encode categorical variables
label_encoders = {}
categorical_columns = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 
                       'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                       'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 
                       'PaperlessBilling', 'PaymentMethod']

for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le


# In[20]:


# Step 5: Model Development
# Defining target variable and features
X = data.drop(columns=['customerID', 'Churn'])
y = data['Churn']


# In[21]:


# Splitting the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[22]:


# Training a Random Forest Classifier
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)


# In[24]:


# Evaluate the model
y_pred = rf_model.predict(X_test)

precision = precision_score(y_test, y_pred, pos_label='Yes')
recall = recall_score(y_test, y_pred, pos_label='Yes')
f1 = f1_score(y_test, y_pred, pos_label='Yes')

print("\nModel Evaluation Metrics:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")


# In[25]:


# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# In[26]:


# Step 6: Insights & Recommendations
# Feature Importance
feature_importances = pd.DataFrame({'Feature': X.columns, 'Importance': rf_model.feature_importances_})
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

print("\nFeature Importances:")
print(feature_importances)


# In[2]:


# Step 6: Insights & Recommendations
print("\nInsights & Recommendations Based on Feature Importances:")

# Insight and recommendations based on TotalCharges
print("\n1. TotalCharges (Importance: 0.189992)")
print("   Insight: The total amount a customer has spent with the company is the most significant factor in predicting churn.")
print("           Customers with lower total charges are more likely to churn, suggesting a correlation between how long customers have been with the company and their likelihood to leave.")
print("   Recommendation: Focus on retaining long-term customers by offering loyalty rewards, discounts, or special offers to increase their total spending. Additionally, consider targeting low-spending customers with promotions to increase their lifetime value.")

# Insight and recommendations based on MonthlyCharges
print("\n2. MonthlyCharges (Importance: 0.177867)")
print("   Insight: Monthly charges are also a strong predictor of churn. Higher monthly charges could be a deterrent for customers, particularly if they perceive they aren't getting enough value for the cost.")
print("   Recommendation: Evaluate your pricing structure to ensure it aligns with the value customers expect. Offering more affordable plans or value-added services could reduce churn, especially in competitive markets.")

# Insight and recommendations based on Tenure
print("\n3. Tenure (Importance: 0.157428)")
print("   Insight: Tenure, or the length of time a customer has been with the company, plays a key role. Customers with shorter tenures are more likely to churn, which indicates that new customers may be less committed or satisfied with their service.")
print("   Recommendation: Improve the onboarding experience for new customers to increase their tenure. Offering introductory benefits or personalized onboarding could help strengthen relationships early on. Additionally, consider offering retention incentives for customers approaching their first year with the company.")

# Insight and recommendations based on Contract
print("\n4. Contract (Importance: 0.077357)")
print("   Insight: The type of contract a customer holds (e.g., month-to-month, one year, two years) affects their churn likelihood. Customers with shorter contracts (such as month-to-month) are more likely to leave.")
print("   Recommendation: Encourage customers to commit to longer-term contracts, which can offer stability for both the customer and the company. Consider offering discounts or incentives to customers who opt for longer contracts, making them less likely to churn.")

# Insight and recommendations based on PaymentMethod
print("\n5. PaymentMethod (Importance: 0.050379)")
print("   Insight: The method by which customers pay their bills, such as credit card, bank transfer, or electronic billing, also influences churn. Certain payment methods may correlate with higher churn, possibly due to issues with payment failures or convenience.")
print("   Recommendation: Review the payment process to ensure it is seamless and convenient. Providing multiple payment options, as well as easy payment reminders, could reduce the likelihood of churn. Offering automated billing or discounts for certain payment methods (e.g., auto-debit) could also improve retention.")


# In[ ]:




