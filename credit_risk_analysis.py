
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report

# Load the dataset for credit risk analysis
file_path = 'Task 3 and 4_Loan_Data.csv'
loan_data = pd.read_csv(file_path)

# Prepare the feature set and target variable
X = loan_data.drop(columns=['customer_id', 'default'])
y = loan_data['default']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale the features for better performance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a Logistic Regression model
model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)

# Predict probabilities of default on the test set
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

# Evaluate the model
roc_auc = roc_auc_score(y_test, y_pred_proba)
accuracy = accuracy_score(y_test, model.predict(X_test_scaled))
classification_rep = classification_report(y_test, model.predict(X_test_scaled))

print(f"ROC AUC Score: {roc_auc}")
print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(classification_rep)

def calculate_expected_loss(
    model, scaler, loan_amt_outstanding, total_debt_outstanding,
    income, years_employed, fico_score, credit_lines_outstanding,
    recovery_rate=0.1
):
    """
    Calculates the expected loss on a loan using the probability of default.

    Parameters:
    - model: The trained predictive model.
    - scaler: The scaler used to normalize the data.
    - loan_amt_outstanding: The amount of loan outstanding.
    - total_debt_outstanding: Total debt the borrower has.
    - income: Annual income of the borrower.
    - years_employed: Number of years the borrower has been employed.
    - fico_score: FICO score of the borrower.
    - credit_lines_outstanding: Number of credit lines currently open.
    - recovery_rate: The rate at which the bank can recover the defaulted loan (default is 10%).

    Returns:
    - The expected loss on the loan.
    """
    
    # Prepare the input data
    input_data = np.array([credit_lines_outstanding, loan_amt_outstanding, total_debt_outstanding, 
                           income, years_employed, fico_score]).reshape(1, -1)
    
    # Scale the input data
    input_data_scaled = scaler.transform(input_data)
    
    # Predict the probability of default
    probability_of_default = model.predict_proba(input_data_scaled)[0, 1]
    
    # Calculate the expected loss
    expected_loss = loan_amt_outstanding * probability_of_default * (1 - recovery_rate)
    
    return expected_loss

# Example calculation
example_loss = calculate_expected_loss(
    model, scaler, loan_amt_outstanding=5000, total_debt_outstanding=10000, 
    income=50000, years_employed=5, fico_score=600, credit_lines_outstanding=3
)

print(f"Expected Loss on Example Loan: ${example_loss}")
