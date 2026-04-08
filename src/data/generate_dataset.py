"""
Generates a synthetic Bank Customer Churn dataset that mirrors the
real-world Kaggle Churn_Modelling.csv schema and class imbalance (~20% churn).
"""
import numpy as np
import pandas as pd
import os

np.random.seed(42)
N = 10_000

geography = np.random.choice(["France", "Germany", "Spain"], size=N, p=[0.50, 0.25, 0.25])
gender = np.random.choice(["Male", "Female"], size=N, p=[0.55, 0.45])
age = np.random.normal(38, 10, N).clip(18, 92).astype(int)
tenure = np.random.randint(0, 11, N)
balance = np.where(np.random.rand(N) < 0.29, 0, np.random.normal(76_000, 62_000, N).clip(0))
num_products = np.random.choice([1, 2, 3, 4], size=N, p=[0.50, 0.46, 0.03, 0.01])
has_cr_card = np.random.choice([0, 1], size=N, p=[0.29, 0.71])
is_active_member = np.random.choice([0, 1], size=N, p=[0.49, 0.51])
estimated_salary = np.random.uniform(11, 200_000, N)
credit_score = np.random.normal(650, 97, N).clip(350, 850).astype(int)

# Churn probability influenced by real-world patterns
churn_prob = (
    0.05
    + 0.15 * (age > 45)
    + 0.10 * (geography == "Germany")
    + 0.08 * (num_products >= 3)
    + 0.07 * (balance == 0)
    - 0.06 * is_active_member
    - 0.03 * has_cr_card
)
churn_prob = np.clip(churn_prob, 0, 1)
exited = (np.random.rand(N) < churn_prob).astype(int)

df = pd.DataFrame({
    "RowNumber": range(1, N + 1),
    "CustomerId": np.random.randint(15_000_000, 16_000_000, N),
    "Surname": ["Customer_" + str(i) for i in range(N)],
    "CreditScore": credit_score,
    "Geography": geography,
    "Gender": gender,
    "Age": age,
    "Tenure": tenure,
    "Balance": balance.round(2),
    "NumOfProducts": num_products,
    "HasCrCard": has_cr_card,
    "IsActiveMember": is_active_member,
    "EstimatedSalary": estimated_salary.round(2),
    "Exited": exited,
})

output_path = os.path.join("data", "raw", "churn.csv")
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df.to_csv(output_path, index=False)

print(f"Dataset saved to {output_path}")
print(f"Shape: {df.shape}")
print(f"Churn rate: {exited.mean():.2%}")
print(df.head(3))
