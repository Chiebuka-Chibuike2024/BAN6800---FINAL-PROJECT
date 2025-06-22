# Import relevant libraries

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = 'C:/Users/user/OneDrive/Desktop/retail_price.csv'
df = pd.read_csv(file_path)

# Display the first few rows
print(df.head())

# Check column names
print(df.columns.tolist())

# Drop unnecessary columns
df = df.drop(columns=['product_id', 'product_category_name', 'month_year'])

# Define features (X) and target (y)
X = df.drop(columns=['unit_price'])
y = df['unit_price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")


# Plot actual vs predicted
plt.figure(figsize=(8, 5))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual Unit Price")
plt.ylabel("Predicted Unit Price")
plt.title("Actual vs Predicted Unit Price")
plt.grid(True)
plt.tight_layout()
plt.savefig("actual_vs_predicted.png")
plt.show()

import joblib

# Save the trained model
joblib.dump(model, 'model.pkl')
