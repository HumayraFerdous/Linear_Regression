import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

df = pd.read_csv('house_price.csv')

print(df.head())
print(df.isnull().sum())
X = df[["area", "bedrooms", "age"]]  # Features
y = df["price"]

X_Train,X_test,y_train,y_test = train_test_split(X.values,y,test_size=0.6,random_state=42)
model = LinearRegression()
model.fit(X_Train,y_train)

y_pred=model.predict(X_test)
print("\nModel Coefficients:")
print(f"Intercept (β₀): {model.intercept_:.2f}")
print(f"Coefficients (β₁, β₂, β₃): {model.coef_}")

print("\nModel Evaluation Metrics:")
print(f"R-squared (R²): {r2_score(y_test, y_pred):.2f}")
print(f"Mean Absolute Error (MAE): ${mean_absolute_error(y_test, y_pred):.2f}")
print(f"Mean Squared Error (MSE): ${mean_squared_error(y_test, y_pred):.2f}")
print(f"Root Mean Squared Error (RMSE): ${np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")

new_house = [[1800,3,5]]
predicted_price = model.predict(new_house)
print(f"\nPredicted price for a 1800 sqft, 3-bed, 5-year-old house: ${predicted_price[0]:.2f}")

plt.figure(figsize=(10,6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Actual Price ($)")
plt.ylabel("Predicted Price ($)")
plt.title("Actual vs Predicted House Prices")
plt.grid(True)
plt.show()