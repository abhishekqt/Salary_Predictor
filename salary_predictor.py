# Salary Predictor using Linear Regression
# This program predicts the salary of an employee based on their years of experience.
# It uses a simple dataset and trains a linear regression model on it.

# Importing the required libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# -------------------------------------------------------
# Step 1: Create a small sample dataset
# (Years of experience and corresponding salary in INR)
# -------------------------------------------------------

experience = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
salary     = np.array([30000, 35000, 42000, 48000, 55000,
                       62000, 70000, 78000, 85000, 92000])

print("=== Sample Dataset ===")
print(f"{'Experience (yrs)':<20} {'Salary (INR)'}")
print("-" * 35)
for exp, sal in zip(experience.flatten(), salary):
    print(f"{exp:<20} {sal}")

# -------------------------------------------------------
# Step 2: Split the data into training and testing sets
# -------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    experience, salary, test_size=0.2, random_state=42
)

# -------------------------------------------------------
# Step 3: Train the Linear Regression model
# -------------------------------------------------------

model = LinearRegression()
model.fit(X_train, y_train)

print("\n=== Model Training Complete ===")
print(f"Slope (coefficient) : {model.coef_[0]:.2f}")
print(f"Intercept           : {model.intercept_:.2f}")

# This basically means: Salary = slope * experience + intercept

# -------------------------------------------------------
# Step 4: Test the model and check accuracy
# -------------------------------------------------------

y_pred = model.predict(X_test)

print("\n=== Model Evaluation ===")
print(f"Mean Absolute Error : {mean_absolute_error(y_test, y_pred):.2f} INR")
print(f"R² Score            : {r2_score(y_test, y_pred):.4f}")
print("(R² closer to 1.0 means the model is more accurate)")

# -------------------------------------------------------
# Step 5: Predict salary for a new employee
# -------------------------------------------------------

print("\n=== Salary Prediction ===")

try:
    user_exp = float(input("Enter years of experience to predict salary: "))
    predicted_salary = model.predict([[user_exp]])
    print(f"Predicted salary for {user_exp} years of experience: ₹{predicted_salary[0]:,.2f}")
except ValueError:
    print("Please enter a valid number.")

# -------------------------------------------------------
# Step 6: Plot the results
# -------------------------------------------------------

plt.figure(figsize=(8, 5))

# Plot actual data points
plt.scatter(experience, salary, color='steelblue', label='Actual Data', zorder=5)

# Plot the regression line
x_line = np.linspace(0, 11, 100).reshape(-1, 1)
y_line = model.predict(x_line)
plt.plot(x_line, y_line, color='tomato', linewidth=2, label='Regression Line')

plt.title("Salary vs Experience - Linear Regression")
plt.xlabel("Years of Experience")
plt.ylabel("Salary (INR)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("salary_vs_experience.png")
plt.show()

print("\nPlot saved as 'salary_vs_experience.png'")
