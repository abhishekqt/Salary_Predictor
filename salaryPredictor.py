# ============================================
#   SALARY PREDICTOR USING LINEAR REGRESSION
#   AIML Mini Project - 1st Year
#   Made by: Abhishek Singh
#   Date: 28 March 2026
# ============================================

# Step 1: Import libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# -----------------------------------------------
# Step 2: Sample Data
# Years of experience vs Salary (in Rupees)
# -----------------------------------------------

experience = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
salary     = np.array([300000, 380000, 460000, 550000, 640000,
                       730000, 820000, 920000, 1010000, 1100000])

# Reshape experience for sklearn (needs 2D array)
X = experience.reshape(-1, 1)
y = salary

# -----------------------------------------------
# Step 3: Train the Linear Regression Model
# -----------------------------------------------

model = LinearRegression()
model.fit(X, y)

print("=" * 50)
print("   SALARY PREDICTOR - Linear Regression")
print("=" * 50)

# -----------------------------------------------
# Step 4: Take Input from User
# -----------------------------------------------

current_exp = float(input("\nEnter your current years of experience: "))

# -----------------------------------------------
# Step 5: Predict Salary for Next 5 Years
# -----------------------------------------------

print("\nPredicted Salary for Next 5 Years:")
print("-" * 40)
print(f"{'Year':<8} {'Experience':<15} {'Salary (Rs.)'}")
print("-" * 40)

future_exp    = []
future_salary = []

for i in range(1, 6):
    exp = current_exp + i
    pred_salary = model.predict([[exp]])[0]
    future_exp.append(exp)
    future_salary.append(pred_salary)
    print(f"Year {i:<4} {exp:<15.1f} Rs. {pred_salary:,.0f}")

print("-" * 40)
print(f"\nModel R2 Score : {model.score(X, y):.4f}")
print(f"Salary hike/yr : Rs. {model.coef_[0]:,.0f}")

# -----------------------------------------------
# Step 6: Plot the Graph
# -----------------------------------------------

plt.figure(figsize=(10, 6))

# Plot training data points
plt.scatter(experience, salary, color='blue', s=80,
            label='Training Data', zorder=5)

# Plot regression line
x_range = np.linspace(1, current_exp + 6, 100).reshape(-1, 1)
plt.plot(x_range, model.predict(x_range), color='green',
         linewidth=2, label='Regression Line')

# Plot future predicted points
plt.scatter(future_exp, future_salary, color='red', s=120,
            marker='*', label='Future Predictions', zorder=5)

# Mark current year
plt.axvline(x=current_exp, color='orange', linestyle='--',
            linewidth=1.5, label=f'You are here ({current_exp} yrs)')

# Labels
plt.title('Salary Prediction for Next 5 Years', fontsize=14)
plt.xlabel('Years of Experience', fontsize=12)
plt.ylabel('Salary (INR)', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('salary_prediction.png', dpi=150)
plt.show()

print("\nGraph saved as 'salary_prediction.png'")
print("\n[Project Complete] - 1st Year AIML Mini Project")
