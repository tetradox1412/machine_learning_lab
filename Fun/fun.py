import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# ==========================================
# Task 1: Load and preprocess the dataset
# ==========================================
df = sns.load_dataset('mpg')
df = df.dropna(subset=['mpg', 'displacement'])

X_raw = df[['displacement']].values
y = df['mpg'].values

# Create Polynomial Features (Degree 2)
X_poly_raw = np.c_[X_raw, X_raw**2]

# Feature Scaling
scaler = StandardScaler()
X_poly_scaled = scaler.fit_transform(X_poly_raw)

# Add a bias term (a column of ones, x0 = 1)
m = len(X_poly_scaled)
X_b = np.c_[np.ones((m, 1)), X_poly_scaled]

# ==========================================
# Task 2: Implement Models
# ==========================================

# --- Method A: Normal Equation ---
theta_normal = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
y_predict_normal = X_b.dot(theta_normal)

# --- Method B: Gradient Descent with Convergence Loop ---
learning_rate = 0.1
tolerance = 1e-6  # The threshold to define "hasn't learned anything new"

np.random.seed(42)
theta_gd = np.random.randn(3) 

iterations = 0
while True:
    # 1. Save the state of the parameters before updating
    theta_prev = np.copy(theta_gd)
    
    # 2. Calculate gradients
    gradients = 2/m * X_b.T.dot(X_b.dot(theta_gd) - y)
    
    # 3. Update weights
    theta_gd = theta_gd - learning_rate * gradients
    
    iterations += 1
    
    # 4. Check for convergence: If the maximum change in any parameter 
    # is smaller than our tolerance, we exit the loop.
    parameter_change = np.max(np.abs(theta_gd - theta_prev))
    if parameter_change < tolerance:
        break

print(f"Gradient Descent reached convergence after {iterations} iterations!\n")
y_predict_gd = X_b.dot(theta_gd)

# --- Method C: Scikit-Learn ---
model = LinearRegression()
model.fit(X_poly_scaled, y)
y_predict_sklearn = model.predict(X_poly_scaled)

# ==========================================
# Task 3: Evaluate model performance
# ==========================================

print("--- Normal Equation Performance ---")
print(f"MSE: {mean_squared_error(y, y_predict_normal):.4f}")
print(f"R-squared: {r2_score(y, y_predict_normal):.4f}\n")

print("--- Gradient Descent Performance ---")
print(f"MSE: {mean_squared_error(y, y_predict_gd):.4f}")
print(f"R-squared: {r2_score(y, y_predict_gd):.4f}\n")

print("--- Scikit-Learn Performance ---")
print(f"MSE: {mean_squared_error(y, y_predict_sklearn):.4f}")
print(f"R-squared: {r2_score(y, y_predict_sklearn):.4f}\n")

print(f"Theta (Normal):  {theta_normal}")
print(f"Theta (GD):      {theta_gd}")
print(f"Theta (Sklearn): [{model.intercept_} {model.coef_[0]} {model.coef_[1]}]\n")

# ==========================================
# Task 4: Visualize the fitted curves
# ==========================================

sort_index = np.argsort(X_raw[:, 0])
X_plot = X_raw[sort_index]
y_plot_normal = y_predict_normal[sort_index]
y_plot_gd = y_predict_gd[sort_index]
y_plot_sklearn = y_predict_sklearn[sort_index]

plt.figure(figsize=(10, 6))

plt.scatter(X_raw, y, alpha=0.4, color='gray', label='Actual Data', s=20)
plt.plot(X_plot, y_plot_normal, color='red', linewidth=4, label='Fitted Curve (Normal Eq)')
plt.plot(X_plot, y_plot_gd, color='yellow', linestyle='--', linewidth=3, label='Fitted Curve (Gradient Descent)')
plt.plot(X_plot, y_plot_sklearn, color='black', linestyle=':', linewidth=2, label='Fitted Curve (sklearn)')

plt.title('Polynomial Regression: Dynamic Convergence Loop')
plt.xlabel('Engine Displacement (cu. inches)')
plt.ylabel('Miles Per Gallon (MPG)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()