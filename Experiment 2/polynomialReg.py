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

# Load the Auto MPG dataset
df = sns.load_dataset('mpg')

# Drop missing values
df = df.dropna(subset=['mpg', 'displacement'])

# Extract feature (X) and target (y)
X_raw = df[['displacement']].values
y = df['mpg'].values

# Create Polynomial Features from scratch (Degree 2)
# We add x^2 as a new feature column next to the original x
X_poly_raw = np.c_[X_raw, X_raw**2]

# Feature Scaling: Absolutely critical for Gradient Descent to converge 
# when dealing with polynomial features (like displacement^2 which gets very large)
scaler = StandardScaler()
X_poly_scaled = scaler.fit_transform(X_poly_raw)

# Add a bias term (a column of ones, x0 = 1) to X for vectorized operations
m = len(X_poly_scaled)
X_b = np.c_[np.ones((m, 1)), X_poly_scaled]

# ==========================================
# Task 2: Implement Polynomial Regression
# ==========================================

# --- Method A: Normal Equation ---
# Mathematical optimization using: θ = (X^T * X)^-1 * X^T * y
theta_normal = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
y_predict_normal = X_b.dot(theta_normal)

# --- Method B: Gradient Descent ---
learning_rate = 0.1
n_iterations = 1000

# Random initialization of weights (3 weights: bias, x, x^2)
np.random.seed(42)
theta_gd = np.random.randn(3) 

for iteration in range(n_iterations):
    # Calculate gradients
    gradients = 2/m * X_b.T.dot(X_b.dot(theta_gd) - y)
    # Update weights
    theta_gd = theta_gd - learning_rate * gradients

y_predict_gd = X_b.dot(theta_gd)

# ==========================================
# Task 3: Scikit-Learn Implementation
# ==========================================

# We pass the same scaled polynomial features to sklearn for a direct comparison
model = LinearRegression()
model.fit(X_poly_scaled, y)
y_predict_sklearn = model.predict(X_poly_scaled)

# ==========================================
# Task 4: Evaluate model performance
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

print(f"Theta (Normal): {theta_normal}")
print(f"Theta (GD):     {theta_gd}")
print(f"Theta (Sklearn): [{model.intercept_} {model.coef_[0]} {model.coef_[1]}]")

# ==========================================
# Task 5: Visualize the fitted curves
# ==========================================

# To plot a smooth curve, we must sort the X values and their corresponding predictions
sort_index = np.argsort(X_raw[:, 0])
X_plot = X_raw[sort_index]
y_plot_normal = y_predict_normal[sort_index]
y_plot_gd = y_predict_gd[sort_index]
y_plot_sklearn = y_predict_sklearn[sort_index]

plt.figure(figsize=(10, 6))

# Scatter plot of the original data points
plt.scatter(X_raw, y, alpha=0.4, color='gray', label='Actual Data', s=20)

# Plot the regression curves
# We use thicker lines and varying styles so overlapping curves are visible
plt.plot(X_plot, y_plot_normal, color='red', linewidth=4, label='Fitted Curve (Normal Eq)')
plt.plot(X_plot, y_plot_gd, color='yellow', linestyle='--', linewidth=3, label='Fitted Curve (Gradient Descent)')
plt.plot(X_plot, y_plot_sklearn, color='black', linestyle=':', linewidth=2, label='Fitted Curve (sklearn)')

plt.title('Polynomial Regression (Degree 2): MPG vs. Displacement')
plt.xlabel('Engine Displacement (cu. inches)')
plt.ylabel('Miles Per Gallon (MPG)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('exp2.png')