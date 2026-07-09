import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# ==========================================
# Task 1: Load and preprocess the dataset
# ==========================================
california = fetch_california_housing()
df = pd.DataFrame(california.data, columns=california.feature_names)
df['Target'] = california.target

X_raw = df[['AveRooms']].values
y = df['Target'].values

# Filter out extreme outliers
mask = X_raw[:, 0] < 10 
X_filtered = X_raw[mask]
y_filtered = y[mask]

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_filtered)

# Add bias term
m = len(X_scaled)
X_b = np.c_[np.ones((m, 1)), X_scaled]

# ==========================================
# Task 2: Implement Linear Regression
# ==========================================
# --- Method A: Normal Equation ---
theta_normal = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_filtered)

# --- Method B: Gradient Descent (Convergence Loop) ---
learning_rate = 0.1

np.random.seed(42)
theta_gd = np.random.randn(2) 

iteration = 0
# We use a very strict tolerance to simulate "not changing at all"
# Exact equality (==) is dangerous in floating-point math, 
# but 1e-8 ensures it has mathematically converged to machine limits.
tolerance = 1e-8 

while True:
    prev_theta = theta_gd.copy()
    
    # Calculate gradients
    gradients = 2/m * X_b.T.dot(X_b.dot(theta_gd) - y_filtered)
    
    # Update weights
    theta_gd = theta_gd - learning_rate * gradients
    iteration += 1
    
    # Convergence check: exit if parameters haven't changed
    if np.max(np.abs(theta_gd - prev_theta)) < tolerance:
        break

y_predict_gd = X_b.dot(theta_gd)

# ==========================================
# Task 3: Evaluate model performance
# ==========================================
print(f"Gradient Descent converged after {iteration} iterations.")
print("--- Gradient Descent Performance ---")
print(f"MSE: {mean_squared_error(y_filtered, y_predict_gd):.4f}")
print(f"R-squared: {r2_score(y_filtered, y_predict_gd):.4f}\n")
print(f"Theta (Normal): {theta_normal}")
print(f"Theta (GD):     {theta_gd}")

# ==========================================
# Task 4: Visualize the fitted line
# ==========================================
plt.figure(figsize=(10, 6))
plt.scatter(X_scaled, y_filtered, alpha=0.2, color='blue', label='Data points', s=10)
plt.plot(X_scaled, y_predict_gd, color='yellow', linestyle='--', linewidth=2, label='Fitted Line (Gradient Descent)')
plt.title(f'Linear Regression: Converged in {iteration} Iterations')
plt.xlabel('Average Rooms per Dwelling (Standardized)')
plt.ylabel('Median House Value (in $100,000s)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('convergence_plot.png')
