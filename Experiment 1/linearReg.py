import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# ==========================================
# Task 1: Load and preprocess the dataset
# ==========================================

# Fetch the dataset and load it into a pandas DataFrame
california = fetch_california_housing()
df = pd.DataFrame(california.data, columns=california.feature_names)
df['Target'] = california.target

# Select the single feature: average number of rooms per dwelling ('AveRooms')
X_raw = df[['AveRooms']].values
y = df['Target'].values

# Preprocessing: Filter out extreme outliers in 'AveRooms' for a more meaningful linear fit
# (The dataset contains some estates with an unusually high number of rooms)
mask = X_raw[:, 0] < 10 
X_filtered = X_raw[mask]
y_filtered = y[mask]

# Feature Scaling: Crucial for Gradient Descent to converge efficiently
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_filtered)

# Add a bias term (a column of ones, x0 = 1) to X for vectorized operations
m = len(X_scaled)
X_b = np.c_[np.ones((m, 1)), X_scaled]

# ==========================================
# Task 2: Implement Linear Regression
# ==========================================

# --- Method A: Normal Equation ---
# Using NumPy's linear algebra module for optimal memory performance
theta_normal = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_filtered)
y_predict_normal = X_b.dot(theta_normal)

# --- Method B: Gradient Descent ---
learning_rate = 0.1
n_iterations = 1000

# Random initialization of weights
np.random.seed(42)
theta_gd = np.random.randn(2) 

for iteration in range(n_iterations):
    # Calculate gradients
    gradients = 2/m * X_b.T.dot(X_b.dot(theta_gd) - y_filtered)
    # Update weights
    theta_gd = theta_gd - learning_rate * gradients

y_predict_gd = X_b.dot(theta_gd)

# ==========================================
# Task 3: Evaluate model performance
# ==========================================

print("--- Normal Equation Performance ---")
print(f"MSE: {mean_squared_error(y_filtered, y_predict_normal):.4f}")
print(f"R-squared: {r2_score(y_filtered, y_predict_normal):.4f}\n")

print("--- Gradient Descent Performance ---")
print(f"MSE: {mean_squared_error(y_filtered, y_predict_gd):.4f}")
print(f"R-squared: {r2_score(y_filtered, y_predict_gd):.4f}\n")

print(f"Theta (Normal): {theta_normal}")
print(f"Theta (GD):     {theta_gd}")


# SKLearn Implementation for comparison
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
# ==========================================
# Task 1: Load and preprocess the dataset
# ==========================================

# Fetch data and extract 'AveRooms' (Index 2) and target
california = fetch_california_housing()
X_raw = california.data[:, 2].reshape(-1, 1) 
y = california.target

# Filter outliers to match previous visualization
mask = X_raw[:, 0] < 10 
X_filtered = X_raw[mask]
y_filtered = y[mask]

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_filtered)

# ==========================================
# Task 2: Implement Scikit-Learn Linear Regression
# ==========================================

# Instantiate and fit the model in two lines
model = LinearRegression()
model.fit(X_scaled, y_filtered)

# Generate predictions
y_predict_sklearn = model.predict(X_scaled)

# ==========================================
# Task 3: Evaluate model performance
# ==========================================

print("--- Scikit-Learn Linear Regression Performance ---")
print(f"MSE: {mean_squared_error(y_filtered, y_predict_sklearn):.4f}")
print(f"R-squared: {r2_score(y_filtered, y_predict_sklearn):.4f}")
print(f"Intercept (Bias): {model.intercept_:.4f}")
print(f"Coefficient (Weight): {model.coef_[0]:.4f}\n")

# ==========================================
# Task 4: Visualize the fitted line
# ==========================================

# Scatter plot of the original data points
plt.scatter(X_scaled, y_filtered, alpha=0.2, color='blue', label='Data points', s=10)

# Plot the regression line (both will overlap if they converged correctly)
plt.plot(X_scaled, y_predict_normal, color='red', linewidth=2, label='Fitted Line (Normal Eq)')
plt.plot(X_scaled, y_predict_gd, color='yellow', linestyle='--', linewidth=2, label='Fitted Line (Gradient Descent)')
plt.plot(X_scaled, y_predict_sklearn, color='green', linewidth=2, label='Fitted Line (sklearn)')

plt.title('Linear Regression: California Housing (AveRooms vs Price)')
plt.xlabel('Average Rooms per Dwelling (Standardized)')
plt.ylabel('Median House Value (in $100,000s)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()