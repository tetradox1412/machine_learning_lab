import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# ==========================================
# Task 1: Load, Clean, and Get User Input
# ==========================================

# Load the Auto MPG dataset
df = sns.load_dataset('mpg').dropna(subset=['mpg', 'displacement'])
X_raw = df[['displacement']].values
y = df['mpg'].values
m = len(y)

# Dynamic User Input for Polynomial Degree
try:
    user_input = input("Enter the desired polynomial degree (e.g., 2, 3, 5): ")
    degree = int(user_input)
    if degree < 1:
        print("Degree must be at least 1. Defaulting to Degree 2.")
        degree = 2
except ValueError:
    print("Invalid integer input. Defaulting to Degree 2.")
    degree = 2

print(f"\n🚀 Initializing Comparison: Degree {degree} Polynomial vs. Degree 1 Linear Baseline...\n")

# Helper function to generate polynomial features from scratch without sklearn
def create_poly_features(X, deg):
    return np.hstack([X**i for i in range(1, deg + 1)])

# ==========================================
# Task 2: Baseline Linear Regression (Degree 1)
# ==========================================

# Create, scale, and add bias term for Degree 1
X_lin_raw = create_poly_features(X_raw, 1)
scaler_lin = StandardScaler()
X_lin_scaled = scaler_lin.fit_transform(X_lin_raw)
X_lin_b = np.c_[np.ones((m, 1)), X_lin_scaled]

# Exact solution via Normal Equation for the linear baseline
theta_linear = np.linalg.inv(X_lin_b.T.dot(X_lin_b)).dot(X_lin_b.T).dot(y)
y_pred_linear = X_lin_b.dot(theta_linear)

# ==========================================
# Task 3: Dynamic Polynomial Regression (Degree N)
# ==========================================

# Create, scale, and add bias term for User Degree N
X_poly_raw = create_poly_features(X_raw, degree)
scaler_poly = StandardScaler()
X_poly_scaled = scaler_poly.fit_transform(X_poly_raw)
X_poly_b = np.c_[np.ones((m, 1)), X_poly_scaled]

# --- Method A: Normal Equation ---
theta_poly_normal = np.linalg.inv(X_poly_b.T.dot(X_poly_b)).dot(X_poly_b.T).dot(y)
y_pred_poly_normal = X_poly_b.dot(theta_poly_normal)

# --- Method B: Gradient Descent with Dynamic Convergence Loop ---
learning_rate = 0.05  # Slightly lowered to maintain stability across higher degrees
tolerance = 1e-6      # Convergence threshold
max_iterations = 100000 # Safety cap against floating-point machine epsilon oscillation

np.random.seed(42)
theta_poly_gd = np.random.randn(degree + 1)

iterations = 0
while True:
    theta_prev = np.copy(theta_poly_gd)
    
    # Vectorized gradient calculation
    gradients = 2/m * X_poly_b.T.dot(X_poly_b.dot(theta_poly_gd) - y)
    theta_poly_gd = theta_poly_gd - learning_rate * gradients
    iterations += 1
    
    # Check if the maximum weight change is below our tolerance threshold
    parameter_change = np.max(np.abs(theta_poly_gd - theta_prev))
    if parameter_change < tolerance:
        print(f"🔥 Gradient Descent converged after {iterations:,} iterations!")
        break
    if iterations >= max_iterations:
        print(f"⚠️ Reached safety limit of {max_iterations:,} iterations. Max weight change: {parameter_change:.8f}")
        break

y_pred_poly_gd = X_poly_b.dot(theta_poly_gd)

# --- Method C: Scikit-Learn Reference ---
model_poly = LinearRegression()
model_poly.fit(X_poly_scaled, y)
y_pred_poly_sklearn = model_poly.predict(X_poly_scaled)

# ==========================================
# Task 4: Evaluate & Compare Performance
# ==========================================

metrics = [
    {"Model": "Baseline Linear (Deg 1)", "MSE": mean_squared_error(y, y_pred_linear), "R-Squared": r2_score(y, y_pred_linear)},
    {"Model": f"Poly Normal Eq (Deg {degree})", "MSE": mean_squared_error(y, y_pred_poly_normal), "R-Squared": r2_score(y, y_pred_poly_normal)},
    {"Model": f"Poly Grad Descent (Deg {degree})", "MSE": mean_squared_error(y, y_pred_poly_gd), "R-Squared": r2_score(y, y_pred_poly_gd)},
    {"Model": f"Poly Scikit-Learn (Deg {degree})", "MSE": mean_squared_error(y, y_pred_poly_sklearn), "R-Squared": r2_score(y, y_pred_poly_sklearn)}
]

metrics_df = pd.DataFrame(metrics)
print("\nModel Performance Comparison:")
print("-" * 65)
print(metrics_df.to_string(index=False))
print("-" * 65 + "\n")

# ==========================================
# Task 5: Simultaneous Visualization
# ==========================================

# Generate a smooth continuous sequence of X values for plotting clean curves
X_plot_continuous = np.linspace(X_raw.min(), X_raw.max(), 200).reshape(-1, 1)

# 1. Transform and predict for Linear Baseline
X_plot_lin_scaled = scaler_lin.transform(create_poly_features(X_plot_continuous, 1))
X_plot_lin_b = np.c_[np.ones((len(X_plot_lin_scaled), 1)), X_plot_lin_scaled]
y_plot_linear = X_plot_lin_b.dot(theta_linear)

# 2. Transform and predict for Polynomial Models
X_plot_poly_scaled = scaler_poly.transform(create_poly_features(X_plot_continuous, degree))
X_plot_poly_b = np.c_[np.ones((len(X_plot_poly_scaled), 1)), X_plot_poly_scaled]

y_plot_poly_normal = X_plot_poly_b.dot(theta_poly_normal)
y_plot_poly_gd = X_plot_poly_b.dot(theta_poly_gd)
y_plot_poly_sklearn = model_poly.predict(X_plot_poly_scaled)

# Plotting
plt.figure(figsize=(11, 7))

# Scatter plot of actual dataset
plt.scatter(X_raw, y, alpha=0.35, color='gray', label='Actual Data', s=25)

# Plot Linear Regression Baseline
plt.plot(X_plot_continuous, y_plot_linear, color='blue', linestyle='-.', linewidth=2.5, 
         label='Baseline Linear Regression (Degree 1)')

# Plot Polynomial Fits (Overlaying styles to verify convergence visually)
if degree > 1:
    plt.plot(X_plot_continuous, y_plot_poly_normal, color='red', linewidth=5, alpha=0.7, 
             label=f'Polynomial Normal Eq (Degree {degree})')
    plt.plot(X_plot_continuous, y_plot_poly_gd, color='yellow', linestyle='--', linewidth=3, 
             label=f'Polynomial Gradient Descent (Degree {degree})')
    plt.plot(X_plot_continuous, y_plot_poly_sklearn, color='black', linestyle=':', linewidth=2, 
             label=f'Polynomial Scikit-Learn (Degree {degree})')

plt.title(f'MPG vs. Displacement: Linear Baseline vs. Degree {degree} Polynomial Fit', fontsize=14, pad=15)
plt.xlabel('Engine Displacement (cu. inches)', fontsize=12)
plt.ylabel('Miles Per Gallon (MPG)', fontsize=12)
plt.legend(frameon=True, facecolor='white', framealpha=0.9)
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()