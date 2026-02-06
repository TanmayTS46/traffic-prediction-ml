import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Load preprocessed data
df = pd.read_csv('preprocessed_indian_traffic_data.csv')
df['date_time'] = pd.to_datetime(df['date_time'])  # For any time-based analysis

# Features and target (same as training)
features = joblib.load('model_features.pkl')
X = df[features]
y = df['traffic_volume']

# Re-split to get test set (use same random_state for consistency)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load the best model
best_model = joblib.load('best_traffic_model.pkl')

# Step 1: Make predictions on test set
y_pred = best_model.predict(X_test)

# Step 2: Compute evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"Evaluation Metrics on Test Set:")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.2f}")

# Step 3: Visualize Actual vs Predicted
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Traffic Volume')
plt.ylabel('Predicted Traffic Volume')
plt.title('Actual vs Predicted Traffic Volume')
plt.grid(True)
plt.savefig('actual_vs_predicted.png')  # Save for dashboard or report
plt.show()

# Step 4: Residual Analysis (to check model errors)
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True, color='green')
plt.xlabel('Residuals')
plt.title('Residual Distribution (Should be Normal for Good Model)')
plt.grid(True)
plt.savefig('residuals_histogram.png')
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.5, color='orange')
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Predicted Traffic Volume')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted (Check for Patterns)')
plt.grid(True)
plt.savefig('residuals_vs_predicted.png')
plt.show()

# Step 5: Feature Importance (if Random Forest; skip or adapt for Linear Regression)
if hasattr(best_model, 'feature_importances_'):
    importances = pd.DataFrame({
        'Feature': features,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=importances, palette='viridis')
    plt.title('Feature Importance')
    plt.grid(True)
    plt.savefig('feature_importance.png')
    plt.show()
    
    print("Top Features by Importance:")
    print(importances.head(10))
else:
    print("Feature importance not available for this model type (e.g., Linear Regression).")

# Optional: Cross-validation for more robust validation (if needed)
from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(best_model, X, y, cv=5, scoring='r2')
print(f"5-Fold Cross-Validation R² Scores: {cv_scores}")
print(f"Mean CV R²: {cv_scores.mean():.2f}")

print("Model evaluation complete. Visualizations saved as PNG files.")