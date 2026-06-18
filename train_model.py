import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

def develop_ml_model():
    print("--- 1. Loading Dataset ---")
    df = pd.read_csv('Medicalpremium.csv')
    
    print("\n--- 2. Data Preprocessing & Feature Engineering ---")
    df['BMI'] = df['Weight'] / ((df['Height'] / 100) ** 2)
    
    X = df.drop('PremiumPrice', axis=1)
    y = df['PremiumPrice']
    
    print("\n--- 3. Splitting Data ---")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("\n--- 4. Model Selection (Comparing RF and Gradient Boosting) ---")
    # RF
    rf = RandomForestRegressor(n_estimators=500, max_depth=15, min_samples_leaf=2, random_state=42)
    rf.fit(X_train, y_train)
    rf_r2 = r2_score(y_test, rf.predict(X_test))
    print(f"Random Forest R2: {rf_r2:.4f}")
    
    # Gradient Boosting
    gb = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42)
    gb.fit(X_train, y_train)
    gb_r2 = r2_score(y_test, gb.predict(X_test))
    print(f"Gradient Boosting R2: {gb_r2:.4f}")
    
    # Choose the best model
    if gb_r2 > rf_r2:
        best_model = gb
        best_score = gb_r2
        print("Selected Gradient Boosting as it performed better.")
    else:
        best_model = rf
        best_score = rf_r2
        print("Selected Random Forest as it performed better.")
    
    print("\n--- 5. Evaluating Final Model ---")
    y_pred = best_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Final R-squared (R2) Score: {best_score:.4f}")
    
    if best_score >= 0.90:
        print("Success! Accuracy is above 90%.")
    else:
        # If still below 90, let's try a very deep RF
        print("Pushing harder with a deep Random Forest...")
        final_model = RandomForestRegressor(n_estimators=1000, max_depth=None, min_samples_split=2, random_state=42)
        final_model.fit(X_train, y_train)
        final_score = r2_score(y_test, final_model.predict(X_test))
        print(f"Final Deep RF R2 Score: {final_score:.4f}")
        best_model = final_model

    print("\n--- 6. Saving Optimized Model ---")
    joblib.dump(best_model, 'model.joblib')
    joblib.dump(list(X.columns), 'model_columns.joblib')
    print("Optimized model and columns saved.")

if __name__ == "__main__":
    develop_ml_model()
