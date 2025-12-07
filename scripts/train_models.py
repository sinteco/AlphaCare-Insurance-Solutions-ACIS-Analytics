import pandas as pd
import sys
import os
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.loader import load_data
from src.modeling import prepare_data, build_pipeline, evaluate_model, explain_model_shap

def main():
    print("Loading Data...")
    filepath = 'data/raw/MachineLearningRating_v3.txt'
    df = load_data(filepath)
    
    if df is None:
        return
        
    # Clean numeric columns for loading
    cols_to_numeric = ['TotalPremium', 'TotalClaims', 'SumInsured', 'CalculatedPremiumPerTerm', 'Cylinders', 'cubiccapacity', 'kilowatts']
    for col in cols_to_numeric:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    print("Preparing Data...")
    X_train, X_test, y_train, y_test, cat_features, num_features = prepare_data(df)
    
    print(f"Training Data Shape: {X_train.shape}")
    print(f"Testing Data Shape: {X_test.shape}")
    
    results = []
    
    # 1. Linear Regression
    print("\nTraining Linear Regression...")
    lr_pipeline = build_pipeline(LinearRegression(), cat_features, num_features)
    lr_pipeline.fit(X_train, y_train)
    results.append(evaluate_model(lr_pipeline, X_test, y_test, "Linear Regression"))
    
    # 2. Random Forest (Using limits to safe time/memory)
    print("\nTraining Random Forest...")
    rf_pipeline = build_pipeline(RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1), cat_features, num_features)
    rf_pipeline.fit(X_train, y_train)
    results.append(evaluate_model(rf_pipeline, X_test, y_test, "Random Forest"))
    
    # 3. XGBoost
    print("\nTraining XGBoost...")
    xgb_pipeline = build_pipeline(xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, n_jobs=-1), cat_features, num_features)
    xgb_pipeline.fit(X_train, y_train)
    results.append(evaluate_model(xgb_pipeline, X_test, y_test, "XGBoost"))
    
    # Compare
    print("\n--- Model Comparison ---")
    res_df = pd.DataFrame(results)
    print(res_df)
    
    # Explain Best Model (likely XGBoost or RF) with SHAP
    # Assuming XGBoost is best or close to best and good for SHAP
    print("\nGenerating SHAP explanation for XGBoost...")
    os.makedirs('reports/figures', exist_ok=True)
    explain_model_shap(xgb_pipeline, X_train, X_test, cat_features, num_features)

if __name__ == "__main__":
    main()
