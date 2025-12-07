import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import os

def prepare_data(df, target_col='TotalClaims'):
    """
    Prepares data for modeling.
    - Filters for positive claims (Risk Model).
    - Selects relevant features.
    - imputes missing values.
    - encodes categorical variables.
    - splits into train/test.
    """
    # 1. Filter for claims > 0 for Severity Model
    df_claims = df[df[target_col] > 0].copy()
    
    # 2. Feature Selection
    # Choosing a subset of potentially predictive features
    # Adding 'make' and 'bodytype' with correct casing
    categorical_features = ['Province', 'VehicleType', 'make', 'Gender', 'MaritalStatus', 'bodytype'] 
    numerical_features = ['TotalPremium', 'SumInsured', 'CalculatedPremiumPerTerm', 'Cylinders', 'cubiccapacity', 'kilowatts']

    # Keep only relevant columns
    features = categorical_features + numerical_features
    
    X = df_claims[features]
    y = df_claims[target_col]
    
    # 3. Handling Missing Data (Basic Imputation) & Encoding
    # We will use a Pipeline for this inside the model training, or pre-process here.
    # Let's pre-process split here for cleaner SHAP usage later.
    
    # Split first to avoid leakage
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, categorical_features, numerical_features

def build_pipeline(model, cat_features, num_features):
    """
    Builds a scikit-learn pipeline with preprocessing and model.
    """
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)) # sparse=False for easier SHAP
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_features),
            ('cat', categorical_transformer, cat_features)
        ])

    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('model', model)])
    
    return pipeline

def evaluate_model(model, X_test, y_test, name):
    """
    Evaluates the model and returns metrics.
    """
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"--- {name} Performance ---")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2 Score: {r2:.4f}")
    
    return {'Model': name, 'RMSE': rmse, 'R2': r2}

def explain_model_shap(model, X_train, X_test, cat_features, num_features, output_path='reports/figures/shap_summary.png'):
    """
    Generates SHAP summary plot.
    """
    # We need to access the transformed data and the final estimator
    preprocessor = model.named_steps['preprocessor']
    regressor = model.named_steps['model']
    
    # Transform data
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    
    # Get feature names
    cat_names = preprocessor.named_transformers_['cat']['encoder'].get_feature_names_out(cat_features)
    feature_names = num_features + list(cat_names)
    
    # Create explainer
    # Use TreeExplainer for Tree based models (XGB/RF), Linear for LinReg
    if isinstance(regressor, (xgb.XGBRegressor, RandomForestRegressor)):
        explainer = shap.TreeExplainer(regressor)
        # TreeExplainer might need a subsample if data is huge, but let's try full or subsample
        shap_values = explainer.shap_values(X_test_transformed)
    else:
        # Linear
        explainer = shap.LinearExplainer(regressor, X_train_transformed)
        shap_values = explainer.shap_values(X_test_transformed)
        
    # Plot
    plt.figure()
    shap.summary_plot(shap_values, X_test_transformed, feature_names=feature_names, show=False)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"SHAP summary plot saved to {output_path}")

