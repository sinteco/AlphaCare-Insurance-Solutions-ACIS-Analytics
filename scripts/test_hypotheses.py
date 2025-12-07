import pandas as pd
import sys
import os

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.loader import load_data
from src.hypothesis_testing import test_risk_differences_categorical, test_margin_differences

def run_tests():
    print("Loading data...")
    filepath = 'data/raw/MachineLearningRating_v3.txt'
    df = load_data(filepath)
    
    if df is None:
        return

    # Clean data similar to EDA
    cols_to_numeric = ['TotalPremium', 'TotalClaims']
    for col in cols_to_numeric:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    # Create Margin
    df['Margin'] = df['TotalPremium'] - df['TotalClaims']

    print("\n---------------------------------------------------")
    print("      Statistical Hypothesis Testing Report        ")
    print("---------------------------------------------------")
    
    # 1. Province Risk Differences
    print("\n1. Testing: Risk Differences across Provinces")
    
    # Frequency
    res_prov_freq = test_risk_differences_categorical(df, 'Province', 'TotalClaims', 'frequency')
    print(f"   [Frequency] p-value: {res_prov_freq['P-Value']:.4e} | Significant: {res_prov_freq['Significant']}")
    if res_prov_freq['Significant']:
        print("   -> REJECT Null Hypothesis. There are significant differences in claim frequency across provinces.")
        
    # Severity
    res_prov_sev = test_risk_differences_categorical(df, 'Province', 'TotalClaims', 'severity')
    print(f"   [Severity]  p-value: {res_prov_sev['P-Value']:.4e} | Significant: {res_prov_sev['Significant']}")
    if res_prov_sev['Significant']:
        print("   -> REJECT Null Hypothesis. There are significant differences in claim severity across provinces.")

    # 2. ZipCode Risk Differences
    print("\n2. Testing: Risk Differences across ZipCodes")
    # ZipCodes are many, filtering to top 50 to ensure statistical validity and performance
    top_zips = df['PostalCode'].value_counts().nlargest(50).index
    df_zip = df[df['PostalCode'].isin(top_zips)].copy()
    
    # Frequency
    res_zip_freq = test_risk_differences_categorical(df_zip, 'PostalCode', 'TotalClaims', 'frequency')
    print(f"   [Frequency] p-value: {res_zip_freq['P-Value']:.4e} | Significant: {res_zip_freq['Significant']}")
    
    # Severity
    res_zip_sev = test_risk_differences_categorical(df_zip, 'PostalCode', 'TotalClaims', 'severity')
    print(f"   [Severity]  p-value: {res_zip_sev['P-Value']:.4e} | Significant: {res_zip_sev['Significant']}")

    # 3. ZipCode Margin Differences
    print("\n3. Testing: Margin Differences between ZipCodes")
    res_zip_margin = test_margin_differences(df_zip, 'PostalCode')
    print(f"   [Margin]    p-value: {res_zip_margin['P-Value']:.4e} | Significant: {res_zip_margin['Significant']}")

    # 4. Gender Risk Differences
    print("\n4. Testing: Risk Differences between Women and Men")
    # Filter for Gender (usually 'Gender' column, verify values)
    # Checking values first
    if 'Gender' in df.columns:
        # Assuming values are 'Male', 'Female' or similar. 
        # Need to clean or filter. Let's assume standard 'Male', 'Female' 
        # But data might have 'Not specified'. We filter for M/F.
        valid_genders = ['Male', 'Female'] # Adjust based on actual data
        # Actually, let's look at unique values dynamically
        print(f"   Gender values found: {df['Gender'].unique()}")
        
        # Filter for known genders only for the test
        df_gender = df[df['Gender'].isin(['Male', 'Female'])].copy()
        
        if not df_gender.empty:
            res_gen_freq = test_risk_differences_categorical(df_gender, 'Gender', 'TotalClaims', 'frequency')
            print(f"   [Frequency] p-value: {res_gen_freq['P-Value']:.4e} | Significant: {res_gen_freq['Significant']}")
            
            res_gen_sev = test_risk_differences_categorical(df_gender, 'Gender', 'TotalClaims', 'severity')
            print(f"   [Severity]  p-value: {res_gen_sev['P-Value']:.4e} | Significant: {res_gen_sev['Significant']}")
        else:
            print("   Insufficient Gender data for test.")

if __name__ == "__main__":
    run_tests()
