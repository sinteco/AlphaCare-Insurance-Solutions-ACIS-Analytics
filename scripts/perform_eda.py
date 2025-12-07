import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.loader import load_data
from src.eda import check_missing_values, get_descriptive_stats

def main():
    print("Loading data...")
    # Using the detected path
    filepath = 'data/raw/MachineLearningRating_v3.txt'
    df = load_data(filepath)
    
    if df is not None:
        print(f"Data Loaded Successfully. Shape: {df.shape}")
        
        # 1. Data Structure & Quality
        print("\n--- Data Info ---")
        print(df.info())
        
        print("\n--- Missing Values ---")
        missing = check_missing_values(df)
        print(missing[missing > 0])
        
        # 2. Descriptive Stats
        print("\n--- Descriptive Statistics ---")
        print(get_descriptive_stats(df))
        
        # 3. Univariate Analysis (Plots)
        # Ensure output directory exists
        os.makedirs('reports/figures', exist_ok=True)
        
        # Total Claims Distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(df['TotalClaims'], bins=50, kde=True)
        plt.title('Distribution of Total Claims')
        plt.savefig('reports/figures/total_claims_dist.png')
        print("Saved total_claims_dist.png")
        
        # Total Premium Distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(df['TotalPremium'], bins=50, kde=True)
        plt.title('Distribution of Total Premium')
        plt.savefig('reports/figures/total_premium_dist.png')
        print("Saved total_premium_dist.png")
        
        # 4. Bivariate Analysis
        # Scatter plot Premium vs Claims
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='TotalPremium', y='TotalClaims', data=df)
        plt.title('Total Premium vs Total Claims')
        plt.savefig('reports/figures/premium_vs_claims.png')
        print("Saved premium_vs_claims.png")
        
        # 5. Categorical Analysis
        # Loss Ratio by Province (Approximate: Sum(Claims) / Sum(Premium))
        # Group by Province
        if 'Province' in df.columns:
            province_stats = df.groupby('Province')[['TotalPremium', 'TotalClaims']].sum()
            province_stats['LossRatio'] = province_stats['TotalClaims'] / province_stats['TotalPremium']
            
            plt.figure(figsize=(12, 6))
            province_stats['LossRatio'].sort_values().plot(kind='bar')
            plt.title('Loss Ratio by Province')
            plt.ylabel('Loss Ratio')
            plt.savefig('reports/figures/loss_ratio_by_province.png')
            print("Saved loss_ratio_by_province.png")

if __name__ == "__main__":
    main()
