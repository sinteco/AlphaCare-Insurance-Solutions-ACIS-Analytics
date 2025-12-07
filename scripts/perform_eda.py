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
    filepath = 'data/raw/MachineLearningRating_v3.txt'
    df = load_data(filepath)
    
    if df is not None:
        print(f"Data Loaded Successfully. Shape: {df.shape}")
        
        # --- Data Cleaning & formatting ---
        # Convert TransactionMonth to datetime
        if 'TransactionMonth' in df.columns:
            df['TransactionMonth'] = pd.to_datetime(df['TransactionMonth'], errors='coerce')
            print("Converted TransactionMonth to datetime.")

        # Ensure numeric columns are numeric (sometimes cleaning is needed)
        numeric_cols = ['TotalPremium', 'TotalClaims']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # --- 1. Data Summarization & Structure ---
        print("\n--- Data Info & Dtypes ---")
        print(df.info())
        
        print("\n--- Descriptive Statistics (Numerical) ---")
        print(df[numeric_cols].describe())
        print("\n--- Variability (Standard Deviation) ---")
        print(df[numeric_cols].std())

        # --- 2. Data Quality (Missing Values) ---
        print("\n--- Missing Values ---")
        missing = check_missing_values(df)
        print(missing[missing > 0])
        
        # Ensure output directory exists
        os.makedirs('reports/figures', exist_ok=True)
        # Set Grid Style
        sns.set_theme(style="whitegrid")

        # --- 3. Outlier Detection (Box Plots) ---
        print("\nGenerating Box Plots for Outlier Detection...")
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        sns.boxplot(y=df['TotalPremium'], color='skyblue')
        plt.title('Box Plot: Total Premium')
        
        plt.subplot(1, 2, 2)
        sns.boxplot(y=df['TotalClaims'], color='salmon')
        plt.title('Box Plot: Total Claims')
        
        plt.tight_layout()
        plt.savefig('reports/figures/outliers_boxplot.png')
        print("Saved outliers_boxplot.png")

        # --- 4. Bivariate Analysis & Correlation ---
        print("\nGenerating Correlation Matrix...")
        # Select numeric columns for correlation
        # We can add more like 'CalculatedPremiumPerTerm' if available and numeric
        potential_numeric = ['TotalPremium', 'TotalClaims', 'CalculatedPremiumPerTerm', 'SumInsured']
        valid_numeric = [c for c in potential_numeric if c in df.columns]
        
        corr_matrix = df[valid_numeric].corr()
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Correlation Matrix')
        plt.savefig('reports/figures/correlation_matrix.png')
        print("Saved correlation_matrix.png")

        # --- 5. Creative Plot 1: Trends Over Time (Monthly changes) ---
        print("\nGenerating Time Series Plot...")
        if 'TransactionMonth' in df.columns:
            monthly_trends = df.groupby('TransactionMonth')[['TotalPremium', 'TotalClaims']].sum().reset_index()
            
            plt.figure(figsize=(14, 7))
            sns.lineplot(x='TransactionMonth', y='TotalPremium', data=monthly_trends, label='Total Premium', linewidth=2.5)
            sns.lineplot(x='TransactionMonth', y='TotalClaims', data=monthly_trends, label='Total Claims', linewidth=2.5, color='red')
            plt.title('Monthly Trends: Total Premium vs Total Claims', fontsize=16)
            plt.xlabel('Month')
            plt.ylabel('Amount (ZAR)')
            plt.legend()
            plt.savefig('reports/figures/monthly_trends.png')
            print("Saved monthly_trends.png")

        # --- 6. Creative Plot 2: Claims by Province (Violin Plot for distribution + density) ---
        print("\nGenerating Distribution by Province Plot...")
        if 'Province' in df.columns:
            # Filter for provinces with significant data to avoid clutter
            top_provinces = df['Province'].value_counts().nlargest(5).index
            df_top_prov = df[df['Province'].isin(top_provinces)]
            
            plt.figure(figsize=(14, 8))
            # using log scale for better visualization of money distributions which are often skewed
            sns.violinplot(x='Province', y='TotalClaims', data=df_top_prov, palette='viridis')
            plt.title('Distribution of Total Claims by Top 5 Provinces', fontsize=16)
            plt.ylim(-1000, 50000) # Limit y-axis to focus on the main distribution body
            plt.savefig('reports/figures/claims_by_province_violin.png')
            print("Saved claims_by_province_violin.png")

        # --- 7. Creative Plot 3: Premium vs Claims Faceted by Vehicle Type (Multivariate) ---
        print("\nGenerating Faceted Scatter Plot...")
        if 'VehicleType' in df.columns:
            # Filter top 3 vehicle types
            top_vehicles = df['VehicleType'].value_counts().nlargest(3).index
            df_top_veh = df[df['VehicleType'].isin(top_vehicles)]
            
            # Sample data for scatter plot to avoid overplotting if too large
            if len(df_top_veh) > 10000:
                df_top_veh = df_top_veh.sample(10000, random_state=42)

            g = sns.FacetGrid(df_top_veh, col="VehicleType", height=5, hue="VehicleType")
            g.map(sns.scatterplot, "TotalPremium", "TotalClaims", alpha=0.6)
            g.add_legend()
            g.fig.suptitle('Total Premium vs Total Claims by Vehicle Type', y=1.05, fontsize=16)
            plt.savefig('reports/figures/premium_vs_claims_by_vehicle.png')
            print("Saved premium_vs_claims_by_vehicle.png")

        # --- 8. ZipCode Analysis (Monthly changes wrapper) ---
        print("\n--- ZipCode Analysis Context ---")
        print("Aggregating data by PostalCode (ZipCode) to see variability.")
        if 'PostalCode' in df.columns:
            zip_stats = df.groupby('PostalCode')[['TotalPremium', 'TotalClaims']].agg(['mean', 'sum', 'std'])
            print(zip_stats.head())

if __name__ == "__main__":
    main()
