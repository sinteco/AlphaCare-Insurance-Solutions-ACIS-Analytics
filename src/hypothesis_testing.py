import pandas as pd
from scipy import stats
import numpy as np

def test_risk_differences_categorical(df, group_col, target_col, risk_type='frequency'):
    """
    Tests for risk differences across categorical groups.
    
    Args:
        df: DataFrame
        group_col: Column defining groups (e.g., 'Province', 'Gender')
        target_col: Target column (e.g., 'TotalClaims')
        risk_type: 'frequency' (Binary Claim/NoClaim) or 'severity' (Claim Amount)
    
    Returns:
        dict: Test results (Test Name, Statistic, P-Value, Conclusion)
    """
    results = {'Feature': group_col, 'RiskType': risk_type}
    
    if risk_type == 'frequency':
        # Chi-Square Test
        # Create a binary column: HasClaim (1 if TotalClaims > 0 else 0)
        # Note: Claims can be negative technically in some accounting, but for frequency we usually look at >0 or !=0.
        # Given the data summary showed min claims < 0, we treat != 0 as a claim event or simple > 0.
        # Let's assume > 0 is a claim for client risk.
        df['HasClaim'] = df[target_col].apply(lambda x: 1 if x > 0 else 0)
        
        contingency_table = pd.crosstab(df[group_col], df['HasClaim'])
        chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
        
        results.update({
            'Test': 'Chi-Squared',
            'Statistic': chi2,
            'P-Value': p,
            'Significant': p < 0.05
        })
        
    elif risk_type == 'severity':
        # ANOVA (Analysis of Variance) for multiple groups, T-test for 2 groups
        # Filter for only where claims occurred
        claims_df = df[df[target_col] > 0]
        
        groups = [group[target_col].values for name, group in claims_df.groupby(group_col)]
        
        # Check if enough groups
        if len(groups) < 2:
            return None
            
        if len(groups) == 2:
            # T-test
            stat, p = stats.ttest_ind(groups[0], groups[1], equal_var=False)
            results['Test'] = 'T-Test (Welch)'
        else:
            # ANOVA
            stat, p = stats.f_oneway(*groups)
            results['Test'] = 'ANOVA'
            
        results.update({
            'Statistic': stat,
            'P-Value': p,
            'Significant': p < 0.05
        })
        
    return results

def test_margin_differences(df, group_col, margin_col='Margin'):
    """
    Tests for margin differences across groups.
    """
    # Create Margin if not exists takes time, best to do outside or here
    if margin_col not in df.columns:
        df[margin_col] = df['TotalPremium'] - df['TotalClaims']
        
    groups = [group[margin_col].values for name, group in df.groupby(group_col)]
    
    if len(groups) < 2:
        return {'Error': 'Not enough groups'}
        
    if len(groups) == 2:
        stat, p = stats.ttest_ind(groups[0], groups[1], equal_var=False)
        test_name = 'T-Test (Welch)'
    else:
        stat, p = stats.f_oneway(*groups)
        test_name = 'ANOVA'
        
    return {
        'Feature': group_col,
        'Metric': 'Margin',
        'Test': test_name,
        'Statistic': stat,
        'P-Value': p,
        'Significant': p < 0.05
    }
