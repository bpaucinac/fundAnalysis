import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Load the data
df = pd.read_csv('fundRetuns.csv')

# Convert date to datetime format
df['tradeDate'] = pd.to_datetime(df['tradeDate'], format='%m/%d/%y')
df = df.sort_values('tradeDate')

# Calculate cumulative returns
df['PGTIX_cum'] = (1 + df['PGTIX']).cumprod() - 1
df['QQQ_cum'] = (1 + df['QQQ']).cumprod() - 1

# Calculate monthly excess returns
df['excess_return'] = df['PGTIX'] - df['QQQ']

# Calculate rolling metrics
window_sizes = [12, 24, 36, 60]  # 1-year, 2-year, 3-year, and 5-year windows

results = {}
for window in window_sizes:
    # Calculate rolling returns
    df[f'PGTIX_rolling_{window}m'] = df['PGTIX'].rolling(window).apply(lambda x: np.prod(1 + x) - 1)
    df[f'QQQ_rolling_{window}m'] = df['QQQ'].rolling(window).apply(lambda x: np.prod(1 + x) - 1)
    df[f'excess_return_{window}m'] = df[f'PGTIX_rolling_{window}m'] - df[f'QQQ_rolling_{window}m']
    
    # Calculate rolling alpha
    df[f'rolling_beta_{window}m'] = df['PGTIX'].rolling(window).cov(df['QQQ']) / df['QQQ'].rolling(window).var()
    risk_free_rate = 0.02 / 12  # Assuming 2% annual risk-free rate
    df[f'rolling_alpha_{window}m'] = (
        df['PGTIX'].rolling(window).mean() - 
        (risk_free_rate + df[f'rolling_beta_{window}m'] * (df['QQQ'].rolling(window).mean() - risk_free_rate))
    )
    
    # Calculate rolling outperformance percentage
    df[f'outperform_{window}m'] = df['excess_return'].rolling(window).apply(lambda x: (x > 0).mean())
    
    # Calculate rolling information ratio
    df[f'tracking_error_{window}m'] = df['excess_return'].rolling(window).std()
    df[f'info_ratio_{window}m'] = df['excess_return'].rolling(window).mean() / df[f'tracking_error_{window}m']

    # Store results for each window
    results[window] = {
        'avg_excess': df[f'excess_return_{window}m'].mean(),
        'pct_positive_excess': (df[f'excess_return_{window}m'] > 0).mean() * 100,
        'avg_alpha': df[f'rolling_alpha_{window}m'].mean() * 12,  # Annualized
        'avg_info_ratio': df[f'info_ratio_{window}m'].mean(),
        'pct_time_outperforming': (df[f'outperform_{window}m'] > 0.5).mean() * 100
    }

# Create time period splits (early, middle, late)
total_periods = len(df)
period_size = total_periods // 3

early_period = df.iloc[:period_size]
middle_period = df.iloc[period_size:2*period_size]
late_period = df.iloc[2*period_size:]

periods = {
    'Early': early_period,
    'Middle': middle_period,
    'Late': late_period,
    'Full': df
}

period_results = {}
for period_name, period_data in periods.items():
    # Calculate key metrics for each period
    period_results[period_name] = {
        'date_range': f"{period_data['tradeDate'].min().strftime('%m/%Y')} - {period_data['tradeDate'].max().strftime('%m/%Y')}",
        'pgtix_return': np.prod(1 + period_data['PGTIX']) - 1,
        'qqq_return': np.prod(1 + period_data['QQQ']) - 1,
        'excess_return': np.prod(1 + period_data['PGTIX']) - np.prod(1 + period_data['QQQ']),
        'pgtix_annualized': (1 + np.prod(1 + period_data['PGTIX']) - 1) ** (12 / len(period_data)) - 1,
        'qqq_annualized': (1 + np.prod(1 + period_data['QQQ']) - 1) ** (12 / len(period_data)) - 1,
        'outperformance_pct': (period_data['excess_return'] > 0).mean() * 100,
        'beta': period_data['PGTIX'].cov(period_data['QQQ']) / period_data['QQQ'].var(),
        'correlation': period_data['PGTIX'].corr(period_data['QQQ']),
        'pgtix_std': period_data['PGTIX'].std() * np.sqrt(12),  # Annualized
        'qqq_std': period_data['QQQ'].std() * np.sqrt(12),  # Annualized
    }
    
    # Calculate alpha
    period_beta = period_results[period_name]['beta']
    period_rf = 0.02 / 12  # Assuming 2% annual risk-free rate
    period_results[period_name]['alpha_annualized'] = (
        period_data['PGTIX'].mean() * 12 - 
        (period_rf * 12 + period_beta * (period_data['QQQ'].mean() * 12 - period_rf * 12))
    )

# Create directory for plots if it doesn't exist
import os
if not os.path.exists('plots'):
    os.makedirs('plots')

# Plot rolling excess returns for different windows
plt.figure(figsize=(15, 10))
for window in window_sizes:
    plt.plot(df['tradeDate'], df[f'excess_return_{window}m'], label=f'{window}-Month Rolling')
plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
plt.title('Rolling Excess Returns: PGTIX vs QQQ')
plt.xlabel('Date')
plt.ylabel('Excess Return')
plt.legend()
plt.grid(True)
plt.savefig('plots/rolling_excess_returns_comparison.png')
plt.close()

# Plot rolling information ratios
plt.figure(figsize=(15, 10))
for window in window_sizes:
    plt.plot(df['tradeDate'], df[f'info_ratio_{window}m'], label=f'{window}-Month Rolling')
plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
plt.title('Rolling Information Ratio: PGTIX vs QQQ')
plt.xlabel('Date')
plt.ylabel('Information Ratio')
plt.legend()
plt.grid(True)
plt.savefig('plots/rolling_information_ratio.png')
plt.close()

# Plot rolling alpha
plt.figure(figsize=(15, 10))
for window in window_sizes:
    plt.plot(df['tradeDate'], df[f'rolling_alpha_{window}m'] * 12, label=f'{window}-Month Rolling')  # Annualized
plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
plt.title('Rolling Annualized Alpha: PGTIX')
plt.xlabel('Date')
plt.ylabel('Annualized Alpha')
plt.legend()
plt.grid(True)
plt.savefig('plots/rolling_alpha.png')
plt.close()

# Plot rolling outperformance percentage
plt.figure(figsize=(15, 10))
for window in window_sizes:
    plt.plot(df['tradeDate'], df[f'outperform_{window}m'] * 100, label=f'{window}-Month Rolling')
plt.axhline(y=50, color='r', linestyle='-', alpha=0.3)
plt.title('Rolling Percentage of Months Outperforming: PGTIX vs QQQ')
plt.xlabel('Date')
plt.ylabel('Percentage (%)')
plt.legend()
plt.grid(True)
plt.savefig('plots/rolling_outperformance_percentage.png')
plt.close()

# Create a report for time period analysis
with open('time_period_analysis.md', 'w') as f:
    f.write("# Time Period Analysis: PGTIX vs QQQ\n\n")
    
    f.write("## Performance by Time Period\n\n")
    
    # Create a table for period results
    f.write("| Period | Date Range | PGTIX Return | QQQ Return | Excess Return | PGTIX Annualized | QQQ Annualized | Outperformance Months | Beta | Correlation | PGTIX Std Dev | QQQ Std Dev | Alpha Annualized |\n")
    f.write("|--------|------------|--------------|------------|---------------|------------------|----------------|----------------------|------|-------------|---------------|------------|------------------|\n")
    
    for period, result in period_results.items():
        f.write(f"| {period} | {result['date_range']} | {result['pgtix_return']:.2%} | {result['qqq_return']:.2%} | {result['excess_return']:.2%} | {result['pgtix_annualized']:.2%} | {result['qqq_annualized']:.2%} | {result['outperformance_pct']:.1f}% | {result['beta']:.2f} | {result['correlation']:.2f} | {result['pgtix_std']:.2%} | {result['qqq_std']:.2%} | {result['alpha_annualized']:.2%} |\n")
    
    f.write("\n## Rolling Window Analysis\n\n")
    
    # Create a table for rolling window results
    f.write("| Window | Avg Excess Return | % Positive Excess | Avg Annualized Alpha | Avg Info Ratio | % Time Outperforming |\n")
    f.write("|--------|-------------------|-------------------|----------------------|----------------|----------------------|\n")
    
    for window, result in results.items():
        f.write(f"| {window}-Month | {result['avg_excess']:.2%} | {result['pct_positive_excess']:.1f}% | {result['avg_alpha']:.2%} | {result['avg_info_ratio']:.2f} | {result['pct_time_outperforming']:.1f}% |\n")
    
    f.write("\n## Key Observations\n\n")
    
    # Early vs Late Period
    early_outperform = period_results['Early']['pgtix_return'] > period_results['Early']['qqq_return']
    late_outperform = period_results['Late']['pgtix_return'] > period_results['Late']['qqq_return']
    
    if early_outperform == late_outperform:
        f.write("- Performance consistency: PGTIX has been consistently ")
        f.write("outperforming" if early_outperform else "underperforming")
        f.write(" QQQ across early and late periods.\n")
    else:
        f.write("- Performance shift: PGTIX has ")
        if early_outperform:
            f.write("outperformed QQQ in the early period but underperformed in the late period.\n")
        else:
            f.write("underperformed QQQ in the early period but outperformed in the late period.\n")
    
    # Correlation trend
    corr_trend = period_results['Late']['correlation'] - period_results['Early']['correlation']
    if abs(corr_trend) > 0.05:
        f.write(f"- Correlation trend: The correlation between PGTIX and QQQ has ")
        if corr_trend > 0:
            f.write(f"increased from {period_results['Early']['correlation']:.2f} to {period_results['Late']['correlation']:.2f} over time.\n")
        else:
            f.write(f"decreased from {period_results['Early']['correlation']:.2f} to {period_results['Late']['correlation']:.2f} over time.\n")
    else:
        f.write(f"- Correlation stability: The correlation between PGTIX and QQQ has remained relatively stable over time.\n")
    
    # Alpha trend
    alpha_trend = period_results['Late']['alpha_annualized'] - period_results['Early']['alpha_annualized']
    if abs(alpha_trend) > 0.01:
        f.write(f"- Alpha trend: PGTIX's alpha has ")
        if alpha_trend > 0:
            f.write(f"improved from {period_results['Early']['alpha_annualized']:.2%} to {period_results['Late']['alpha_annualized']:.2%} annually.\n")
        else:
            f.write(f"deteriorated from {period_results['Early']['alpha_annualized']:.2%} to {period_results['Late']['alpha_annualized']:.2%} annually.\n")
    else:
        f.write(f"- Alpha stability: PGTIX's alpha has remained relatively stable over time.\n")
    
    # Rolling window analysis
    consistent_underperf = all(result['avg_excess'] < 0 for result in results.values())
    if consistent_underperf:
        f.write("- Consistent underperformance: PGTIX has shown negative excess returns across all rolling periods analyzed.\n")
    else:
        timeframes_with_outperf = [window for window, result in results.items() if result['avg_excess'] > 0]
        if timeframes_with_outperf:
            f.write(f"- Mixed performance: PGTIX has shown positive excess returns in the {', '.join(f'{w}-month' for w in timeframes_with_outperf)} rolling periods.\n")
        else:
            f.write("- General underperformance: PGTIX has shown negative excess returns across most rolling periods analyzed.\n")
    
    # Overall conclusion about consistency over time
    has_alpha_shifts = abs(period_results['Late']['alpha_annualized'] - period_results['Early']['alpha_annualized']) > 0.02
    has_outperf_shifts = abs(period_results['Late']['outperformance_pct'] - period_results['Early']['outperformance_pct']) > 10
    
    f.write("\n## Conclusion: Does the Analysis Hold Over Time?\n\n")
    
    if has_alpha_shifts or has_outperf_shifts:
        f.write("The performance characteristics of PGTIX relative to QQQ have shown **meaningful changes** over time. ")
        f.write("This suggests that the conclusions from the full-period analysis may not hold consistently across all sub-periods. ")
        f.write("Investors should be aware that the relative performance has varied over time ")
        f.write("and may continue to do so in the future.\n")
    else:
        f.write("The performance characteristics of PGTIX relative to QQQ have remained **relatively consistent** over time. ")
        f.write("This suggests that the conclusions from the full-period analysis generally hold across different sub-periods. ")
        f.write("The persistent nature of the performance pattern increases confidence in the overall assessment ")
        f.write("that the observed relationship between PGTIX and QQQ is structural rather than coincidental.\n")

print("Time period analysis complete! Check the 'time_period_analysis.md' file and the new plots in the 'plots' directory.") 