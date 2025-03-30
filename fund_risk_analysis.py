import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import datetime
import os

# Create directory for risk plots if it doesn't exist
if not os.path.exists('risk_plots'):
    os.makedirs('risk_plots')

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

# Calculate risk metrics
# 1. Rolling standard deviation (volatility) - 12-month window
df['PGTIX_rolling_vol'] = df['PGTIX'].rolling(12).std() * np.sqrt(12)  # Annualized
df['QQQ_rolling_vol'] = df['QQQ'].rolling(12).std() * np.sqrt(12)  # Annualized
df['relative_vol'] = df['PGTIX_rolling_vol'] / df['QQQ_rolling_vol']

# 2. Rolling beta - 24-month window
df['rolling_beta_24m'] = df['PGTIX'].rolling(24).cov(df['QQQ']) / df['QQQ'].rolling(24).var()

# 3. Rolling downside deviation - 12-month window
def downside_deviation(returns, threshold=0):
    downside_returns = returns.copy()
    downside_returns[returns > threshold] = 0
    return np.sqrt(np.sum(downside_returns**2) / len(returns)) * np.sqrt(12)  # Annualized

df['PGTIX_downside_dev'] = df['PGTIX'].rolling(12).apply(downside_deviation, raw=True)
df['QQQ_downside_dev'] = df['QQQ'].rolling(12).apply(downside_deviation, raw=True)
df['relative_downside_dev'] = df['PGTIX_downside_dev'] / df['QQQ_downside_dev']

# 4. Maximum drawdown (rolling 12-month)
def max_drawdown(returns):
    cum_returns = (1 + returns).cumprod()
    running_max = np.maximum.accumulate(cum_returns)
    drawdown = (cum_returns / running_max) - 1
    return drawdown.min()

df['PGTIX_max_drawdown'] = df['PGTIX'].rolling(12).apply(max_drawdown, raw=True)
df['QQQ_max_drawdown'] = df['QQQ'].rolling(12).apply(max_drawdown, raw=True)
df['relative_max_drawdown'] = df['PGTIX_max_drawdown'] / df['QQQ_max_drawdown']

# 5. Value at Risk (VaR) - 95% confidence, 1-month
def var_95(returns):
    return np.percentile(returns, 5)

df['PGTIX_VaR_95'] = df['PGTIX'].rolling(24).apply(var_95, raw=True)
df['QQQ_VaR_95'] = df['QQQ'].rolling(24).apply(var_95, raw=True)
df['relative_VaR'] = df['PGTIX_VaR_95'] / df['QQQ_VaR_95']

# 6. Rolling Sharpe ratio - 24-month window
risk_free_rate = 0.02 / 12  # Assuming 2% annual risk-free rate
df['PGTIX_sharpe'] = (df['PGTIX'].rolling(24).mean() - risk_free_rate) / df['PGTIX'].rolling(24).std() * np.sqrt(12)
df['QQQ_sharpe'] = (df['QQQ'].rolling(24).mean() - risk_free_rate) / df['QQQ'].rolling(24).std() * np.sqrt(12)
df['relative_sharpe'] = df['PGTIX_sharpe'] / df['QQQ_sharpe']

# 7. Rolling Sortino ratio - 24-month window
df['PGTIX_sortino'] = (df['PGTIX'].rolling(24).mean() - risk_free_rate) / df['PGTIX'].rolling(24).apply(downside_deviation, raw=True)
df['QQQ_sortino'] = (df['QQQ'].rolling(24).mean() - risk_free_rate) / df['QQQ'].rolling(24).apply(downside_deviation, raw=True)
df['relative_sortino'] = df['PGTIX_sortino'] / df['QQQ_sortino']

# 8. Conditional value at risk (CVaR) / Expected Shortfall - 95% confidence
def cvar_95(returns):
    var_95 = np.percentile(returns, 5)
    return returns[returns <= var_95].mean()

df['PGTIX_CVaR_95'] = df['PGTIX'].rolling(24).apply(cvar_95, raw=True)
df['QQQ_CVaR_95'] = df['QQQ'].rolling(24).apply(cvar_95, raw=True)
df['relative_CVaR'] = df['PGTIX_CVaR_95'] / df['QQQ_CVaR_95']

# Create time period splits for analysis
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

# Risk metrics by period
risk_by_period = {}
for period_name, period_data in periods.items():
    risk_by_period[period_name] = {
        'date_range': f"{period_data['tradeDate'].min().strftime('%m/%Y')} - {period_data['tradeDate'].max().strftime('%m/%Y')}",
        'PGTIX_vol': period_data['PGTIX'].std() * np.sqrt(12),
        'QQQ_vol': period_data['QQQ'].std() * np.sqrt(12),
        'relative_vol': (period_data['PGTIX'].std() / period_data['QQQ'].std()),
        'PGTIX_downside_dev': downside_deviation(period_data['PGTIX']),
        'QQQ_downside_dev': downside_deviation(period_data['QQQ']),
        'relative_downside': (downside_deviation(period_data['PGTIX']) / downside_deviation(period_data['QQQ'])),
        'PGTIX_VaR_95': np.percentile(period_data['PGTIX'], 5),
        'QQQ_VaR_95': np.percentile(period_data['QQQ'], 5),
        'relative_VaR': (np.percentile(period_data['PGTIX'], 5) / np.percentile(period_data['QQQ'], 5)),
        'beta': period_data['PGTIX'].cov(period_data['QQQ']) / period_data['QQQ'].var(),
        'PGTIX_max_dd': max_drawdown(period_data['PGTIX']),
        'QQQ_max_dd': max_drawdown(period_data['QQQ']),
        'relative_max_dd': (max_drawdown(period_data['PGTIX']) / max_drawdown(period_data['QQQ'])),
        'PGTIX_sharpe': (period_data['PGTIX'].mean() - risk_free_rate) / period_data['PGTIX'].std() * np.sqrt(12),
        'QQQ_sharpe': (period_data['QQQ'].mean() - risk_free_rate) / period_data['QQQ'].std() * np.sqrt(12),
        'relative_sharpe': ((period_data['PGTIX'].mean() - risk_free_rate) / period_data['PGTIX'].std()) / 
                           ((period_data['QQQ'].mean() - risk_free_rate) / period_data['QQQ'].std()),
    }

# Analyze risk in different market environments
# Define up/down markets based on QQQ performance
df['market_environment'] = 'neutral'
up_threshold = df['QQQ'].quantile(0.7)  # Top 30% returns define up markets
down_threshold = df['QQQ'].quantile(0.3)  # Bottom 30% returns define down markets
df.loc[df['QQQ'] > up_threshold, 'market_environment'] = 'up'
df.loc[df['QQQ'] < down_threshold, 'market_environment'] = 'down'

# Calculate risk in different market environments
risk_by_environment = {}
for env in ['up', 'neutral', 'down']:
    env_data = df[df['market_environment'] == env]
    if len(env_data) > 0:
        risk_by_environment[env] = {
            'count': len(env_data),
            'PGTIX_vol': env_data['PGTIX'].std() * np.sqrt(12),
            'QQQ_vol': env_data['QQQ'].std() * np.sqrt(12),
            'relative_vol': (env_data['PGTIX'].std() / env_data['QQQ'].std()),
            'beta': env_data['PGTIX'].cov(env_data['QQQ']) / env_data['QQQ'].var() if env_data['QQQ'].var() != 0 else np.nan,
            'PGTIX_mean': env_data['PGTIX'].mean() * 12,  # Annualized
            'QQQ_mean': env_data['QQQ'].mean() * 12,  # Annualized
            'excess_return': (env_data['PGTIX'].mean() - env_data['QQQ'].mean()) * 12,  # Annualized
        }

# Analyze risk-taking and subsequent performance
# Identify periods of high and low relative risk
df['high_relative_risk'] = df['relative_vol'] > df['relative_vol'].quantile(0.7)
df['low_relative_risk'] = df['relative_vol'] < df['relative_vol'].quantile(0.3)

# Calculate subsequent performance after high/low risk periods
subsequent_performance = {}
for risk_type in ['high_relative_risk', 'low_relative_risk']:
    # Get indices where the risk condition is met
    risk_indices = df[df[risk_type]].index
    
    subsequent_performance[risk_type] = {}
    
    for window in [1, 3, 6, 12]:  # Months ahead
        excess_returns = []
        
        for idx in risk_indices:
            if idx + window < len(df):
                # Calculate cumulative excess return over the subsequent period
                start_idx = df.index.get_loc(idx) + 1
                end_idx = min(start_idx + window, len(df))
                
                if end_idx > start_idx:
                    subsequent_excess = (1 + df['PGTIX'].iloc[start_idx:end_idx]).prod() - (1 + df['QQQ'].iloc[start_idx:end_idx]).prod()
                    excess_returns.append(subsequent_excess)
        
        if excess_returns:
            subsequent_performance[risk_type][f'{window}m_ahead'] = {
                'mean_excess': np.mean(excess_returns),
                'median_excess': np.median(excess_returns),
                'positive_pct': (np.array(excess_returns) > 0).mean() * 100,
                'count': len(excess_returns)
            }

# Detect drastic changes in relative risk
df['rel_vol_pct_change'] = df['relative_vol'].pct_change()
drastic_risk_changes = df[abs(df['rel_vol_pct_change']) > 0.1].copy()
drastic_risk_changes['direction'] = np.where(drastic_risk_changes['rel_vol_pct_change'] > 0, 'increase', 'decrease')
drastic_change_freq = len(drastic_risk_changes) / len(df[~df['relative_vol'].isna()])

# Skill assessment: Risk-adjusted performance trends
df['rolling_info_ratio'] = df['excess_return'].rolling(24).mean() / df['excess_return'].rolling(24).std()

# Visualizations
# 1. Risk over time
plt.figure(figsize=(12, 8))
plt.plot(df['tradeDate'], df['PGTIX_rolling_vol'], label='PGTIX Volatility (12m)')
plt.plot(df['tradeDate'], df['QQQ_rolling_vol'], label='QQQ Volatility (12m)')
plt.title('Rolling 12-Month Annualized Volatility')
plt.xlabel('Date')
plt.ylabel('Annualized Volatility')
plt.legend()
plt.grid(True)
plt.savefig('risk_plots/rolling_volatility.png')
plt.close()

# 2. Relative risk over time
plt.figure(figsize=(12, 8))
plt.plot(df['tradeDate'], df['relative_vol'], label='Relative Volatility (PGTIX/QQQ)')
plt.axhline(y=1, color='r', linestyle='--', alpha=0.5)
plt.title('Relative Volatility (PGTIX/QQQ) - 12-Month Rolling')
plt.xlabel('Date')
plt.ylabel('Ratio')
plt.legend()
plt.grid(True)
plt.savefig('risk_plots/relative_volatility.png')
plt.close()

# 3. Beta over time
plt.figure(figsize=(12, 8))
plt.plot(df['tradeDate'], df['rolling_beta_24m'], label='PGTIX Beta to QQQ (24m)')
plt.axhline(y=1, color='r', linestyle='--', alpha=0.5)
plt.title('Rolling 24-Month Beta (PGTIX to QQQ)')
plt.xlabel('Date')
plt.ylabel('Beta')
plt.legend()
plt.grid(True)
plt.savefig('risk_plots/rolling_beta.png')
plt.close()

# 4. Downside risk comparison
plt.figure(figsize=(12, 8))
plt.plot(df['tradeDate'], df['PGTIX_downside_dev'], label='PGTIX Downside Deviation')
plt.plot(df['tradeDate'], df['QQQ_downside_dev'], label='QQQ Downside Deviation')
plt.title('Rolling 12-Month Downside Deviation')
plt.xlabel('Date')
plt.ylabel('Downside Deviation')
plt.legend()
plt.grid(True)
plt.savefig('risk_plots/downside_deviation.png')
plt.close()

# 5. Value at Risk comparison
plt.figure(figsize=(12, 8))
plt.plot(df['tradeDate'], df['PGTIX_VaR_95'], label='PGTIX 95% VaR')
plt.plot(df['tradeDate'], df['QQQ_VaR_95'], label='QQQ 95% VaR')
plt.title('Rolling 24-Month 95% Value at Risk')
plt.xlabel('Date')
plt.ylabel('Monthly VaR (95%)')
plt.legend()
plt.grid(True)
plt.savefig('risk_plots/value_at_risk.png')
plt.close()

# 6. Sharpe ratio comparison
plt.figure(figsize=(12, 8))
plt.plot(df['tradeDate'], df['PGTIX_sharpe'], label='PGTIX Sharpe Ratio')
plt.plot(df['tradeDate'], df['QQQ_sharpe'], label='QQQ Sharpe Ratio')
plt.title('Rolling 24-Month Sharpe Ratio')
plt.xlabel('Date')
plt.ylabel('Sharpe Ratio')
plt.legend()
plt.grid(True)
plt.savefig('risk_plots/sharpe_ratio.png')
plt.close()

# 7. Risk vs Return by period
periods_for_plot = ['Early', 'Middle', 'Late', 'Full']
risk_values = [risk_by_period[period]['PGTIX_vol'] for period in periods_for_plot]
return_values = [(1 + periods[period]['PGTIX'].mean())**12 - 1 for period in periods_for_plot]

plt.figure(figsize=(10, 8))
plt.scatter(risk_values, return_values)
for i, period in enumerate(periods_for_plot):
    plt.annotate(period, (risk_values[i], return_values[i]), 
                 fontsize=12, ha='center')
plt.title('Risk vs Return by Time Period - PGTIX')
plt.xlabel('Annualized Volatility')
plt.ylabel('Annualized Return')
plt.grid(True)
plt.savefig('risk_plots/risk_vs_return_by_period.png')
plt.close()

# 8. Information ratio (skill metric)
plt.figure(figsize=(12, 8))
plt.plot(df['tradeDate'], df['rolling_info_ratio'], label='24-Month Information Ratio')
plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
plt.title('Rolling 24-Month Information Ratio (PGTIX vs QQQ)')
plt.xlabel('Date')
plt.ylabel('Information Ratio')
plt.legend()
plt.grid(True)
plt.savefig('risk_plots/information_ratio.png')
plt.close()

# 9. Risk changes over time
plt.figure(figsize=(12, 8))
plt.plot(df['tradeDate'], df['rel_vol_pct_change'], label='Relative Vol % Change')
plt.axhline(y=0.1, color='r', linestyle='--', alpha=0.5, label='Drastic Change Threshold (+10%)')
plt.axhline(y=-0.1, color='r', linestyle='--', alpha=0.5, label='Drastic Change Threshold (-10%)')
plt.title('Monthly Changes in Relative Risk (PGTIX/QQQ)')
plt.xlabel('Date')
plt.ylabel('Percentage Change')
plt.legend()
plt.grid(True)
plt.savefig('risk_plots/risk_changes.png')
plt.close()

# 10. Histogram of relative volatility
plt.figure(figsize=(12, 8))
plt.hist(df['relative_vol'].dropna(), bins=20, alpha=0.7)
plt.axvline(x=1, color='r', linestyle='--', alpha=0.7, label='Equal Volatility')
plt.axvline(x=df['relative_vol'].mean(), color='g', linestyle='-', alpha=0.7, 
            label=f'Mean Relative Vol: {df["relative_vol"].mean():.2f}')
plt.title('Distribution of Relative Volatility (PGTIX/QQQ)')
plt.xlabel('Relative Volatility')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
plt.savefig('risk_plots/relative_vol_distribution.png')
plt.close()

# Save risk data for the PDF report
risk_data = {
    'risk_by_period': risk_by_period,
    'risk_by_environment': risk_by_environment,
    'subsequent_performance': subsequent_performance,
    'drastic_change_freq': drastic_change_freq,
    'current_relative_vol': df['relative_vol'].iloc[-1],
    'historical_relative_vol': {
        'mean': df['relative_vol'].mean(),
        'median': df['relative_vol'].median(),
        'std': df['relative_vol'].std(),
        'min': df['relative_vol'].min(),
        'max': df['relative_vol'].max(),
        'current_percentile': stats.percentileofscore(df['relative_vol'].dropna(), df['relative_vol'].iloc[-1])
    }
}

# Output risk data as text file for reference
with open('risk_analysis_data.txt', 'w') as f:
    f.write(f"Risk Analysis Data for PGTIX vs QQQ\n")
    f.write(f"Analysis Date: {datetime.now().strftime('%B %d, %Y')}\n\n")
    
    f.write("== Risk by Period ==\n")
    for period, metrics in risk_by_period.items():
        f.write(f"{period} Period ({metrics['date_range']}):\n")
        for key, value in metrics.items():
            if key != 'date_range':
                f.write(f"  {key}: {value:.4f}\n")
        f.write("\n")
    
    f.write("== Risk by Market Environment ==\n")
    for env, metrics in risk_by_environment.items():
        f.write(f"{env.capitalize()} Market Environment ({metrics['count']} months):\n")
        for key, value in metrics.items():
            if key != 'count':
                f.write(f"  {key}: {value:.4f}\n")
        f.write("\n")
    
    f.write("== Subsequent Performance After Risk Changes ==\n")
    for risk_type, windows in subsequent_performance.items():
        f.write(f"After {risk_type.replace('_', ' ')}:\n")
        for window, perf in windows.items():
            f.write(f"  {window}:\n")
            for key, value in perf.items():
                f.write(f"    {key}: {value:.4f}\n")
        f.write("\n")
    
    f.write("== Risk Change Frequency ==\n")
    f.write(f"Frequency of drastic risk changes: {drastic_change_freq:.2%}\n\n")
    
    f.write("== Current vs Historical Risk ==\n")
    f.write(f"Current relative volatility: {df['relative_vol'].iloc[-1]:.4f}\n")
    f.write(f"Historical mean: {df['relative_vol'].mean():.4f}\n")
    f.write(f"Historical median: {df['relative_vol'].median():.4f}\n")
    f.write(f"Historical standard deviation: {df['relative_vol'].std():.4f}\n")
    f.write(f"Historical min: {df['relative_vol'].min():.4f}\n")
    f.write(f"Historical max: {df['relative_vol'].max():.4f}\n")
    f.write(f"Current percentile: {stats.percentileofscore(df['relative_vol'].dropna(), df['relative_vol'].iloc[-1]):.1f}%\n")

print("Risk analysis complete! Check the 'risk_plots' directory for visualizations and risk_analysis_data.txt for detailed risk metrics.") 