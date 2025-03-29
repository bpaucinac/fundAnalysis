import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
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

# Calculate rolling 12-month performance
df['PGTIX_rolling_12m'] = df['PGTIX'].rolling(12).apply(lambda x: np.prod(1 + x) - 1)
df['QQQ_rolling_12m'] = df['QQQ'].rolling(12).apply(lambda x: np.prod(1 + x) - 1)
df['excess_return_12m'] = df['PGTIX_rolling_12m'] - df['QQQ_rolling_12m']

# Calculate monthly statistics
mean_pgtix = df['PGTIX'].mean()
mean_qqq = df['QQQ'].mean()
std_pgtix = df['PGTIX'].std()
std_qqq = df['QQQ'].std()
sharpe_pgtix = mean_pgtix / std_pgtix
sharpe_qqq = mean_qqq / std_qqq

# Calculate correlation
correlation = df['PGTIX'].corr(df['QQQ'])

# Calculate beta
beta = df['PGTIX'].cov(df['QQQ']) / df['QQQ'].var()

# Calculate alpha (Jensen's Alpha)
risk_free_rate = 0.02 / 12  # Assuming 2% annual risk-free rate
alpha = mean_pgtix - (risk_free_rate + beta * (mean_qqq - risk_free_rate))

# Count outperformance periods
outperform_count = (df['excess_return'] > 0).sum()
total_periods = len(df)
outperform_percentage = outperform_count / total_periods * 100

# Calculate longest streak of outperformance and underperformance
df['outperform'] = df['excess_return'] > 0

# Function to find longest streak
def longest_streak(series):
    curr_streak = 0
    max_streak = 0
    for val in series:
        if val:
            curr_streak += 1
            max_streak = max(max_streak, curr_streak)
        else:
            curr_streak = 0
    return max_streak

longest_outperform = longest_streak(df['outperform'])
longest_underperform = longest_streak(~df['outperform'])

# Calculate Information Ratio
tracking_error = df['excess_return'].std()
information_ratio = df['excess_return'].mean() / tracking_error

# Calculate up-market and down-market capture ratios
up_months = df[df['QQQ'] > 0]
down_months = df[df['QQQ'] < 0]
up_market_capture = np.mean(up_months['PGTIX']) / np.mean(up_months['QQQ']) if len(up_months) > 0 else 0
down_market_capture = np.mean(down_months['PGTIX']) / np.mean(down_months['QQQ']) if len(down_months) > 0 else 0

# Calculate win/loss ratio
wins = df[df['excess_return'] > 0]['excess_return'].sum()
losses = abs(df[df['excess_return'] < 0]['excess_return'].sum())
win_loss_ratio = wins / losses if losses > 0 else float('inf')

# Create directory for plots if it doesn't exist
import os
if not os.path.exists('plots'):
    os.makedirs('plots')

# 1. Plot cumulative returns
plt.figure(figsize=(12, 8))
plt.plot(df['tradeDate'], df['PGTIX_cum'], label='PGTIX')
plt.plot(df['tradeDate'], df['QQQ_cum'], label='QQQ')
plt.title('Cumulative Returns: PGTIX vs QQQ')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.legend()
plt.grid(True)
plt.savefig('plots/cumulative_returns.png')
plt.close()

# 2. Plot rolling 12-month returns
plt.figure(figsize=(12, 8))
plt.plot(df['tradeDate'], df['PGTIX_rolling_12m'], label='PGTIX 12m')
plt.plot(df['tradeDate'], df['QQQ_rolling_12m'], label='QQQ 12m')
plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
plt.title('Rolling 12-Month Returns: PGTIX vs QQQ')
plt.xlabel('Date')
plt.ylabel('12-Month Return')
plt.legend()
plt.grid(True)
plt.savefig('plots/rolling_12m_returns.png')
plt.close()

# 3. Plot rolling 12-month excess returns
plt.figure(figsize=(12, 8))
plt.plot(df['tradeDate'], df['excess_return_12m'], label='Excess Return 12m')
plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
plt.title('Rolling 12-Month Excess Returns: PGTIX - QQQ')
plt.xlabel('Date')
plt.ylabel('Excess Return')
plt.legend()
plt.grid(True)
plt.savefig('plots/rolling_12m_excess.png')
plt.close()

# 4. Plot monthly excess returns
plt.figure(figsize=(12, 8))
plt.bar(df['tradeDate'], df['excess_return'])
plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
plt.title('Monthly Excess Returns: PGTIX - QQQ')
plt.xlabel('Date')
plt.ylabel('Excess Return')
plt.grid(True)
plt.savefig('plots/monthly_excess_returns.png')
plt.close()

# 5. Create a scatter plot of PGTIX vs QQQ returns
plt.figure(figsize=(10, 10))
plt.scatter(df['QQQ'], df['PGTIX'], alpha=0.5)
plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='r', linestyle='-', alpha=0.3)

# Add regression line
slope, intercept, r_value, p_value, std_err = stats.linregress(df['QQQ'], df['PGTIX'])
x = np.linspace(df['QQQ'].min(), df['QQQ'].max(), 100)
plt.plot(x, intercept + slope * x, 'r', label=f'y = {slope:.2f}x + {intercept:.4f}')

plt.title('PGTIX Returns vs QQQ Returns')
plt.xlabel('QQQ Returns')
plt.ylabel('PGTIX Returns')
plt.legend()
plt.grid(True)
plt.savefig('plots/returns_scatter.png')
plt.close()

# 6. Distribution of returns
plt.figure(figsize=(12, 8))
sns.histplot(df['PGTIX'], kde=True, label='PGTIX', alpha=0.5)
sns.histplot(df['QQQ'], kde=True, label='QQQ', alpha=0.5)
plt.title('Distribution of Monthly Returns')
plt.xlabel('Monthly Return')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
plt.savefig('plots/return_distribution.png')
plt.close()

# 7. Distribution of excess returns
plt.figure(figsize=(12, 8))
sns.histplot(df['excess_return'], kde=True, color='green')
plt.axvline(x=0, color='r', linestyle='-', alpha=0.5)
plt.title('Distribution of Monthly Excess Returns')
plt.xlabel('Excess Return')
plt.ylabel('Frequency')
plt.grid(True)
plt.savefig('plots/excess_return_distribution.png')
plt.close()

# 8. Calendar month performance heatmap
df['Year'] = df['tradeDate'].dt.year
df['Month'] = df['tradeDate'].dt.month
heatmap_data = df.pivot_table(values='excess_return', index='Year', columns='Month')
plt.figure(figsize=(14, 10))
sns.heatmap(heatmap_data, cmap='RdBu_r', center=0, annot=True, fmt=".2f")
plt.title('Calendar Month Excess Returns: PGTIX - QQQ')
plt.savefig('plots/calendar_heatmap.png')
plt.close()

# 9. Drawdown analysis
def calculate_drawdown(returns):
    cum_returns = (1 + returns).cumprod()
    running_max = np.maximum.accumulate(cum_returns)
    drawdown = (cum_returns / running_max) - 1
    return drawdown

df['PGTIX_drawdown'] = calculate_drawdown(df['PGTIX'])
df['QQQ_drawdown'] = calculate_drawdown(df['QQQ'])

plt.figure(figsize=(12, 8))
plt.plot(df['tradeDate'], df['PGTIX_drawdown'], label='PGTIX')
plt.plot(df['tradeDate'], df['QQQ_drawdown'], label='QQQ')
plt.title('Drawdown Analysis: PGTIX vs QQQ')
plt.xlabel('Date')
plt.ylabel('Drawdown')
plt.legend()
plt.grid(True)
plt.savefig('plots/drawdown.png')
plt.close()

# Prepare a comprehensive analysis report
with open('fund_analysis_report.md', 'w') as f:
    f.write("# Fund Performance Analysis: PGTIX vs QQQ\n\n")
    
    f.write("## 1. Overall Performance Analysis\n\n")
    final_pgtix_cum = df['PGTIX_cum'].iloc[-1]
    final_qqq_cum = df['QQQ_cum'].iloc[-1]
    f.write(f"- Total Return PGTIX: {final_pgtix_cum:.2%}\n")
    f.write(f"- Total Return QQQ: {final_qqq_cum:.2%}\n")
    f.write(f"- Outperformance: {final_pgtix_cum - final_qqq_cum:.2%}\n\n")
    
    annualized_pgtix = (1 + final_pgtix_cum) ** (12 / len(df)) - 1
    annualized_qqq = (1 + final_qqq_cum) ** (12 / len(df)) - 1
    
    f.write(f"- Annualized Return PGTIX: {annualized_pgtix:.2%}\n")
    f.write(f"- Annualized Return QQQ: {annualized_qqq:.2%}\n")
    f.write(f"- Annualized Outperformance: {annualized_pgtix - annualized_qqq:.2%}\n\n")
    
    f.write("## 2. Consistency of Performance\n\n")
    f.write(f"- Periods where PGTIX outperformed QQQ: {outperform_count} out of {total_periods} ({outperform_percentage:.1f}%)\n")
    f.write(f"- Longest streak of consecutive outperformance: {longest_outperform} months\n")
    f.write(f"- Longest streak of consecutive underperformance: {longest_underperform} months\n\n")
    
    f.write("## 3. Risk Metrics\n\n")
    f.write(f"- PGTIX Standard Deviation: {std_pgtix:.2%}\n")
    f.write(f"- QQQ Standard Deviation: {std_qqq:.2%}\n")
    f.write(f"- PGTIX Sharpe Ratio: {sharpe_pgtix:.2f}\n")
    f.write(f"- QQQ Sharpe Ratio: {sharpe_qqq:.2f}\n")
    f.write(f"- Correlation between PGTIX and QQQ: {correlation:.2f}\n")
    f.write(f"- Beta of PGTIX to QQQ: {beta:.2f}\n")
    f.write(f"- Alpha (Jensen's): {alpha:.2%} monthly\n")
    f.write(f"- Information Ratio: {information_ratio:.2f}\n")
    f.write(f"- Tracking Error: {tracking_error:.2%}\n\n")
    
    f.write("## 4. Market Environment Analysis\n\n")
    f.write(f"- Up-Market Capture Ratio: {up_market_capture:.2f}\n")
    f.write(f"- Down-Market Capture Ratio: {down_market_capture:.2f}\n")
    f.write(f"- Win/Loss Ratio: {win_loss_ratio:.2f}\n\n")
    
    f.write("## 5. Performance Distribution\n\n")
    f.write(f"- Mean Monthly Return PGTIX: {mean_pgtix:.2%}\n")
    f.write(f"- Mean Monthly Return QQQ: {mean_qqq:.2%}\n")
    f.write(f"- Mean Monthly Excess Return: {df['excess_return'].mean():.2%}\n")
    f.write(f"- Median Monthly Excess Return: {df['excess_return'].median():.2%}\n")
    
    f.write("\n## 6. Key Conclusions\n\n")
    
    # Fund performance vs benchmark conclusion
    if final_pgtix_cum > final_qqq_cum:
        f.write("1. **Performance vs Benchmark**: PGTIX has outperformed QQQ over the full period.\n\n")
    else:
        f.write("1. **Performance vs Benchmark**: PGTIX has underperformed QQQ over the full period.\n\n")
    
    # Consistency conclusion
    if outperform_percentage > 55:
        consistency = "fairly consistent in outperforming"
    elif outperform_percentage < 45:
        consistency = "fairly consistent in underperforming"
    else:
        consistency = "inconsistent relative to"
    f.write(f"2. **Consistency**: PGTIX has been {consistency} QQQ with {outperform_percentage:.1f}% of months showing outperformance.\n\n")
    
    # Market environment conclusion
    if up_market_capture > 1 and down_market_capture < 1:
        mkt_env = "outperforms in up markets while providing downside protection"
    elif up_market_capture > 1 and down_market_capture > 1:
        mkt_env = "captures more upside but also more downside than"
    elif up_market_capture < 1 and down_market_capture < 1:
        mkt_env = "underperforms in up markets but provides good downside protection versus"
    else:
        mkt_env = "underperforms in both up and down markets compared to"
    f.write(f"3. **Market Environments**: PGTIX {mkt_env} QQQ.\n\n")
    
    # Skill vs luck conclusion
    if abs(alpha) > 0.002 and abs(information_ratio) > 0.3:
        skill = "There is evidence of manager skill"
    else:
        skill = "The performance difference could be attributed more to randomness than skill"
    f.write(f"4. **Skill vs Luck**: {skill}, with a monthly alpha of {alpha:.2%} and information ratio of {information_ratio:.2f}.\n\n")
    
    # Distinctiveness conclusion
    if correlation < 0.85:
        distinct = "quite distinct"
    elif correlation < 0.95:
        distinct = "somewhat distinct"
    else:
        distinct = "very similar"
    f.write(f"5. **Distinctiveness**: PGTIX returns are {distinct} from QQQ with a correlation of {correlation:.2f} and beta of {beta:.2f}.\n\n")
    
    # Distribution conclusion
    if abs(df['excess_return']).max() > 0.1 and df['excess_return'].std() > 0.05:
        dist = "significantly influenced by a few extreme periods"
    else:
        dist = "a result of consistent smaller differences rather than extreme outliers"
    f.write(f"6. **Return Distribution**: The performance differential appears to be {dist}.\n\n")
    
    f.write("\n## 7. Visualizations\n\n")
    f.write("Please refer to the plots directory for visual representations of the analysis:\n\n")
    f.write("- Cumulative Returns\n")
    f.write("- Rolling 12-Month Returns\n")
    f.write("- Monthly Excess Returns\n")
    f.write("- Returns Scatter Plot\n")
    f.write("- Return Distributions\n")
    f.write("- Calendar Month Heatmap\n")
    f.write("- Drawdown Analysis\n")

print("Analysis complete! Check the 'fund_analysis_report.md' file and the 'plots' directory for full results.") 