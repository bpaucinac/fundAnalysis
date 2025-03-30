import os
import pandas as pd
import numpy as np
import json
from datetime import datetime
import argparse
from pathlib import Path

def extract_analytics(fund1_ticker, fund2_ticker, start_date=None, end_date=None, output_file=None):
    """
    Extract all analytics from comprehensive analysis and save to a JSON file.
    
    Parameters:
    fund1_ticker (str): Ticker symbol for the first fund
    fund2_ticker (str): Ticker symbol for the second fund (benchmark)
    start_date (str, optional): Start date for analysis in 'YYYY-MM-DD' format
    end_date (str, optional): End date for analysis in 'YYYY-MM-DD' format
    output_file (str, optional): Output JSON filename
    """
    # Load the data
    df = pd.read_csv('fundRetuns.csv')
    
    # Convert date to datetime format
    df['tradeDate'] = pd.to_datetime(df['tradeDate'], format='%m/%d/%y')
    df = df.sort_values('tradeDate')
    
    # Filter by date range if provided
    if start_date:
        df = df[df['tradeDate'] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df['tradeDate'] <= pd.to_datetime(end_date)]
        
    # Check if the fund tickers exist in the dataframe
    if fund1_ticker not in df.columns:
        raise ValueError(f"Fund ticker {fund1_ticker} not found in dataset")
    if fund2_ticker not in df.columns:
        raise ValueError(f"Fund ticker {fund2_ticker} not found in dataset")
    
    # Set default output filename if not provided
    if output_file is None:
        output_file = f"fund_analytics_{fund1_ticker}_vs_{fund2_ticker}.json"
    
    # Define risk-free rate (adjust as needed)
    risk_free_rate = 0.02 / 12  # Assuming 2% annual risk-free rate
    
    # Calculate cumulative returns
    df[f'{fund1_ticker}_cum'] = (1 + df[fund1_ticker]).cumprod() - 1
    df[f'{fund2_ticker}_cum'] = (1 + df[fund2_ticker]).cumprod() - 1
    
    # Calculate excess returns
    df['excess_return'] = df[fund1_ticker] - df[fund2_ticker]
    
    # Calculate risk metrics
    # 1. Rolling volatility - 12-month window
    df[f'{fund1_ticker}_rolling_vol'] = df[fund1_ticker].rolling(12).std() * np.sqrt(12)
    df[f'{fund2_ticker}_rolling_vol'] = df[fund2_ticker].rolling(12).std() * np.sqrt(12)
    df['relative_vol'] = df[f'{fund1_ticker}_rolling_vol'] / df[f'{fund2_ticker}_rolling_vol']
    
    # 2. Rolling beta - 24-month window
    df['rolling_beta_24m'] = df[fund1_ticker].rolling(24).cov(df[fund2_ticker]) / df[fund2_ticker].rolling(24).var()
    
    # 3. Rolling Sharpe ratio - 24-month window
    df[f'{fund1_ticker}_sharpe'] = (df[fund1_ticker].rolling(24).mean() - risk_free_rate) / df[fund1_ticker].rolling(24).std() * np.sqrt(12)
    df[f'{fund2_ticker}_sharpe'] = (df[fund2_ticker].rolling(24).mean() - risk_free_rate) / df[fund2_ticker].rolling(24).std() * np.sqrt(12)
    
    # 4. Information Ratio - 24-month window
    df['rolling_info_ratio'] = df['excess_return'].rolling(24).mean() / df['excess_return'].rolling(24).std()
    
    # PART 1: Generate Key Performance Metrics
    
    # Overall statistics
    fund1_total_return = df[f'{fund1_ticker}_cum'].iloc[-1]
    fund2_total_return = df[f'{fund2_ticker}_cum'].iloc[-1]
    excess_total_return = fund1_total_return - fund2_total_return
    
    # Annualized returns
    num_years = len(df) / 12
    fund1_annual_return = (1 + fund1_total_return) ** (1 / num_years) - 1
    fund2_annual_return = (1 + fund2_total_return) ** (1 / num_years) - 1
    excess_annual_return = fund1_annual_return - fund2_annual_return
    
    # Risk metrics for the entire period
    fund1_vol = df[fund1_ticker].std() * np.sqrt(12)  # Annualized
    fund2_vol = df[fund2_ticker].std() * np.sqrt(12)  # Annualized
    relative_vol = fund1_vol / fund2_vol
    
    fund1_sharpe = (df[fund1_ticker].mean() - risk_free_rate) / df[fund1_ticker].std() * np.sqrt(12)
    fund2_sharpe = (df[fund2_ticker].mean() - risk_free_rate) / df[fund2_ticker].std() * np.sqrt(12)
    
    correlation = df[fund1_ticker].corr(df[fund2_ticker])
    beta = df[fund1_ticker].cov(df[fund2_ticker]) / df[fund2_ticker].var()
    
    alpha_monthly = df[fund1_ticker].mean() - (risk_free_rate + beta * (df[fund2_ticker].mean() - risk_free_rate))
    alpha_annual = alpha_monthly * 12
    
    tracking_error = df['excess_return'].std() * np.sqrt(12)
    info_ratio = (df['excess_return'].mean() * 12) / tracking_error
    
    # Outperformance frequency
    outperform_count = (df['excess_return'] > 0).sum()
    total_periods = len(df)
    outperform_percentage = (outperform_count / total_periods) * 100
    
    # Calculate drawdowns
    def calculate_drawdown(returns):
        cum_returns = (1 + returns).cumprod()
        running_max = np.maximum.accumulate(cum_returns)
        drawdown = (cum_returns / running_max) - 1
        return drawdown
    
    df[f'{fund1_ticker}_drawdown'] = calculate_drawdown(df[fund1_ticker])
    df[f'{fund2_ticker}_drawdown'] = calculate_drawdown(df[fund2_ticker])
    
    # Calculate max drawdown
    max_drawdown_fund1 = df[f'{fund1_ticker}_drawdown'].min()
    max_drawdown_fund2 = df[f'{fund2_ticker}_drawdown'].min()
    
    # PART 2: Create Time Period Analysis
    
    # Split the data into three equal periods for analysis
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
    
    # Calculate performance metrics for each period
    period_results = {}
    
    for period_name, period_data in periods.items():
        # Calculate returns
        period_fund1_return = (1 + period_data[fund1_ticker]).prod() - 1
        period_fund2_return = (1 + period_data[fund2_ticker]).prod() - 1
        
        # Calculate annualized returns
        period_years = len(period_data) / 12
        if period_years > 0:
            period_fund1_annualized = (1 + period_fund1_return) ** (1 / period_years) - 1
            period_fund2_annualized = (1 + period_fund2_return) ** (1 / period_years) - 1
        else:
            period_fund1_annualized = period_fund1_return
            period_fund2_annualized = period_fund2_return
            
        # Calculate risk metrics
        period_fund1_std = period_data[fund1_ticker].std() * np.sqrt(12)
        period_fund2_std = period_data[fund2_ticker].std() * np.sqrt(12)
        
        period_correlation = period_data[fund1_ticker].corr(period_data[fund2_ticker])
        period_beta = period_data[fund1_ticker].cov(period_data[fund2_ticker]) / period_data[fund2_ticker].var() if period_data[fund2_ticker].var() != 0 else np.nan
        
        # Calculate alpha
        period_alpha_monthly = period_data[fund1_ticker].mean() - (risk_free_rate + period_beta * (period_data[fund2_ticker].mean() - risk_free_rate))
        period_alpha_annualized = period_alpha_monthly * 12
        
        # Calculate outperformance percentage
        period_outperform = (period_data[fund1_ticker] > period_data[fund2_ticker]).sum()
        period_outperform_pct = (period_outperform / len(period_data)) * 100
        
        # Store results
        period_results[period_name] = {
            'date_range': f"{period_data['tradeDate'].min().strftime('%m/%Y')} - {period_data['tradeDate'].max().strftime('%m/%Y')}",
            f'{fund1_ticker}_return': period_fund1_return,
            f'{fund2_ticker}_return': period_fund2_return,
            'excess_return': period_fund1_return - period_fund2_return,
            f'{fund1_ticker}_annualized': period_fund1_annualized,
            f'{fund2_ticker}_annualized': period_fund2_annualized,
            f'{fund1_ticker}_std': period_fund1_std,
            f'{fund2_ticker}_std': period_fund2_std,
            'relative_vol': period_fund1_std / period_fund2_std if period_fund2_std != 0 else np.nan,
            'correlation': period_correlation,
            'beta': period_beta,
            'alpha_annualized': period_alpha_annualized,
            'outperformance_pct': period_outperform_pct
        }
    
    # PART 3: Risk in Different Market Environments
    
    # Define up/down markets based on benchmark performance
    df['market_environment'] = 'neutral'
    up_threshold = df[fund2_ticker].quantile(0.7)  # Top 30% returns define up markets
    down_threshold = df[fund2_ticker].quantile(0.3)  # Bottom 30% returns define down markets
    df.loc[df[fund2_ticker] > up_threshold, 'market_environment'] = 'up'
    df.loc[df[fund2_ticker] < down_threshold, 'market_environment'] = 'down'
    
    # Calculate risk in different market environments
    risk_by_environment = {}
    for env in ['up', 'neutral', 'down']:
        env_data = df[df['market_environment'] == env]
        if len(env_data) > 0:
            risk_by_environment[env] = {
                'count': len(env_data),
                f'{fund1_ticker}_vol': env_data[fund1_ticker].std() * np.sqrt(12),
                f'{fund2_ticker}_vol': env_data[fund2_ticker].std() * np.sqrt(12),
                'relative_vol': (env_data[fund1_ticker].std() / env_data[fund2_ticker].std()),
                'beta': env_data[fund1_ticker].cov(env_data[fund2_ticker]) / env_data[fund2_ticker].var() if env_data[fund2_ticker].var() != 0 else np.nan,
                f'{fund1_ticker}_mean': env_data[fund1_ticker].mean() * 12,  # Annualized
                f'{fund2_ticker}_mean': env_data[fund2_ticker].mean() * 12,  # Annualized
                'excess_return': (env_data[fund1_ticker].mean() - env_data[fund2_ticker].mean()) * 12,  # Annualized
            }
    
    # Create risk-return quadrant
    mid_return = (fund1_annual_return + fund2_annual_return) / 2
    mid_risk = (fund1_vol + fund2_vol) / 2
    
    # Determine quadrant positioning for both funds
    fund1_quadrant = "Undefined"
    if fund1_annual_return > mid_return and fund1_vol < mid_risk:
        fund1_quadrant = "Low Risk, High Return (Optimal)"
    elif fund1_annual_return > mid_return and fund1_vol >= mid_risk:
        fund1_quadrant = "High Risk, High Return"
    elif fund1_annual_return <= mid_return and fund1_vol < mid_risk:
        fund1_quadrant = "Low Risk, Low Return"
    elif fund1_annual_return <= mid_return and fund1_vol >= mid_risk:
        fund1_quadrant = "High Risk, Low Return (Worst)"
    
    fund2_quadrant = "Undefined"
    if fund2_annual_return > mid_return and fund2_vol < mid_risk:
        fund2_quadrant = "Low Risk, High Return (Optimal)"
    elif fund2_annual_return > mid_return and fund2_vol >= mid_risk:
        fund2_quadrant = "High Risk, High Return"
    elif fund2_annual_return <= mid_return and fund2_vol < mid_risk:
        fund2_quadrant = "Low Risk, Low Return"
    elif fund2_annual_return <= mid_return and fund2_vol >= mid_risk:
        fund2_quadrant = "High Risk, Low Return (Worst)"
    
    # Calculate which periods had best/worst performance
    best_period_idx = np.argmax([period_results[p]['excess_return'] for p in ['Early', 'Middle', 'Late']])
    worst_period_idx = np.argmin([period_results[p]['excess_return'] for p in ['Early', 'Middle', 'Late']])
    best_period = ['Early', 'Middle', 'Late'][best_period_idx]
    worst_period = ['Early', 'Middle', 'Late'][worst_period_idx]
    
    # Market Environment Observations
    up_market_outperformance = risk_by_environment['up']['excess_return'] > 0
    down_market_outperformance = risk_by_environment['down']['excess_return'] > 0
    
    # Time Series Data
    time_series_data = []
    for idx, row in df.iterrows():
        time_series_data.append({
            'date': row['tradeDate'].strftime('%m/%d/%Y'),
            f'{fund1_ticker}_return': row[fund1_ticker],
            f'{fund2_ticker}_return': row[fund2_ticker],
            f'{fund1_ticker}_cum_return': row[f'{fund1_ticker}_cum'],
            f'{fund2_ticker}_cum_return': row[f'{fund2_ticker}_cum'],
            'excess_return': row['excess_return'],
            f'{fund1_ticker}_rolling_vol': row[f'{fund1_ticker}_rolling_vol'],
            f'{fund2_ticker}_rolling_vol': row[f'{fund2_ticker}_rolling_vol'],
            'relative_vol': row['relative_vol'],
            'rolling_beta_24m': row['rolling_beta_24m'],
            f'{fund1_ticker}_drawdown': row[f'{fund1_ticker}_drawdown'],
            f'{fund2_ticker}_drawdown': row[f'{fund2_ticker}_drawdown'],
            'market_environment': row['market_environment'],
        })
    
    # Build final analytics dictionary
    analytics = {
        'metadata': {
            'fund1_ticker': fund1_ticker,
            'fund2_ticker': fund2_ticker,
            'start_date': df['tradeDate'].min().strftime('%m/%d/%Y'),
            'end_date': df['tradeDate'].max().strftime('%m/%d/%Y'),
            'analysis_date': datetime.now().strftime('%m/%d/%Y'),
            'total_periods': total_periods,
            'years_analyzed': num_years,
            'risk_free_rate_annual': risk_free_rate * 12,
        },
        'overall_metrics': {
            'cumulative_returns': {
                'fund1_total_return': fund1_total_return,
                'fund2_total_return': fund2_total_return,
                'excess_total_return': excess_total_return,
            },
            'annualized_returns': {
                'fund1_annual_return': fund1_annual_return,
                'fund2_annual_return': fund2_annual_return,
                'excess_annual_return': excess_annual_return,
            },
            'volatility': {
                'fund1_vol': fund1_vol,
                'fund2_vol': fund2_vol,
                'relative_vol': relative_vol,
            },
            'risk_metrics': {
                'fund1_sharpe': fund1_sharpe,
                'fund2_sharpe': fund2_sharpe,
                'correlation': correlation,
                'beta': beta,
                'alpha_monthly': alpha_monthly,
                'alpha_annual': alpha_annual,
                'tracking_error': tracking_error,
                'info_ratio': info_ratio,
            },
            'outperformance': {
                'outperform_count': outperform_count,
                'total_periods': total_periods,
                'outperform_percentage': outperform_percentage,
            },
            'drawdowns': {
                'max_drawdown_fund1': max_drawdown_fund1,
                'max_drawdown_fund2': max_drawdown_fund2,
            },
            'risk_return_quadrant': {
                'fund1_quadrant': fund1_quadrant,
                'fund2_quadrant': fund2_quadrant,
                'mid_return': mid_return,
                'mid_risk': mid_risk,
            },
        },
        'period_analysis': period_results,
        'market_environment_analysis': risk_by_environment,
        'time_period_insights': {
            'best_period': best_period,
            'worst_period': worst_period,
            'early_to_late_correlation_change': period_results['Late']['correlation'] - period_results['Early']['correlation'],
            'early_to_late_beta_change': period_results['Late']['beta'] - period_results['Early']['beta'],
            'early_to_late_rel_vol_change': period_results['Late']['relative_vol'] - period_results['Early']['relative_vol'],
            'early_to_late_alpha_change': period_results['Late']['alpha_annualized'] - period_results['Early']['alpha_annualized'],
        },
        'market_environment_insights': {
            'up_market_outperformance': up_market_outperformance,
            'down_market_outperformance': down_market_outperformance,
            'highest_beta_market': max(risk_by_environment.items(), key=lambda x: x[1]['beta'])[0],
            'highest_beta_value': risk_by_environment[max(risk_by_environment.items(), key=lambda x: x[1]['beta'])[0]]['beta'],
        },
        'time_series_data': time_series_data,
    }
    
    # Convert NaN values to None for JSON compatibility
    analytics_json = json.loads(pd.DataFrame([analytics]).to_json(orient='records'))[0]
    
    # Save to JSON file
    with open(output_file, 'w') as f:
        json.dump(analytics_json, f, indent=2)
    
    print(f"Analytics saved to {output_file}")
    return analytics_json

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Extract fund analytics and save to JSON file.')
    parser.add_argument('--fund1', default='PGTIX', help='First fund ticker symbol (default: PGTIX)')
    parser.add_argument('--fund2', default='QQQ', help='Second fund ticker symbol (default: QQQ)')
    parser.add_argument('--start_date', help='Start date in YYYY-MM-DD format')
    parser.add_argument('--end_date', help='End date in YYYY-MM-DD format')
    parser.add_argument('--output', help='Output JSON filename')
    
    args = parser.parse_args()
    
    # Execute the analysis with command-line arguments
    result = extract_analytics(
        fund1_ticker=args.fund1,
        fund2_ticker=args.fund2,
        start_date=args.start_date,
        end_date=args.end_date,
        output_file=args.output
    )
    
    # Print summary
    print("\nAnalysis Summary:")
    print(f"Fund Comparison: {result['metadata']['fund1_ticker']} vs {result['metadata']['fund2_ticker']}")
    print(f"Total Return: {result['metadata']['fund1_ticker']} ({result['overall_metrics']['cumulative_returns']['fund1_total_return']:.2%}) vs {result['metadata']['fund2_ticker']} ({result['overall_metrics']['cumulative_returns']['fund2_total_return']:.2%})")
    print(f"Risk-Return Positioning: {result['metadata']['fund1_ticker']} is in '{result['overall_metrics']['risk_return_quadrant']['fund1_quadrant']}' quadrant")
    print(f"Alpha: {result['overall_metrics']['risk_metrics']['alpha_annual']:.2%} | Beta: {result['overall_metrics']['risk_metrics']['beta']:.2f}")
    print(f"Analytics saved to: {args.output if args.output else f'fund_analytics_{args.fund1}_vs_{args.fund2}.json'}") 