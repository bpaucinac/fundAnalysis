import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import datetime
from fpdf import FPDF
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import argparse

# Create necessary directories if they don't exist
for directory in ['plots', 'risk_plots', 'comprehensive_plots']:
    if not os.path.exists(directory):
        os.makedirs(directory)

class ComprehensiveAnalysisPDF(FPDF):
    def header(self):
        # Add header
        self.set_font('Arial', 'B', 15)
        self.set_text_color(0, 51, 102)  # Dark blue
        self.cell(0, 10, 'Fund Comparison: Comprehensive Analysis', 0, 1, 'C')
        self.ln(5)
        
    def footer(self):
        # Position at 1.5 cm from bottom
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.set_text_color(128)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
        self.cell(0, 10, f'Analysis Date: {datetime.now().strftime("%B %d, %Y")}', 0, 0, 'R')

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.set_fill_color(200, 220, 255)
        self.cell(0, 6, title, 0, 1, 'L', 1)
        self.ln(4)
        
    def chapter_body(self, body):
        self.set_font('Arial', '', 10)
        self.multi_cell(0, 5, body)
        self.ln()

    def section_title(self, title):
        self.set_font('Arial', 'B', 11)
        self.cell(0, 6, title, 0, 1, 'L')
        self.ln(2)
        
    def add_image(self, image_path, w=0, h=0, caption=None):
        if os.path.exists(image_path):
            self.image(image_path, x=None, y=None, w=w, h=h)
            if caption:
                self.set_font('Arial', 'I', 8)
                self.cell(0, 5, caption, 0, 1, 'C')
            self.ln(5)
        else:
            self.cell(0, 5, f"Image not found: {image_path}", 0, 1, 'C')
            self.ln(5)
            
    def create_table(self, headers, data, col_widths=None):
        # Table header
        self.set_font('Arial', 'B', 9)
        self.set_fill_color(230, 230, 250)  # Light purple background
        
        # Calculate column widths if not provided
        if col_widths is None:
            col_widths = [self.w / len(headers)] * len(headers)
            
        # Print header
        for i, header in enumerate(headers):
            self.cell(col_widths[i], 7, header, 1, 0, 'C', 1)
        self.ln()
        
        # Table data
        self.set_font('Arial', '', 8)
        self.set_fill_color(245, 245, 245)  # Light gray for alternating rows
        
        # Print rows
        fill = False
        for row in data:
            for i, cell in enumerate(row):
                self.cell(col_widths[i], 6, str(cell), 1, 0, 'C', fill)
            self.ln()
            fill = not fill

def analyze_funds(fund1_ticker, fund2_ticker, start_date=None, end_date=None, output_file=None):
    """
    Perform comprehensive analysis comparing two funds and generate a PDF report.
    
    Parameters:
    fund1_ticker (str): Ticker symbol for the first fund
    fund2_ticker (str): Ticker symbol for the second fund (benchmark)
    start_date (str, optional): Start date for analysis in 'YYYY-MM-DD' format
    end_date (str, optional): End date for analysis in 'YYYY-MM-DD' format
    output_file (str, optional): Output PDF filename
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
        output_file = f"{fund1_ticker}_vs_{fund2_ticker}_Analysis.pdf"
    
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
    
    # PART 4: Create Visualizations
    
    # 1. Cumulative returns plot
    plt.figure(figsize=(12, 8))
    plt.plot(df['tradeDate'], df[f'{fund1_ticker}_cum'], label=fund1_ticker)
    plt.plot(df['tradeDate'], df[f'{fund2_ticker}_cum'], label=fund2_ticker)
    plt.title(f'Cumulative Returns: {fund1_ticker} vs {fund2_ticker}')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True)
    plt.savefig('comprehensive_plots/cumulative_returns.png')
    plt.close()
    
    # 2. Monthly returns scatter plot
    plt.figure(figsize=(12, 8))
    plt.scatter(df[fund2_ticker], df[fund1_ticker], alpha=0.7)
    
    # Add regression line
    m, b = np.polyfit(df[fund2_ticker], df[fund1_ticker], 1)
    plt.plot(df[fund2_ticker], m * df[fund2_ticker] + b, color='red', linestyle='--')
    
    # Add y=x line for reference
    min_val = min(df[fund1_ticker].min(), df[fund2_ticker].min())
    max_val = max(df[fund1_ticker].max(), df[fund2_ticker].max())
    plt.plot([min_val, max_val], [min_val, max_val], color='green', linestyle=':')
    
    plt.title(f'Monthly Returns: {fund1_ticker} vs {fund2_ticker}')
    plt.xlabel(f'{fund2_ticker} Returns')
    plt.ylabel(f'{fund1_ticker} Returns')
    plt.grid(True)
    plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
    plt.savefig('comprehensive_plots/returns_scatter.png')
    plt.close()
    
    # 3. Rolling volatility plot
    plt.figure(figsize=(12, 8))
    plt.plot(df['tradeDate'], df[f'{fund1_ticker}_rolling_vol'], label=f'{fund1_ticker} Volatility (12m)')
    plt.plot(df['tradeDate'], df[f'{fund2_ticker}_rolling_vol'], label=f'{fund2_ticker} Volatility (12m)')
    plt.title('Rolling 12-Month Annualized Volatility')
    plt.xlabel('Date')
    plt.ylabel('Annualized Volatility')
    plt.legend()
    plt.grid(True)
    plt.savefig('comprehensive_plots/rolling_volatility.png')
    plt.close()
    
    # 4. Risk-Return Quadrant Chart
    plt.figure(figsize=(12, 12))
    
    # Create data for the different periods
    periods_for_plot = ['Early', 'Middle', 'Late', 'Full']
    colors = ['blue', 'green', 'red', 'purple']
    
    # Prepare the data
    fund1_returns = [period_results[period][f'{fund1_ticker}_annualized'] for period in periods_for_plot]
    fund2_returns = [period_results[period][f'{fund2_ticker}_annualized'] for period in periods_for_plot]
    fund1_risks = [period_results[period][f'{fund1_ticker}_std'] for period in periods_for_plot]
    fund2_risks = [period_results[period][f'{fund2_ticker}_std'] for period in periods_for_plot]
    
    # Calculate the midpoints for the quadrant lines
    mid_return = (max(fund1_returns + fund2_returns) + min(fund1_returns + fund2_returns)) / 2
    mid_risk = (max(fund1_risks + fund2_risks) + min(fund1_risks + fund2_risks)) / 2
    
    # Create the plot with connected pairs
    for i, period in enumerate(periods_for_plot):
        plt.scatter(fund1_risks[i], fund1_returns[i], color=colors[i], s=100, label=f'{fund1_ticker} - {period}')
        plt.scatter(fund2_risks[i], fund2_returns[i], color=colors[i], marker='s', s=100, label=f'{fund2_ticker} - {period}')
        plt.plot([fund1_risks[i], fund2_risks[i]], [fund1_returns[i], fund2_returns[i]], color=colors[i], linestyle='--', alpha=0.7)
        
        # Add annotations
        plt.annotate(f"{period}", (fund1_risks[i], fund1_returns[i]), xytext=(10, 5), textcoords='offset points')
        plt.annotate(f"{period}", (fund2_risks[i], fund2_returns[i]), xytext=(10, 5), textcoords='offset points')
    
    # Add quadrant lines and labels
    plt.axvline(x=mid_risk, color='black', linestyle='--', alpha=0.5)
    plt.axhline(y=mid_return, color='black', linestyle='--', alpha=0.5)
    
    # Add quadrant labels
    plt.text(min(fund1_risks + fund2_risks), mid_return + 0.01, "HIGH RETURN", ha='left', va='bottom')
    plt.text(min(fund1_risks + fund2_risks), mid_return - 0.03, "LOW RETURN", ha='left', va='top')
    plt.text(mid_risk + 0.01, max(fund1_returns + fund2_returns), "HIGH RISK", ha='center', va='top')
    plt.text(mid_risk - 0.01, max(fund1_returns + fund2_returns), "LOW RISK", ha='center', va='top')
    
    # Add quadrant annotations
    plt.text(mid_risk - 0.02, mid_return + 0.02, "Low Risk\nHigh Return\n(Optimal)", ha='right', va='bottom', fontweight='bold')
    plt.text(mid_risk + 0.02, mid_return + 0.02, "High Risk\nHigh Return", ha='left', va='bottom')
    plt.text(mid_risk - 0.02, mid_return - 0.02, "Low Risk\nLow Return", ha='right', va='top')
    plt.text(mid_risk + 0.02, mid_return - 0.02, "High Risk\nLow Return\n(Worst)", ha='left', va='top', fontweight='bold')
    
    plt.title(f'Risk-Return Quadrant Analysis: {fund1_ticker} vs {fund2_ticker}')
    plt.xlabel('Risk (Annualized Standard Deviation)')
    plt.ylabel('Return (Annualized)')
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('comprehensive_plots/risk_return_quadrant.png')
    plt.close() 
    
    # 5. Drawdown chart
    plt.figure(figsize=(12, 8))
    plt.plot(df['tradeDate'], df[f'{fund1_ticker}_drawdown'], label=fund1_ticker)
    plt.plot(df['tradeDate'], df[f'{fund2_ticker}_drawdown'], label=fund2_ticker)
    plt.title('Drawdown Analysis')
    plt.xlabel('Date')
    plt.ylabel('Drawdown')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('comprehensive_plots/drawdown.png')
    plt.close()
    
    # 6. Relative volatility over time
    plt.figure(figsize=(12, 8))
    plt.plot(df['tradeDate'][11:], df['relative_vol'][11:])  # Skip first 11 rows which are NaN
    plt.axhline(y=1.0, color='r', linestyle='--')
    plt.title(f'Relative Volatility: {fund1_ticker} / {fund2_ticker}')
    plt.xlabel('Date')
    plt.ylabel('Relative Volatility')
    plt.grid(True)
    plt.savefig('comprehensive_plots/relative_volatility.png')
    plt.close()
    
    # PART 5: Generate PDF Report
    
    # Create PDF instance
    pdf = ComprehensiveAnalysisPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Cover page
    pdf.add_page()
    pdf.set_font('Arial', 'B', 24)
    pdf.set_text_color(0, 51, 102)  # Dark blue
    pdf.cell(0, 20, '', 0, 1, 'C')  # Space at top
    pdf.cell(0, 20, f'Fund Comparison: {fund1_ticker} vs {fund2_ticker}', 0, 1, 'C')
    pdf.cell(0, 10, 'Comprehensive Analysis', 0, 1, 'C')
    
    # Add horizontal line
    pdf.cell(0, 1, '', 0, 1)
    pdf.line(30, pdf.get_y(), 180, pdf.get_y())
    pdf.cell(0, 10, '', 0, 1)  # Space after line
    
    # Add main visualization to cover
    pdf.image('comprehensive_plots/risk_return_quadrant.png', x=30, y=None, w=150)
    
    pdf.set_font('Arial', 'I', 10)
    pdf.set_text_color(128)
    pdf.cell(0, 10, '', 0, 1)  # Space before date
    pdf.cell(0, 10, f'Prepared on: {datetime.now().strftime("%B %d, %Y")}', 0, 1, 'C')
    pdf.cell(0, 10, 'For Investment Professional Use Only', 0, 1, 'C')
    
    # Table of contents
    pdf.add_page()
    pdf.set_font('Arial', 'B', 16)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 10, 'Table of Contents', 0, 1, 'L')
    pdf.ln(5)
    
    pdf.set_font('Arial', '', 12)
    toc_items = [
        ("1. Executive Summary", 3),
        ("2. Performance Analysis", 4),
        ("3. Risk Analysis", 6),
        ("4. Risk-Return Profile", 8),
        ("5. Time Period Analysis", 9),
        ("6. Market Environment Analysis", 10),
        ("7. Investment Implications", 11)
    ]
    
    for item, page in toc_items:
        pdf.cell(0, 8, item, 0, 0)
        pdf.cell(0, 8, f'Page {page}', 0, 1, 'R')
    
    # Executive Summary
    pdf.add_page()
    pdf.chapter_title("1. Executive Summary")
    
    # Determine primary quadrant positioning for both funds
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
    
    executive_summary = (
        f"This report provides a comprehensive analysis comparing {fund1_ticker} and {fund2_ticker} "
        f"from {df['tradeDate'].min().strftime('%B %Y')} to {df['tradeDate'].max().strftime('%B %Y')}. "
        f"The analysis evaluates performance, risk metrics, and behavior across different market environments.\n\n"
        
        f"Key Findings:\n"
        f"- Total Return: {fund1_ticker} ({fund1_total_return:.2%}) vs {fund2_ticker} ({fund2_total_return:.2%}), "
        f"differential of {excess_total_return:.2%}\n"
        f"- Annualized Return: {fund1_ticker} ({fund1_annual_return:.2%}) vs {fund2_ticker} ({fund2_annual_return:.2%}), "
        f"differential of {excess_annual_return:.2%}\n"
        f"- Volatility: {fund1_ticker} ({fund1_vol:.2%}) vs {fund2_ticker} ({fund2_vol:.2%}), "
        f"relative volatility of {relative_vol:.2f}x\n"
        f"- Risk-Return Profile: {fund1_ticker} is in the '{fund1_quadrant}' quadrant, while {fund2_ticker} is in the '{fund2_quadrant}' quadrant\n"
        f"- Sharpe Ratio: {fund1_ticker} ({fund1_sharpe:.2f}) vs {fund2_ticker} ({fund2_sharpe:.2f})\n"
        f"- Maximum Drawdown: {fund1_ticker} ({max_drawdown_fund1:.2%}) vs {fund2_ticker} ({max_drawdown_fund2:.2%})\n"
        f"- Beta: {beta:.2f}, Alpha (annualized): {alpha_annual:.2%}, Correlation: {correlation:.2f}\n\n"
        
        f"This analysis shows that {fund1_ticker} has " + 
        ("outperformed" if excess_total_return > 0 else "underperformed") + 
        f" {fund2_ticker} with " +
        ("lower" if fund1_vol < fund2_vol else "higher") + 
        f" volatility over the analysis period."
    )
    
    pdf.chapter_body(executive_summary)
    
    # Performance Analysis
    pdf.add_page()
    pdf.chapter_title("2. Performance Analysis")
    
    # Performance Overview Section
    pdf.section_title("Performance Overview")
    
    # Performance Summary Table
    perf_headers = ["Metric", fund1_ticker, fund2_ticker, "Difference/Ratio"]
    perf_data = [
        ["Total Return", f"{fund1_total_return:.2%}", f"{fund2_total_return:.2%}", f"{excess_total_return:.2%}"],
        ["Annualized Return", f"{fund1_annual_return:.2%}", f"{fund2_annual_return:.2%}", f"{excess_annual_return:.2%}"],
        ["Monthly Mean Return", f"{df[fund1_ticker].mean():.2%}", f"{df[fund2_ticker].mean():.2%}", f"{df[fund1_ticker].mean() - df[fund2_ticker].mean():.2%}"],
        ["Monthly Median Return", f"{df[fund1_ticker].median():.2%}", f"{df[fund2_ticker].median():.2%}", f"{df[fund1_ticker].median() - df[fund2_ticker].median():.2%}"],
        ["Positive Months", f"{(df[fund1_ticker] > 0).sum()} ({(df[fund1_ticker] > 0).mean():.1%})", 
                           f"{(df[fund2_ticker] > 0).sum()} ({(df[fund2_ticker] > 0).mean():.1%})", ""],
        ["Outperformance Frequency", f"{outperform_count} of {total_periods}", f"{total_periods - outperform_count} of {total_periods}", f"{outperform_percentage:.1f}%"]
    ]
    
    pdf.create_table(perf_headers, perf_data, [45, 45, 45, 55])
    pdf.ln(5)
    
    # Add Cumulative Returns Chart
    pdf.add_image('comprehensive_plots/cumulative_returns.png', w=180, caption=f"Figure 1: Cumulative Returns ({df['tradeDate'].min().strftime('%m/%Y')} - {df['tradeDate'].max().strftime('%m/%Y')})")
    
    # Add Returns Scatter Plot
    pdf.add_image('comprehensive_plots/returns_scatter.png', w=180, caption=f"Figure 2: Monthly Returns Comparison with Regression Line (beta={beta:.2f})")
    
    # Risk Analysis Page
    pdf.add_page()
    pdf.chapter_title("3. Risk Analysis")
    
    # Risk Metrics Section
    pdf.section_title("Risk Metrics")
    
    # Risk Metrics Table
    risk_headers = ["Metric", fund1_ticker, fund2_ticker, "Difference/Ratio"]
    risk_data = [
        ["Annualized Volatility", f"{fund1_vol:.2%}", f"{fund2_vol:.2%}", f"{relative_vol:.2f}x"],
        ["Sharpe Ratio", f"{fund1_sharpe:.2f}", f"{fund2_sharpe:.2f}", f"{fund1_sharpe - fund2_sharpe:.2f}"],
        ["Maximum Drawdown", f"{max_drawdown_fund1:.2%}", f"{max_drawdown_fund2:.2%}", f"{max_drawdown_fund1 - max_drawdown_fund2:.2%}"],
        ["Correlation", f"{correlation:.2f}", "1.00", ""],
        ["Beta", f"{beta:.2f}", "1.00", ""],
        ["Alpha (Annualized)", f"{alpha_annual:.2%}", "0.00%", ""],
        ["Information Ratio", f"{info_ratio:.2f}", "N/A", ""],
        ["Tracking Error", f"{tracking_error:.2%}", "N/A", ""]
    ]
    
    pdf.create_table(risk_headers, risk_data, [45, 45, 45, 55])
    pdf.ln(5)
    
    # Add Rolling Volatility Chart
    pdf.add_image('comprehensive_plots/rolling_volatility.png', w=180, caption="Figure 3: Rolling 12-Month Annualized Volatility")
    
    # Add Drawdown Analysis
    pdf.add_image('comprehensive_plots/drawdown.png', w=180, caption="Figure 4: Drawdown Analysis")
    
    # Add Relative Volatility Chart
    pdf.add_image('comprehensive_plots/relative_volatility.png', w=180, caption=f"Figure 5: Relative Volatility ({fund1_ticker}/{fund2_ticker})")
    
    # Risk-Return Profile
    pdf.add_page()
    pdf.chapter_title("4. Risk-Return Profile")
    
    pdf.section_title("Risk-Return Quadrant Analysis")
    pdf.chapter_body(
        f"The Risk-Return Quadrant Analysis plots both funds across different time periods, categorizing them into four quadrants:\n\n"
        f"- Low Risk, High Return (Optimal): The ideal position, offering high returns with low risk\n"
        f"- High Risk, High Return: Offers high returns but with increased volatility\n"
        f"- Low Risk, Low Return: Provides stability but with modest returns\n"
        f"- High Risk, Low Return (Worst): The least favorable position, exposing investors to high risk without commensurate returns\n\n"
        
        f"Currently, {fund1_ticker} is positioned in the '{fund1_quadrant}' quadrant, while {fund2_ticker} is in the '{fund2_quadrant}' quadrant."
    )
    
    pdf.add_image('comprehensive_plots/risk_return_quadrant.png', w=180, caption="Figure 6: Risk-Return Quadrant Analysis by Time Period")
    
    # Time Period Analysis
    pdf.add_page()
    pdf.chapter_title("5. Time Period Analysis")
    
    pdf.section_title("Performance by Time Period")
    
    # Period Analysis Table
    period_headers = ["Period", "Date Range", f"{fund1_ticker} Return", f"{fund2_ticker} Return", "Excess Return", "Rel. Vol."]
    period_data = []
    
    for period in ['Early', 'Middle', 'Late', 'Full']:
        result = period_results[period]
        period_data.append([
            period,
            result['date_range'],
            f"{result[f'{fund1_ticker}_return']:.2%}",
            f"{result[f'{fund2_ticker}_return']:.2%}",
            f"{result['excess_return']:.2%}",
            f"{result['relative_vol']:.2f}x"
        ])
    
    pdf.create_table(period_headers, period_data, [30, 40, 35, 35, 35, 25])
    pdf.ln(5)
    
    # Period Risk-Adjusted Performance Table
    period_risk_headers = ["Period", "Beta", "Correlation", f"{fund1_ticker} Sharpe", f"{fund2_ticker} Sharpe", "Alpha (Annualized)"]
    period_risk_data = []
    
    for period in ['Early', 'Middle', 'Late', 'Full']:
        result = period_results[period]
        # Calculate Sharpe ratios for each period
        period_fund1_sharpe = (result[f'{fund1_ticker}_annualized'] - (risk_free_rate * 12)) / result[f'{fund1_ticker}_std']
        period_fund2_sharpe = (result[f'{fund2_ticker}_annualized'] - (risk_free_rate * 12)) / result[f'{fund2_ticker}_std']
        
        period_risk_data.append([
            period,
            f"{result['beta']:.2f}",
            f"{result['correlation']:.2f}",
            f"{period_fund1_sharpe:.2f}",
            f"{period_fund2_sharpe:.2f}",
            f"{result['alpha_annualized']:.2%}"
        ])
    
    pdf.create_table(period_risk_headers, period_risk_data, [30, 32, 32, 32, 32, 42])
    pdf.ln(10)
    
    # Key observations about time periods
    pdf.section_title("Key Time Period Observations")
    
    # Calculate which periods had best/worst performance
    best_period_idx = np.argmax([period_results[p]['excess_return'] for p in ['Early', 'Middle', 'Late']])
    worst_period_idx = np.argmin([period_results[p]['excess_return'] for p in ['Early', 'Middle', 'Late']])
    best_period = ['Early', 'Middle', 'Late'][best_period_idx]
    worst_period = ['Early', 'Middle', 'Late'][worst_period_idx]
    
    period_observations = (
        f"- {fund1_ticker} {period_results[best_period]['excess_return'] > 0 and 'outperformed' or 'underperformed less'} "
        f"in the {best_period.lower()} period ({period_results[best_period]['date_range']})\n"
        
        f"- {fund1_ticker} {period_results[worst_period]['excess_return'] > 0 and 'still outperformed' or 'underperformed most significantly'} "
        f"in the {worst_period.lower()} period ({period_results[worst_period]['date_range']})\n"
        
        f"- The correlation between the funds has {period_results['Late']['correlation'] > period_results['Early']['correlation'] and 'increased' or 'decreased'} "
        f"from {period_results['Early']['correlation']:.2f} to {period_results['Late']['correlation']:.2f}\n"
        
        f"- Beta has {abs(period_results['Late']['beta']) > abs(period_results['Early']['beta']) and 'increased' or 'decreased'} "
        f"from {period_results['Early']['beta']:.2f} to {period_results['Late']['beta']:.2f}\n"
        
        f"- Relative volatility has {period_results['Late']['relative_vol'] > period_results['Early']['relative_vol'] and 'increased' or 'decreased'} "
        f"from {period_results['Early']['relative_vol']:.2f}x to {period_results['Late']['relative_vol']:.2f}x\n"
        
        f"- Alpha has {period_results['Late']['alpha_annualized'] > period_results['Early']['alpha_annualized'] and 'improved' or 'deteriorated'} "
        f"from {period_results['Early']['alpha_annualized']:.2%} to {period_results['Late']['alpha_annualized']:.2%}"
    )
    
    pdf.chapter_body(period_observations)
    
    # Market Environment Analysis
    pdf.add_page()
    pdf.chapter_title("6. Market Environment Analysis")
    
    pdf.section_title("Performance in Different Market Environments")
    
    # Market Environment Table
    env_headers = ["Market Environment", "% of Time", f"{fund1_ticker} Return", f"{fund2_ticker} Return", "Excess Return", "Rel. Vol.", "Beta"]
    env_data = []
    
    for env in ['up', 'neutral', 'down']:
        if env in risk_by_environment:
            result = risk_by_environment[env]
            env_data.append([
                env.capitalize() + " Market",
                f"{(result['count'] / total_periods):.1%}",
                f"{result[f'{fund1_ticker}_mean']:.2%}",
                f"{result[f'{fund2_ticker}_mean']:.2%}",
                f"{result['excess_return']:.2%}",
                f"{result['relative_vol']:.2f}x",
                f"{result['beta']:.2f}"
            ])
    
    pdf.create_table(env_headers, env_data, [35, 25, 30, 30, 30, 25, 25])
    pdf.ln(10)
    
    # Market Environment Observations
    up_market_outperformance = risk_by_environment['up']['excess_return'] > 0
    down_market_outperformance = risk_by_environment['down']['excess_return'] > 0
    
    env_observations = (
        f"- {fund1_ticker} {up_market_outperformance and 'outperforms' or 'underperforms'} in up markets "
        f"with {abs(risk_by_environment['up']['excess_return']):.2%} {'outperformance' if up_market_outperformance else 'underperformance'}\n"
        
        f"- {fund1_ticker} {down_market_outperformance and 'outperforms' or 'underperforms'} in down markets "
        f"with {abs(risk_by_environment['down']['excess_return']):.2%} {'outperformance' if down_market_outperformance else 'underperformance'}\n"
        
        f"- {fund1_ticker}'s beta is highest in {max(risk_by_environment.items(), key=lambda x: x[1]['beta'])[0]} markets "
        f"({risk_by_environment[max(risk_by_environment.items(), key=lambda x: x[1]['beta'])[0]]['beta']:.2f})\n"
        
        f"- Relative volatility is {max(risk_by_environment.items(), key=lambda x: x[1]['relative_vol'])[1]['relative_vol'] > 1 and 'consistently higher' or 'variable'} across market environments\n"
    )
    
    pdf.chapter_body(env_observations)
    
    # Investment Implications
    pdf.add_page()
    pdf.chapter_title("7. Investment Implications")
    
    # Determine if fund1 is preferred over fund2
    fund1_preferred = False
    
    # Simple model: if sharpe ratio and returns are higher, or if returns are similar but risk is much lower
    if (fund1_sharpe > fund2_sharpe and fund1_annual_return > fund2_annual_return) or \
       (abs(fund1_annual_return - fund2_annual_return) < 0.01 and fund1_vol < 0.8 * fund2_vol):
        fund1_preferred = True
    
    implications = (
        f"Based on the comprehensive analysis of {fund1_ticker} and {fund2_ticker}, the following investment implications emerge:\n\n"
        
        f"Risk-Return Profile:\n"
        f"- {fund1_ticker} offers {'higher' if fund1_annual_return > fund2_annual_return else 'lower'} returns with "
        f"{'higher' if fund1_vol > fund2_vol else 'lower'} risk compared to {fund2_ticker}\n"
        f"- {fund1_ticker} generates {alpha_annual > 0 and 'positive' or 'negative'} alpha ({alpha_annual:.2%} annualized)\n"
        f"- The risk-adjusted performance ({fund1_sharpe:.2f} vs {fund2_sharpe:.2f}) is {'superior' if fund1_sharpe > fund2_sharpe else 'inferior'} for {fund1_ticker}\n\n"
        
        f"Time Consistency:\n"
        f"- Performance has been {'consistent' if np.std([r['excess_return'] for r in period_results.values()]) < 0.1 else 'inconsistent'} across different time periods\n"
        f"- The {'early' if best_period == 'Early' else 'middle' if best_period == 'Middle' else 'late'} period showed the best relative performance\n"
        f"- The trend in relative performance is {'improving' if period_results['Late']['excess_return'] > period_results['Middle']['excess_return'] else 'deteriorating'}\n\n"
        
        f"Market Environment Suitability:\n"
        f"- {fund1_ticker} is {'better' if up_market_outperformance else 'worse'} suited for rising markets\n"
        f"- {fund1_ticker} {'provides' if down_market_outperformance else 'lacks'} downside protection in declining markets\n"
        f"- The fund is most appropriate for investors seeking {fund1_vol > fund2_vol and 'higher growth with increased volatility' or 'moderate growth with reduced volatility'}\n\n"
        
        f"Recommendation:\n"
        f"- For investors prioritizing {'risk-adjusted returns' if fund1_preferred else 'absolute performance'}, "
        f"{fund1_ticker if fund1_preferred else fund2_ticker} appears to be the more suitable option\n"
        f"- {fund1_ticker} {'should' if fund1_quadrant in ['Low Risk, High Return (Optimal)', 'High Risk, High Return'] else 'may not'} be considered for new investments at this time\n"
        f"- Existing holders of {fund2_ticker} {'should' if fund1_preferred else 'may not want to'} consider switching to {fund1_ticker}\n"
    )
    
    pdf.chapter_body(implications)
    
    # Add disclaimer
    pdf.add_page()
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Disclaimer', 0, 1, 'L')
    pdf.set_font('Arial', '', 8)
    
    disclaimer = (
        "This report is provided for informational purposes only and should not be considered an offer to buy or "
        "sell any securities. Past performance is not indicative of future results. The information contained herein "
        "has been obtained from sources believed to be reliable, but its accuracy cannot be guaranteed.\n\n"
        
        "The analysis in this report is based on historical data and is not intended to forecast future performance. "
        "Different market conditions may lead to different results in the future. Investors should consider their "
        "investment objectives, risk tolerance, and time horizon before making investment decisions.\n\n"
        
        "For investment professional use only. "
        "Not for distribution to the general public.\n\n"
        
        f"Report generated on {datetime.now().strftime('%B %d, %Y')}"
    )
    
    pdf.multi_cell(0, 5, disclaimer)
    
    # Save the PDF
    pdf.output(output_file, 'F')
    print(f"Comprehensive Analysis PDF report generated: {output_file}")
    
    # Return key metrics for reference
    return {
        'fund1_ticker': fund1_ticker,
        'fund2_ticker': fund2_ticker,
        'fund1_total_return': fund1_total_return,
        'fund2_total_return': fund2_total_return,
        'fund1_annual_return': fund1_annual_return,
        'fund2_annual_return': fund2_annual_return,
        'fund1_vol': fund1_vol,
        'fund2_vol': fund2_vol,
        'fund1_sharpe': fund1_sharpe,
        'fund2_sharpe': fund2_sharpe,
        'fund1_quadrant': fund1_quadrant,
        'fund2_quadrant': fund2_quadrant,
        'alpha_annual': alpha_annual,
        'beta': beta
    }

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Generate comprehensive fund comparison analysis.')
    parser.add_argument('--fund1', default='PGTIX', help='First fund ticker symbol (default: PGTIX)')
    parser.add_argument('--fund2', default='QQQ', help='Second fund ticker symbol (default: QQQ)')
    parser.add_argument('--start_date', help='Start date in YYYY-MM-DD format')
    parser.add_argument('--end_date', help='End date in YYYY-MM-DD format')
    parser.add_argument('--output', help='Output PDF filename')
    
    args = parser.parse_args()
    
    # Execute the analysis with command-line arguments
    result = analyze_funds(
        fund1_ticker=args.fund1,
        fund2_ticker=args.fund2,
        start_date=args.start_date,
        end_date=args.end_date,
        output_file=args.output
    )
    
    # Print summary
    print("\nAnalysis Summary:")
    print(f"Fund Comparison: {result['fund1_ticker']} vs {result['fund2_ticker']}")
    print(f"Total Return: {result['fund1_ticker']} ({result['fund1_total_return']:.2%}) vs {result['fund2_ticker']} ({result['fund2_total_return']:.2%})")
    print(f"Risk-Return Positioning: {result['fund1_ticker']} is in '{result['fund1_quadrant']}' quadrant")
    print(f"Alpha: {result['alpha_annual']:.2%} | Beta: {result['beta']:.2f}")
    print(f"Report saved to: {args.output if args.output else f'{args.fund1}_vs_{args.fund2}_Analysis.pdf'}") 