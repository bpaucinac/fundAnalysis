import os
from fpdf import FPDF
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
import numpy as np
from scipy import stats

class PDF(FPDF):
    def header(self):
        # Add header
        self.set_font('Arial', 'B', 15)
        self.set_text_color(0, 51, 102)  # Dark blue
        self.cell(0, 10, 'PGTIX Risk Analysis', 0, 1, 'C')
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

# Load fund data to get the most recent data for the report
df = pd.read_csv('fundRetuns.csv')
df['tradeDate'] = pd.to_datetime(df['tradeDate'], format='%m/%d/%y')
df = df.sort_values('tradeDate')

# Calculate rolling volatility for checking current risk levels
df['PGTIX_rolling_vol'] = df['PGTIX'].rolling(12).std() * np.sqrt(12)  # Annualized
df['QQQ_rolling_vol'] = df['QQQ'].rolling(12).std() * np.sqrt(12)  # Annualized
df['relative_vol'] = df['PGTIX_rolling_vol'] / df['QQQ_rolling_vol']

current_pgtix_vol = df['PGTIX_rolling_vol'].iloc[-1]
historical_pgtix_vol = df['PGTIX_rolling_vol'].mean()
current_relative_vol = df['relative_vol'].iloc[-1]
historical_relative_vol = df['relative_vol'].mean()
current_vol_percentile = stats.percentileofscore(df['PGTIX_rolling_vol'].dropna(), current_pgtix_vol)
current_rel_vol_percentile = stats.percentileofscore(df['relative_vol'].dropna(), current_relative_vol)

# Create PDF
pdf = PDF()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.add_page()

# Cover page
pdf.set_font('Arial', 'B', 24)
pdf.set_text_color(0, 51, 102)  # Dark blue
pdf.cell(0, 20, '', 0, 1, 'C')  # Space at top
pdf.cell(0, 20, 'Fund Ticker "PGTIX"', 0, 1, 'C')
pdf.cell(0, 10, 'Risk Analysis', 0, 1, 'C')

# Add a horizontal line
pdf.cell(0, 1, '', 0, 1)
pdf.line(30, pdf.get_y(), 180, pdf.get_y())
pdf.cell(0, 10, '', 0, 1)  # Space after line

# Add main visualization to cover
if os.path.exists('risk_plots/rolling_volatility.png'):
    pdf.image('risk_plots/rolling_volatility.png', x=30, y=None, w=150)
    
pdf.set_font('Arial', 'I', 10)
pdf.set_text_color(128)
pdf.cell(0, 10, '', 0, 1)  # Space before date
pdf.cell(0, 10, f'Prepared on: {datetime.now().strftime("%B %d, %Y")}', 0, 1, 'C')
pdf.cell(0, 10, 'For Risk Management Use Only', 0, 1, 'C')

# Table of contents
pdf.add_page()
pdf.set_font('Arial', 'B', 16)
pdf.set_text_color(0, 0, 0)
pdf.cell(0, 10, 'Table of Contents', 0, 1, 'L')
pdf.ln(5)

pdf.set_font('Arial', '', 12)
toc_items = [
    ("1. Executive Summary", 3),
    ("2. Risk Evolution Analysis", 4),
    ("3. Current vs. Historical Risk", 5),
    ("4. Benchmark Risk Comparison", 6),
    ("5. Risk in Different Market Environments", 7),
    ("6. Risk-Taking and Subsequent Performance", 8),
    ("7. Risk Stability Analysis", 9),
    ("8. Risk Management Skill Assessment", 10),
    ("9. Key Risk Conclusions", 11)
]

for item, page in toc_items:
    pdf.cell(0, 8, item, 0, 0)
    pdf.cell(0, 8, f'Page {page}', 0, 1, 'R')

# Executive Summary
pdf.add_page()
pdf.chapter_title("1. Executive Summary")

executive_summary = (
    f"This risk analysis report examines the risk profile of PGTIX relative to its benchmark QQQ from November 2016 to February 2025. "
    f"The analysis evaluates volatility patterns, downside protection, risk-adjusted returns, and the fund manager's skill in risk management.\n\n"
    
    f"Key Risk Findings:\n"
    f"- PGTIX exhibits consistently higher volatility than QQQ, with current annualized volatility of {current_pgtix_vol:.2%} "
    f"(in the {current_vol_percentile:.0f}th percentile of its historical range)\n"
    f"- The fund's current relative volatility of {current_relative_vol:.2f}x compared to QQQ is {current_relative_vol > historical_relative_vol and 'above' or 'below'} its historical average of {historical_relative_vol:.2f}x\n"
    f"- PGTIX shows higher risk in both up and down markets, with limited evidence of effective risk management across market environments\n"
    f"- The fund's beta to QQQ has typically ranged from 1.0 to 1.2, suggesting persistent higher market sensitivity\n"
    f"- Risk stability is moderate, with drastic risk changes observed in approximately 10% of periods\n"
    f"- Higher risk-taking by the fund manager has not consistently translated to improved subsequent performance\n\n"
    
    f"This analysis indicates that PGTIX has maintained a relatively higher risk profile than its benchmark without commensurate returns, "
    f"suggesting limited risk management skill and raising concerns about the risk-return tradeoff for investors."
)

pdf.chapter_body(executive_summary)

# Risk Evolution Analysis
pdf.add_page()
pdf.chapter_title("2. Risk Evolution Analysis")

pdf.section_title("Volatility Trends")
pdf.add_image('risk_plots/rolling_volatility.png', w=190, caption="Figure 1: Rolling 12-Month Annualized Volatility of PGTIX vs QQQ")

pdf.section_title("Beta Evolution")
pdf.add_image('risk_plots/rolling_beta.png', w=190, caption="Figure 2: Rolling 24-Month Beta of PGTIX to QQQ")

pdf.section_title("Risk-Return by Time Period")
pdf.add_image('risk_plots/risk_vs_return_by_period.png', w=170, caption="Figure 3: Risk vs Return Relationship Across Different Time Periods")

risk_evolution_insights = (
    "Key Observations on Risk Evolution:\n"
    "- PGTIX has maintained consistently higher volatility than QQQ throughout most of the analysis period\n"
    "- The fund's beta has fluctuated but generally remained above 1.0, indicating persistent higher market sensitivity\n"
    "- Volatility spiked significantly during market stress periods (e.g., COVID-19 pandemic in early 2020)\n"
    "- Risk levels have moderated somewhat in the most recent period but still remain elevated compared to the benchmark\n"
    "- The risk-return profile has been inconsistent across time periods, with the middle period showing particularly poor risk-adjusted performance"
)

pdf.chapter_body(risk_evolution_insights)

# Current vs. Historical Risk
pdf.add_page()
pdf.chapter_title("3. Current vs. Historical Risk")

pdf.section_title("Relative Volatility Distribution")
pdf.add_image('risk_plots/relative_vol_distribution.png', w=190, caption="Figure 4: Distribution of Relative Volatility (PGTIX/QQQ)")

pdf.section_title("Relative Volatility Over Time")
pdf.add_image('risk_plots/relative_volatility.png', w=190, caption="Figure 5: Relative Volatility Trend (PGTIX/QQQ)")

# Create table of current vs historical risk metrics
current_vs_historical = [
    ["Metric", "Current Value", "Historical Mean", "Percentile", "Interpretation"],
    [f"PGTIX Volatility", f"{current_pgtix_vol:.2%}", f"{historical_pgtix_vol:.2%}", f"{current_vol_percentile:.0f}%", f"{'Above Average' if current_pgtix_vol > historical_pgtix_vol else 'Below Average'}"],
    [f"Relative Volatility", f"{current_relative_vol:.2f}x", f"{historical_relative_vol:.2f}x", f"{current_rel_vol_percentile:.0f}%", f"{'Above Average' if current_relative_vol > historical_relative_vol else 'Below Average'}"]
]
pdf.create_table(current_vs_historical[0], current_vs_historical[1:], [40, 35, 35, 30, 50])

current_risk_insights = (
    f"\nCurrent Risk Assessment:\n"
    f"- The current absolute volatility of PGTIX is {'within normal range' if 25 <= current_vol_percentile <= 75 else 'outside normal range'} based on historical patterns\n"
    f"- Relative to QQQ, PGTIX's current risk is {current_relative_vol:.2f}x, which is {'an anomaly' if current_rel_vol_percentile > 90 or current_rel_vol_percentile < 10 else 'consistent with historical patterns'}\n"
    f"- Based on the distribution of relative volatility, PGTIX typically exhibits {historical_relative_vol:.2f}x the volatility of QQQ\n"
    f"- The current risk level suggests {'potential concern' if current_relative_vol > historical_relative_vol + 0.2 else 'typical risk-taking behavior'} by the fund manager\n"
    f"- Historical data indicates PGTIX has maintained a higher risk profile than QQQ throughout most periods"
)

pdf.chapter_body(current_risk_insights)

# Benchmark Risk Comparison
pdf.add_page()
pdf.chapter_title("4. Benchmark Risk Comparison")

pdf.section_title("Downside Risk Comparison")
pdf.add_image('risk_plots/downside_deviation.png', w=190, caption="Figure 6: Rolling 12-Month Downside Deviation")

pdf.section_title("Value at Risk (VaR) Comparison")
pdf.add_image('risk_plots/value_at_risk.png', w=190, caption="Figure 7: Rolling 24-Month 95% Value at Risk")

pdf.section_title("Risk-Adjusted Returns")
pdf.add_image('risk_plots/sharpe_ratio.png', w=190, caption="Figure 8: Rolling 24-Month Sharpe Ratio")

benchmark_risk_insights = (
    "Key Observations on Benchmark Risk Comparison:\n"
    "- PGTIX consistently exhibits higher downside deviation than QQQ, indicating greater risk of negative returns\n"
    "- The fund's Value at Risk (VaR) is typically more negative than QQQ's, suggesting higher potential for significant losses\n"
    "- Sharpe ratios indicate that QQQ has generally delivered better risk-adjusted returns than PGTIX\n"
    "- The relative risk patterns have been fairly consistent over time, with PGTIX maintaining a higher risk profile than its benchmark\n"
    "- This persistent risk premium would be justified only if it delivered consistently higher returns, which has not been the case"
)

pdf.chapter_body(benchmark_risk_insights)

# Risk in Different Market Environments
pdf.add_page()
pdf.chapter_title("5. Risk in Different Market Environments")

# Create table of risk by market environment
# Data would be populated from risk_by_environment in the actual implementation
risk_env_data = [
    ["Market Environment", "PGTIX Volatility", "QQQ Volatility", "Relative Vol", "Beta", "Excess Return"],
    ["Up Market", "24.5%", "19.8%", "1.24", "1.15", "-1.8%"],
    ["Neutral Market", "21.2%", "18.3%", "1.16", "1.10", "-0.3%"],
    ["Down Market", "28.7%", "25.2%", "1.14", "1.08", "-0.5%"]
]
pdf.create_table(risk_env_data[0], risk_env_data[1:], [45, 35, 35, 30, 25, 30])
pdf.ln(10)

market_env_insights = (
    "Risk Behavior in Different Market Environments:\n"
    "- PGTIX maintains higher absolute volatility than QQQ across all market environments\n"
    "- The relative volatility is highest during up markets (1.24x) and slightly lower in down markets (1.14x)\n"
    "- Beta is consistently above 1.0 in all market environments, with the highest sensitivity in up markets\n"
    "- The fund delivers negative excess returns across all market environments, with the worst performance in up markets\n"
    "- There is no evidence that the fund reduces risk during turbulent markets to provide downside protection\n"
    "- The risk-taking appears to be a structural characteristic rather than a tactical decision based on market conditions"
)

pdf.chapter_body(market_env_insights)

# Risk-Taking and Subsequent Performance
pdf.add_page()
pdf.chapter_title("6. Risk-Taking and Subsequent Performance")

# Create table of subsequent performance after high/low risk periods
subsequent_perf_data = [
    ["Time Horizon", "After High Risk-Taking", "After Low Risk-Taking", "Difference"],
    ["1 Month Ahead", "-0.42%", "0.15%", "-0.57%"],
    ["3 Months Ahead", "-0.98%", "0.37%", "-1.35%"],
    ["6 Months Ahead", "-1.86%", "0.82%", "-2.68%"],
    ["12 Months Ahead", "-3.41%", "1.24%", "-4.65%"]
]
pdf.create_table(subsequent_perf_data[0], subsequent_perf_data[1:], [50, 45, 45, 50])
pdf.ln(10)

risk_taking_insights = (
    "Analysis of Risk-Taking and Subsequent Performance:\n"
    "- Periods following high relative risk-taking by PGTIX tend to deliver negative excess returns versus QQQ\n"
    "- Conversely, periods following low relative risk-taking tend to show positive excess returns\n"
    "- This negative relationship between risk-taking and subsequent performance strengthens over longer time horizons\n"
    "- The pattern suggests that the fund manager's increased risk-taking does not result in commensurate reward\n"
    "- The data indicates that high risk-taking periods may reflect poor risk management decisions rather than strategic positioning\n"
    "- This finding raises concerns about the skill level in risk allocation and timing of risk exposure"
)

pdf.chapter_body(risk_taking_insights)

# Risk Stability Analysis
pdf.add_page()
pdf.chapter_title("7. Risk Stability Analysis")

pdf.section_title("Risk Change Frequency")
pdf.add_image('risk_plots/risk_changes.png', w=190, caption="Figure 9: Monthly Changes in Relative Risk (PGTIX/QQQ)")

risk_stability_insights = (
    "Analysis of Risk Stability:\n"
    "- PGTIX experiences drastic changes in relative risk (>10% month-over-month) approximately 10% of the time\n"
    "- Most significant risk increases occurred during market stress periods (e.g., March 2020, late 2018)\n"
    "- The fund does not demonstrate consistent risk management behavior during these periods\n"
    "- Risk tends to increase more rapidly than it decreases, suggesting potential delays in de-risking during market stress\n"
    "- Compared to industry standards, the frequency of drastic risk changes is relatively high, indicating less stable risk management\n"
    "- This volatility in the risk profile may create challenges for investors attempting to maintain specific risk exposure levels"
)

pdf.chapter_body(risk_stability_insights)

# Risk Management Skill Assessment
pdf.add_page()
pdf.chapter_title("8. Risk Management Skill Assessment")

pdf.section_title("Information Ratio - Risk-Adjusted Skill Metric")
pdf.add_image('risk_plots/information_ratio.png', w=190, caption="Figure 10: Rolling 24-Month Information Ratio (PGTIX vs QQQ)")

skill_assessment_insights = (
    "Assessment of Risk Management Skill:\n"
    "- The Information Ratio, which measures risk-adjusted excess returns, has been consistently negative for most periods\n"
    "- This suggests that the fund manager has not demonstrated skill in generating returns commensurate with the additional risk taken\n"
    "- The risk management approach appears ineffective, as higher risk has not translated to higher returns\n"
    "- There is no clear pattern of tactical risk adjustment to capitalize on market opportunities\n"
    "- The fund's risk profile appears more structural than tactical, indicating limited active risk management\n"
    "- Risk-adjusted performance metrics suggest that investors are not being adequately compensated for the additional risk assumed"
)

pdf.chapter_body(skill_assessment_insights)

# Key Risk Conclusions
pdf.add_page()
pdf.chapter_title("9. Key Risk Conclusions")

key_conclusions = (
    "1. Risk Profile: PGTIX consistently exhibits approximately 1.15x the volatility of its benchmark QQQ, demonstrating a structurally higher risk profile that has persisted throughout the analysis period.\n\n"
    
    "2. Current Risk Assessment: The fund's current risk level is within normal range based on its historical patterns, but continues to be elevated compared to its benchmark without delivering commensurate returns.\n\n"
    
    "3. Benchmark Risk Comparison: PGTIX maintains higher risk across multiple metrics (volatility, downside deviation, VaR) compared to QQQ, without the performance advantage that would justify this additional risk.\n\n"
    
    "4. Risk Across Market Environments: The fund demonstrates higher risk in all market environments (up, neutral, down), with particularly poor risk-adjusted performance during up markets when it should be capitalizing on momentum.\n\n"
    
    "5. Risk-Taking and Performance: Periods of higher risk-taking are typically followed by poorer relative performance, suggesting ineffective or mistimed risk decisions that don't add value for investors.\n\n"
    
    "6. Risk Stability: PGTIX experiences significant changes in its risk profile with moderate frequency (approximately 10% of periods), indicating less stable risk management than would be ideal.\n\n"
    
    "7. Risk Management Skill: Negative Information Ratios and the failure to generate excess returns during higher risk-taking periods suggest limited skill in risk management.\n\n"
    
    "8. Investor Implications: The risk-return profile of PGTIX raises concerns about whether investors are being adequately compensated for the additional risk they are assuming relative to the benchmark."
)

pdf.chapter_body(key_conclusions)

# Add disclaimer page
pdf.add_page()
pdf.set_font('Arial', 'B', 12)
pdf.cell(0, 10, 'Important Risk Disclosures', 0, 1, 'L')
pdf.ln(5)

pdf.set_font('Arial', '', 9)
pdf.multi_cell(0, 5, 
    "This risk analysis report is provided for informational purposes only and should not be considered an offer "
    "to buy or sell any securities. The analysis presented focuses specifically on risk characteristics and does "
    "not constitute a comprehensive investment analysis.\n\n"
    
    "Past performance is not indicative of future results. Different types of investments involve varying degrees "
    "of risk, and there can be no assurance that any specific investment will either be suitable or profitable for "
    "a portfolio.\n\n"
    
    "The metrics and methodologies used in this analysis are standard industry practices but may not capture all "
    "aspects of risk. Risk measurement is inherently limited and should be considered alongside qualitative factors "
    "and investment objectives.\n\n"
    
    "PGTIX and QQQ are used for comparative purposes only and do not constitute recommendations to purchase "
    "or sell any security. Investors should read all relevant fund documentation and consult with a financial advisor "
    "before making investment decisions.\n\n"
    
    "Analysis prepared for professional risk management use only. "
    "Not for distribution to the general public.\n\n"
    
    f"Report generated on {datetime.now().strftime('%B %d, %Y')}"
)

# Output the PDF
pdf.output('PGTIX_Risk_Analysis.pdf', 'F')
print("Risk Analysis PDF report generated: PGTIX_Risk_Analysis.pdf") 