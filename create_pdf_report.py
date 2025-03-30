import os
from fpdf import FPDF
import matplotlib.pyplot as plt
from datetime import datetime

class PDF(FPDF):
    def header(self):
        # Add logo
        self.set_font('Arial', 'B', 15)
        self.set_text_color(0, 51, 102)  # Dark blue
        self.cell(0, 10, 'PGTIX Performance Analysis', 0, 1, 'C')
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

# Create PDF
pdf = PDF()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.add_page()

# Cover page
pdf.set_font('Arial', 'B', 24)
pdf.set_text_color(0, 51, 102)  # Dark blue
pdf.cell(0, 20, '', 0, 1, 'C')  # Space at top
pdf.cell(0, 20, 'Fund Ticker "PGTIX"', 0, 1, 'C')
pdf.cell(0, 10, 'Performance Analysis', 0, 1, 'C')

# Add a horizontal line
pdf.cell(0, 1, '', 0, 1)
pdf.line(30, pdf.get_y(), 180, pdf.get_y())
pdf.cell(0, 10, '', 0, 1)  # Space after line

# Add main visualization to cover
if os.path.exists('plots/cumulative_returns.png'):
    pdf.image('plots/cumulative_returns.png', x=30, y=None, w=150)
    
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
    ("2. Overall Performance Analysis", 4),
    ("3. Risk Metrics", 5),
    ("4. Consistency of Performance", 6),
    ("5. Performance by Time Period", 7),
    ("6. Rolling Performance Analysis", 8),
    ("7. Market Environment Analysis", 9),
    ("8. Key Conclusions", 10),
    ("9. Investment Implications", 11)
]

for item, page in toc_items:
    pdf.cell(0, 8, item, 0, 0)
    pdf.cell(0, 8, f'Page {page}', 0, 1, 'R')

# Executive Summary
pdf.add_page()
pdf.chapter_title("1. Executive Summary")
pdf.chapter_body(
    "This report provides a comprehensive analysis of the performance of PGTIX compared to the QQQ ETF "
    "from November 2016 to February 2025. The analysis evaluates returns, risk metrics, consistency, and "
    "performance across different time periods.\n\n"
    
    "Key Findings:\n"
    "- PGTIX has underperformed QQQ over the full period with total returns of 230.70% vs. 382.45%\n"
    "- PGTIX exhibits higher volatility with lower returns (Standard Deviation: 6.88% vs. 5.45%)\n"
    "- The underperformance becomes more consistent and pronounced over longer time horizons\n"
    "- The fund shows negative alpha across all time periods, suggesting structural underperformance\n"
    "- Performance characteristics appear to be persistent over time, with the fund struggling particularly "
    "during tech-driven bull markets\n\n"
    
    "This analysis suggests limited justification for including PGTIX in portfolios where QQQ is a viable alternative, "
    "especially for long-term investors."
)

# Overall Performance
pdf.add_page()
pdf.chapter_title("2. Overall Performance Analysis")

# Add performance metrics
pdf.section_title("Performance Metrics")
performance_data = [
    ["Metric", "PGTIX", "QQQ", "Difference"],
    ["Total Return", "230.70%", "382.45%", "-151.75%"],
    ["Annualized Return", "15.43%", "20.79%", "-5.35%"],
    ["Mean Monthly Return", "1.44%", "1.73%", "-0.29%"],
    ["Median Monthly Excess Return", "", "", "0.01%"]
]
pdf.create_table(performance_data[0], performance_data[1:], [40, 50, 50, 50])
pdf.ln(10)

# Add cumulative returns chart
pdf.section_title("Cumulative Returns")
pdf.add_image('plots/cumulative_returns.png', w=190, caption="Figure 1: Cumulative Returns of PGTIX vs. QQQ (Nov 2016 - Feb 2025)")

# Risk Metrics
pdf.add_page()
pdf.chapter_title("3. Risk Metrics")

# Add risk metrics table
pdf.section_title("Risk and Return Metrics")
risk_data = [
    ["Metric", "PGTIX", "QQQ"],
    ["Standard Deviation", "6.88%", "5.45%"],
    ["Sharpe Ratio", "0.21", "0.32"],
    ["Correlation", "0.88", "1.00"],
    ["Beta", "1.12", "1.00"],
    ["Alpha (Monthly)", "-0.48%", "0.00%"],
    ["Information Ratio", "-0.09", "N/A"],
    ["Tracking Error", "3.28%", "N/A"]
]
pdf.create_table(risk_data[0], risk_data[1:], [60, 65, 65])
pdf.ln(10)

# Add scatter plot
pdf.section_title("Return Comparison")
pdf.add_image('plots/returns_scatter.png', w=190, caption="Figure 2: Monthly Returns Scatter Plot with Regression Line")

# Add drawdown analysis
pdf.section_title("Drawdown Analysis")
pdf.add_image('plots/drawdown.png', w=190, caption="Figure 3: Drawdown Analysis - PGTIX vs. QQQ")

# Consistency of Performance
pdf.add_page()
pdf.chapter_title("4. Consistency of Performance")

pdf.section_title("Monthly Outperformance")
pdf.chapter_body(
    "- Periods where PGTIX outperformed QQQ: 50 out of 100 (50.0%)\n"
    "- Longest streak of consecutive outperformance: 8 months\n"
    "- Longest streak of consecutive underperformance: 8 months"
)
pdf.ln(5)

# Add monthly excess returns chart
pdf.add_image('plots/monthly_excess_returns.png', w=190, caption="Figure 4: Monthly Excess Returns (PGTIX - QQQ)")
pdf.ln(5)

# Add excess return distribution
pdf.section_title("Excess Return Distribution")
pdf.add_image('plots/excess_return_distribution.png', w=190, caption="Figure 5: Distribution of Monthly Excess Returns")

# Performance by Time Period
pdf.add_page()
pdf.chapter_title("5. Performance by Time Period")

# Add time period performance table
pdf.section_title("Performance in Different Time Periods")
time_data = [
    ["Period", "Date Range", "PGTIX Return", "QQQ Return", "Excess Return"],
    ["Early", "11/2016 - 07/2019", "65.89%", "65.22%", "0.67%"],
    ["Middle", "08/2019 - 04/2022", "21.89%", "66.79%", "-44.89%"],
    ["Late", "05/2022 - 02/2025", "63.54%", "75.08%", "-11.53%"],
    ["Full", "11/2016 - 02/2025", "230.70%", "382.45%", "-151.75%"]
]
pdf.create_table(time_data[0], time_data[1:], [30, 50, 40, 40, 40])
pdf.ln(10)

# Add calendar heatmap
pdf.section_title("Calendar Month Performance")
pdf.add_image('plots/calendar_heatmap.png', w=190, caption="Figure 6: Calendar Month Excess Returns Heatmap")
pdf.ln(5)

pdf.section_title("Key Time Period Observations")
pdf.chapter_body(
    "- PGTIX slightly outperformed QQQ in the early period (11/2016 - 07/2019) by 0.67%\n"
    "- Significant underperformance occurred in the middle period (08/2019 - 04/2022) by -44.89%\n"
    "- Underperformance continued in the late period (05/2022 - 02/2025) by -11.53%\n"
    "- The correlation between PGTIX and QQQ has increased slightly from 0.86 to 0.90 over time\n"
    "- Alpha deteriorated from -1.38% annually in the early period to -12.85% in the middle period\n"
    "- The middle period coincides with the COVID-19 pandemic and subsequent tech rally"
)

# Rolling Performance Analysis
pdf.add_page()
pdf.chapter_title("6. Rolling Performance Analysis")

# Add rolling window analysis table
pdf.section_title("Rolling Window Performance")
rolling_data = [
    ["Window", "Avg Excess Return", "% Positive Excess", "Avg Annualized Alpha", "% Time Outperforming"],
    ["12-Month", "-2.13%", "44.0%", "-6.72%", "33.0%"],
    ["24-Month", "-9.52%", "30.0%", "-7.25%", "29.0%"],
    ["36-Month", "-26.12%", "13.0%", "-8.13%", "19.0%"],
    ["60-Month", "-68.36%", "2.0%", "-8.00%", "7.0%"]
]
pdf.create_table(rolling_data[0], rolling_data[1:], [40, 40, 40, 40, 40])
pdf.ln(10)

# Add rolling excess returns chart
pdf.add_image('plots/rolling_excess_returns_comparison.png', w=190, caption="Figure 7: Rolling Excess Returns for Different Time Windows")
pdf.ln(5)

pdf.section_title("Rolling Performance Insights")
pdf.chapter_body(
    "- Longer time horizons show more consistent and pronounced underperformance\n"
    "- The percentage of time PGTIX outperforms QQQ drops dramatically from 33% (12-month) to 7% (60-month)\n"
    "- All rolling period measurements show negative average excess returns\n"
    "- Persistent negative alpha across all rolling periods suggests structural underperformance"
)

# Market Environment Analysis
pdf.add_page()
pdf.chapter_title("7. Market Environment Analysis")

# Add market capture data
pdf.section_title("Market Environment Performance")
pdf.chapter_body(
    "- Up-Market Capture Ratio: 0.95\n"
    "- Down-Market Capture Ratio: 1.10\n"
    "- Win/Loss Ratio: 0.79"
)
pdf.ln(10)

# Add return distribution chart
pdf.section_title("Return Distributions")
pdf.add_image('plots/return_distribution.png', w=190, caption="Figure 8: Distribution of Monthly Returns - PGTIX vs. QQQ")
pdf.ln(5)

pdf.section_title("Market Environment Insights")
pdf.chapter_body(
    "- PGTIX underperforms in up markets, capturing only 95% of QQQ's upside\n"
    "- PGTIX also underperforms in down markets, capturing 110% of QQQ's downside\n"
    "- This creates a lose-lose scenario where the fund provides neither superior returns in bull markets nor effective downside protection\n"
    "- The most significant underperformance occurred during the tech-driven bull market following the COVID-19 pandemic\n"
    "- The win/loss ratio of 0.79 indicates that when PGTIX outperforms, it does so by smaller margins than when it underperforms"
)

# Key Conclusions
pdf.add_page()
pdf.chapter_title("8. Key Conclusions")

pdf.chapter_body(
    "1. Performance vs Benchmark: PGTIX has underperformed QQQ over the full period, with the underperformance becoming more pronounced in recent years and over longer time horizons.\n\n"
    
    "2. Consistency: While PGTIX outperformed QQQ in 50% of individual months, longer-term performance (1-5 years) shows consistent underperformance. This suggests that the fund occasionally outperforms in the short term but struggles to maintain this advantage.\n\n"
    
    "3. Market Environments: PGTIX underperforms in both up markets (capture ratio 0.95) and down markets (capture ratio 1.10), providing neither superior returns in bull markets nor effective downside protection in bear markets.\n\n"
    
    "4. Skill vs Luck: The persistent negative alpha across all time periods (deteriorating from -1.38% to -12.85% before improving to -2.95% annually) and negative information ratios suggest structural rather than coincidental underperformance.\n\n"
    
    "5. Distinctiveness: PGTIX returns are somewhat distinct from QQQ with a correlation of 0.88 and beta of 1.12, but this distinctiveness has not translated into superior performance. The correlation has slightly increased over time (from 0.86 to 0.90), suggesting the fund is becoming more similar to its benchmark.\n\n"
    
    "6. Return Distribution: The overall performance differential results from consistent small differences rather than extreme outliers, though the middle period (08/2019 - 04/2022) shows a particularly severe performance gap.\n\n"
    
    "7. Time Period Consistency: The fundamental characteristics of underperformance hold across different time periods, especially for longer investment horizons. While there was slight outperformance in the early period, the long-term pattern of underperformance has been remarkably persistent."
)

# Investment Implications
pdf.add_page()
pdf.chapter_title("9. Investment Implications")

pdf.chapter_body(
    "1. Long-Term Investors: Based on the rolling window analysis, investors with a time horizon of 3+ years would have only a 7-19% chance of outperforming by choosing PGTIX over QQQ, making it difficult to justify an allocation to this fund for long-term investors.\n\n"
    
    "2. Higher Risk Without Reward: PGTIX consistently exhibits higher volatility than QQQ (overall 6.88% vs 5.45% monthly standard deviation) while delivering lower returns, creating an unfavorable risk-return profile.\n\n"
    
    "3. Lack of Diversification Benefit: With its relatively high correlation to QQQ (0.88 overall, increasing to 0.90 in recent periods) and negative alpha, PGTIX does not provide meaningful diversification benefits that would justify its inclusion in a portfolio alongside or in place of QQQ.\n\n"
    
    "4. Market Environment Considerations: The fund's worst relative performance occurred during the strong tech rally following the COVID-19 market downturn, suggesting it may continue to struggle in tech-led bull markets.\n\n"
    
    "5. Portfolio Recommendation: Based on this comprehensive analysis, investors seeking exposure to this market segment would likely be better served by utilizing QQQ rather than PGTIX. There appears to be limited justification for including PGTIX in portfolios where QQQ is a viable alternative."
)

# Add disclaimer page
pdf.add_page()
pdf.set_font('Arial', 'B', 12)
pdf.cell(0, 10, 'Important Disclosures', 0, 1, 'L')
pdf.ln(5)

pdf.set_font('Arial', '', 9)
pdf.multi_cell(0, 5, 
    "This report is provided for informational purposes only and should not be considered an offer to buy or "
    "sell any securities. Past performance is not indicative of future results. The information contained herein "
    "has been obtained from sources believed to be reliable, but its accuracy cannot be guaranteed.\n\n"
    
    "The analysis in this report is based on historical data and is not intended to forecast future performance. "
    "Different market conditions may lead to different results in the future. Investors should consider their "
    "investment objectives, risk tolerance, and time horizon before making investment decisions.\n\n"
    
    "PGTIX and QQQ are used for comparative purposes only and do not constitute recommendations to "
    "purchase or sell any security. Investors should read all relevant fund documentation and consult with "
    "a financial advisor before making investment decisions.\n\n"
    
    "Analysis prepared for investment professional use only. "
    "Not for distribution to the general public.\n\n"
    
    f"Report generated on {datetime.now().strftime('%B %d, %Y')}"
)

# Output the PDF
pdf.output('PGTIX_Performance_Analysis.pdf', 'F')
print("PDF report generated: PGTIX_Performance_Analysis.pdf") 