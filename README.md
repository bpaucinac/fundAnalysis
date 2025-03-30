# Fund Analysis Framework

A comprehensive framework for analyzing and comparing investment funds, focusing on both risk and return metrics.

## Features

- Complete risk and return analysis in a single framework
- Risk-Return quadrant positioning (Low/High Risk, Low/High Return)
- Performance analysis across different market environments
- Time period breakdown and trend analysis
- Comprehensive PDF report generation with visualization
- Flexible command-line interface for comparing any funds

## Requirements

- Python 3.x
- Required packages: pandas, numpy, matplotlib, seaborn, scipy, fpdf

## Installation

1. Clone this repository
2. Install dependencies:
```
pip install pandas numpy matplotlib seaborn scipy fpdf
```

## Usage

### Basic Usage

Run the analysis with default values (PGTIX vs QQQ):

```
python create_comprehensive_analysis.py
```

### Advanced Usage

Compare specific funds:

```
python create_comprehensive_analysis.py --fund1 FUND1 --fund2 FUND2
```

Specify date range:

```
python create_comprehensive_analysis.py --start_date 2020-01-01 --end_date 2023-12-31
```

Custom output file:

```
python create_comprehensive_analysis.py --output MyAnalysisReport.pdf
```

### All Options

```
usage: create_comprehensive_analysis.py [-h] [--fund1 FUND1] [--fund2 FUND2] 
                                       [--start_date START_DATE]
                                       [--end_date END_DATE] [--output OUTPUT]

Generate comprehensive fund comparison analysis.

optional arguments:
  -h, --help            show this help message and exit
  --fund1 FUND1         First fund ticker symbol (default: PGTIX)
  --fund2 FUND2         Second fund ticker symbol (default: QQQ)
  --start_date START_DATE
                        Start date in YYYY-MM-DD format
  --end_date END_DATE   End date in YYYY-MM-DD format
  --output OUTPUT       Output PDF filename
```

## Report Components

The generated PDF report includes:

1. **Executive Summary**: Overview of key findings and metrics
2. **Performance Analysis**: Returns, consistency, and relative performance
3. **Risk Analysis**: Volatility, drawdowns, and risk-adjusted metrics
4. **Risk-Return Profile**: Quadrant analysis and positioning
5. **Time Period Analysis**: Performance breakdown and trends
6. **Market Environment Analysis**: Behavior in up, neutral, and down markets
7. **Investment Implications**: Conclusions and recommendations

## Key Visualizations

- Cumulative returns comparison
- Monthly returns scatter with regression line
- Rolling volatility
- Drawdown analysis
- Relative volatility over time
- Risk-Return quadrant positioning

## Data Format

The script expects a CSV file named `fundRetuns.csv` with the following columns:
- `tradeDate`: Date in MM/DD/YY format
- Fund columns with ticker names (e.g., `PGTIX`, `QQQ`)

## License

This project is licensed under the MIT License. 