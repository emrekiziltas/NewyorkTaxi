# 🚕 NYC Taxi Data Pipeline & Analytics Dashboard

An end-to-end data pipeline and interactive analytics dashboard for NYC Taxi trip data. Built with Python, DuckDB, and Streamlit for efficient processing and visualization of millions of taxi trip records.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red.svg)
![DuckDB](https://img.shields.io/badge/DuckDB-Latest-yellow.svg)

## 🎯 Features

### Data Pipeline
- **Automated Data Ingestion**: Downloads NYC TLC trip data directly from official sources
- **Data Transformation**: Cleans and enriches raw data with derived features
- **Quality Checks**: Built-in data validation and quality reporting
- **Efficient Storage**: Processes and stores data in optimized Parquet format
- **Multi-Year Support**: Handles data from 2023-2024 (easily extensible)

### Interactive Dashboard
- **Real-Time Filtering**: Dynamic filters for time, fare, distance, and passenger count
- **Multiple Analytics Views**:
  - ⏰ Temporal Analysis (hourly, daily, weekly patterns)
  - 💰 Financial Insights (fare distribution, tips, revenue trends)
  - 🗺️ Trip Patterns (distance, passenger analysis)
  - 📊 Statistical Analysis (correlations, distributions)
  - 🔍 Deep Dive (custom queries, data export)
- **Interactive Visualizations**: Built with Plotly for responsive charts
- **Data Export**: Export filtered results to CSV, Excel, or Parquet
- **Custom SQL Queries**: Built-in query builder for advanced analysis

## 📁 Project Structure

```
Nyc-taxi-pipeline/
├── data/
│   ├── raw/              # Downloaded Parquet files from NYC TLC
│   └── processed/        # Cleaned and transformed data
├── src/
│   ├── data_ingestion.py # Data download and processing pipeline
│   └── dashboard.py      # Streamlit analytics dashboard
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- 10+ GB free disk space (for data storage)
- Internet connection (for initial data download)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/emrekiziltas/Nyc-taxi-pipeline.git
cd Nyc-taxi-pipeline
```

2. **Create virtual environment**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Usage

#### Step 1: Run Data Pipeline

Download and process NYC taxi data:

```bash
python src/data_ingestion.py
```

This will:
- Download Yellow Taxi trip data from NYC TLC
- Process data for 2023-2024 (all 12 months per year)
- Apply data cleaning and transformations
- Save processed files to `data/processed/`
- Merge all monthly files into a combined dataset

**Note**: Initial download may take 30-60 minutes depending on your internet speed. Processed files will be reused on subsequent runs.

#### Step 2: Launch Dashboard

Start the interactive analytics dashboard:

```bash
streamlit run src/dashboard.py
```

The dashboard will open in your browser at `http://localhost:8501`

## 📊 Dashboard Features

### KPI Metrics
- Total trips count
- Average fare amount
- Total revenue
- Average trip distance
- Average tip amount and percentage
- Average passenger count
- Revenue per trip

### Analysis Tabs

**⏰ Temporal Analysis**
- Hourly trip volume and revenue patterns
- Day-of-week comparison
- Peak hour identification

**💰 Financial Insights**
- Fare distribution histograms
- Tip percentage analysis
- Daily revenue trends

**🗺️ Trip Patterns**
- Distance distribution analysis
- Passenger count breakdown
- Trip characteristic patterns

**📊 Statistical Analysis**
- Comprehensive statistical summaries (min, max, median, quartiles)
- Correlation matrix between key variables
- Distribution analysis

**🔍 Deep Dive**
- Custom SQL query builder
- Pre-built query templates
- Data export functionality (CSV/Excel/Parquet)

### Filtering Options

- **Time Filters**: Hour of day, day of week
- **Financial Filters**: Fare range, minimum tip percentage
- **Trip Filters**: Distance range, passenger count
- **Date Selection**: Specific months, date ranges, or all data

## 🛠️ Technical Details

### Data Pipeline (`data_ingestion.py`)

**Key Components:**
- `NYCTaxiDataIngestion`: Main class for data processing
  - `download_parquet_file()`: Downloads data from NYC TLC
  - `basic_transformation()`: Cleans and enriches data
  - `basic_data_quality_checks()`: Validates data quality
  - `process_month()`: Orchestrates monthly data processing
- `merge_parquet_files()`: Combines monthly files into yearly datasets

**Data Transformations:**
- Datetime parsing and feature extraction (hour, day, weekday, month, year)
- Negative value removal (fares, distances, tips)
- Data type optimization
- Quality validation

### Dashboard (`dashboard.py`)

**Technology Stack:**
- **Streamlit**: Web application framework
- **DuckDB**: In-memory analytical database (no setup required!)
- **Plotly**: Interactive visualizations
- **Pandas**: Data manipulation

**Performance Features:**
- Cached database connections
- Efficient Parquet file reading
- Query optimization with DuckDB
- Memory-efficient data loading

## 📦 Dependencies

```txt
streamlit>=1.20.0
duckdb>=0.9.0
pandas>=1.5.0
plotly>=5.14.0
requests>=2.28.0
openpyxl>=3.1.0  # Optional, for Excel export
```

## 🔧 Configuration

### Change Data Directory

Edit the `DATA_DIR` path in `dashboard.py`:

```python
DATA_DIR = Path("your/custom/path/data/processed")
```

### Customize Data Range

Modify the year range in `data_ingestion.py`:

```python
for year in range(2023, 2025):  # Change years here
    for month in range(1, 13):  # Or specific months
        # ...
```

### Taxi Type Selection

Download different taxi types (Yellow, Green, FHV):

```python
output_path = ingestion.process_month(
    year=2024, 
    month=1, 
    taxi_type='green'  # Change to 'green' or 'fhv'
)
```

## 📈 Sample Insights

The pipeline enables analysis like:

- **Peak Hours**: Identify busiest hours for taxi demand
- **Revenue Optimization**: Analyze high-revenue periods and locations
- **Tipping Patterns**: Understand tipping behavior across different scenarios
- **Distance Analysis**: Profile typical trip distances and outliers
- **Weekly Trends**: Compare weekday vs weekend patterns

## 🐛 Troubleshooting

### "No data files found" Error
```bash
# Run the data pipeline first
python src/data_ingestion.py
```

### Memory Issues
- Process fewer months at a time
- Increase available RAM
- Use the built-in filters to reduce data volume

### Download Failures
- Check internet connection
- Verify NYC TLC data availability for specific months
- Some months may not be available yet for recent dates

## 🤝 Contributing

Contributions are welcome! Feel free to:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -m 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Open a Pull Request

## 📝 Data Source

Data is sourced from the official [NYC Taxi & Limousine Commission](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page).

**Dataset includes:**
- Pickup/dropoff dates and times
- Trip distances
- Fare amounts, tips, tolls, and total amounts
- Payment types
- Passenger counts
- Rate codes

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 👤 Author

**Emre Kiziltas**

- GitHub: [@emrekiziltas](https://github.com/emrekiziltas)
- Project: [NYC Taxi Pipeline](https://github.com/emrekiziltas/Nyc-taxi-pipeline)

## 🙏 Acknowledgments

- NYC Taxi & Limousine Commission for providing open data
- Streamlit team for the excellent dashboard framework
- DuckDB team for the blazing-fast analytics database

---

**⭐ Star this repository if you find it helpful!**
