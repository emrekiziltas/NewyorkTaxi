from pathlib import Path
import pandas as pd

# Folder where your parquet files are stored
data_dir = Path("data/processed")

# File name
file_name = "yellow_taxi_2023_01_processed.parquet"

# Full path
file_path = data_dir / file_name
print("Full path to parquet file:", file_path)

# Step 1: Read the Parquet file
df = pd.read_parquet(file_path)  # use the full path

# Step 2: Randomly sample 100 rows
df_sample = df.sample(n=100, random_state=42)

# Step 3: Export to Excel
df_sample.to_excel("sample_100_rows.xlsx", index=False)
print("Sample of 100 rows saved to sample_100_rows.xlsx")
