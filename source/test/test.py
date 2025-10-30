from pathlib import Path
import pandas as pd

# Directory containing your processed parquet files
processed_dir = Path(r"C:\Users\ek675\PycharmProjects\PythonProject\NewyorkTaxi\source\data\processed")

# Output folder for Excel samples
sample_dir = Path("samples")
sample_dir.mkdir(exist_ok=True)

parquet_files = [f.name for f in processed_dir.glob("*.parquet")]

# Optional: sort by name (for predictable order)
parquet_files.sort()

# <--- Initialize the list here! --->
excel_files = []

# Step 1: Create Excel samples from parquet files
for file_name in parquet_files:
    file_path = processed_dir / file_name
    df = pd.read_parquet(file_path)

    n_samples = min(1000, len(df))
    df_sample = df.sample(n=n_samples, random_state=42)

    excel_name = sample_dir / f"{file_name.replace('.parquet', '')}_sample.xlsx"
    df_sample.to_excel(excel_name, index=False)
    excel_files.append(excel_name)  # Now this works!

    print(f"Sample of {n_samples} rows saved to {excel_name}")

# Step 2: Merge all Excel samples into a single Excel file
merged_df = pd.concat([pd.read_excel(f) for f in excel_files], ignore_index=True)
merged_excel_path = sample_dir / "merged_samples.xlsx"
merged_df.to_excel(merged_excel_path, index=False)

print(f"\nAll samples merged into {merged_excel_path}")
print(f"Total rows in merged file: {len(merged_df)}")
