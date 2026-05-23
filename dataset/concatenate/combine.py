import pyarrow.parquet as pq
import glob
import os

input_pattern = "data/payoff/*.parquet"
output_path = "data/processed/merged_data.parquet"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
files = glob.glob(input_pattern)
if files:
    dataset = pq.ParquetDataset(files)
    table = dataset.read()
    
    pq.write_table(table, output_path)
    print(f"Saved at: {output_path}")
else:
    print("Invalid Path")