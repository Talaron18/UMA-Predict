import pandas as pd
base_file = "./data/store/metadata/horse_index.parquet"

target_files = [
    "horse_perfect_final.parquet",
    "horse_perfect_from_row_0.parquet",
    "horse_perfect_from_row_2528.parquet",
    "horse_perfect_from_row_5258.parquet",
    "horse_perfect_from_row_7836.parquet",
    "horse_perfect_from_row_11374.parquet"
]

print("reading index...")
try:
    base_df = pd.read_parquet(base_file)
    all_horse_ids = set(base_df['horse_id'].dropna().unique())
    print(f"There are {len(all_horse_ids)} unique horse_ids.")
except Exception as e:
    print(f"Failed to find the path: {e}")
    exit()

covered_horse_ids = set()

print("\nStart scanniing...")
for file in target_files:
    try:
        df = pd.read_parquet(file, columns=['horse_id'])
        file_ids = df['horse_id'].dropna().unique()
        covered_horse_ids.update(file_ids)
        print(f"Finishing {file}, involving {len(file_ids)} unique IDs。")
    except FileNotFoundError:
        print(f"Failed to find: {file}.")
    except Exception as e:
        print(f"Error reading {file}: {e}")

missing_ids = all_horse_ids - covered_horse_ids

print("\n" + "="*30)
if len(missing_ids) == 0:
    print("Check finished, checking all horse_id.")
else:
    print(f"Finishing checking: {len(missing_ids)} are missed.")
    print("missing horse_id list")
    for idx, missing_id in enumerate(list(missing_ids)[:100], 1):
        print(f"  {idx}. {missing_id}")
    
    if len(missing_ids) > 100:
        print(f"  ... there are still {len(missing_ids) - 100} IDs missing")
    # print the missing list if necessary
    # pd.Series(list(missing_ids)).to_csv("missing_horse_ids.csv", index=False, header=["missing_horse_id"])