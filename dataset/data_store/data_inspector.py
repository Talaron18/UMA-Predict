import os
import json
import numpy as np
import pandas as pd

def inspect_horse_bin_data(store_root: str, horse_id: str):
    horse_path = os.path.join(store_root, "horse", str(horse_id))
    meta_path = os.path.join(horse_path, "meta.json")
    bin_path = os.path.join(horse_path, "features.bin")
    dates_path = os.path.join(horse_path, "dates.npy")
    map_path = os.path.join(store_root, "metadata", "feature_map.json")

    if not os.path.exists(horse_path):
        print(f"WARNING: UNABLE TO LOCATE {horse_id} AT {horse_path}. Please double check the path")
        return

    with open(meta_path, "r") as f:
        meta = json.load(f)
    shape = tuple(meta["shape"])  

    with open(map_path, "r", encoding="utf-8") as f:
        feature_to_idx = json.load(f)
    columns = [None] * len(feature_to_idx)
    for name, idx in feature_to_idx.items():
        columns[idx] = name

    features_mmap = np.memmap(bin_path, dtype='float32', mode='r', shape=shape)
    dates = np.load(dates_path, mmap_mode='r')

    df_inspect = pd.DataFrame(np.array(features_mmap), columns=columns)
    df_inspect.insert(0, "race_date", dates) 

    print("=" * 60)
    print(f"Horse ID: {horse_id} bitstream visualize")
    print(f"Matrix (race_num, feature_num): {shape}")
    print("=" * 60)
    print("Reading bitstream from disk:")
    
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(df_inspect)
    print("=" * 60)

if __name__ == "__main__":
    STORE_ROOT = "../../data/datastore1"
    TARGET_HORSE = "2007103143" 
    inspect_horse_bin_data(store_root=STORE_ROOT, horse_id=TARGET_HORSE)