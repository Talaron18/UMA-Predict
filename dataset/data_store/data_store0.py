import os
import glob
import json
import re
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple

class FeatureRegistry:
    # map features with tensors
    def __init__(self, path: str):
        self.path = path
        self.feature_to_idx: Dict[str, int] = {}
        
    def register_features(self, features: List[str]):
        self.feature_to_idx = {feat: idx for idx, feat in enumerate(features)}
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, 'w', encoding='utf-8') as f:
            json.dump(self.feature_to_idx, f, indent=2, ensure_ascii=False)
            
    def get_idx(self, name: str) -> int:
        return self.feature_to_idx[name]


class StorageEngine:
    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self.metadata_dir = os.path.join(root_dir, "metadata")
        self.horse_dir = os.path.join(root_dir, "horse")
        self.race_dir = os.path.join(root_dir, "race")
        
        os.makedirs(self.metadata_dir, exist_ok=True)
        os.makedirs(self.horse_dir, exist_ok=True)
        os.makedirs(self.race_dir, exist_ok=True)

    def save_horse_data(self, horse_id: str, data: np.ndarray, dates: List[str]):
        path = os.path.join(self.horse_dir, str(horse_id))
        os.makedirs(path, exist_ok=True)
        data = data.astype(np.float32)
        fp = np.memmap(os.path.join(path, "features.bin"), dtype='float32', mode='w+', shape=data.shape)
        fp[:] = data[:]
        fp.flush()
        with open(os.path.join(path, "meta.json"), "w") as f:
            json.dump({"shape": list(data.shape)}, f)
        np.save(os.path.join(path, "dates.npy"), np.array(dates, dtype='datetime64[D]'))

    def save_race_tensor(self, race_id: str, data: np.ndarray):
        path = os.path.join(self.race_dir, f"{str(race_id)}.npy")
        np.save(path, data.astype(np.float32))

class QuantDataIngestionPipeline:
    def __init__(self, store_root: str):
        self.storage = StorageEngine(store_root)
        self.registry = FeatureRegistry(os.path.join(self.storage.metadata_dir, "feature_map.json"))
        # 15 basic features
        self.atomic_features = [
            "surface_code",       # 芝=0, 砂/ダート=1
            "direction_code",     # 右=0, 左=1, 直=2
            "distance",           # (m)
            "weather_code",       # 晴=0, 曇=1, 雨=2, 雪=3
            "condition_code",     # 良=0, 稍重=1, 重=2, 不良=3
            "gender_code",        # 牝=0, 牡=1, 騸/セ=2
            "age",
            "weight_carried",     # (kg)
            "time_seconds",       # (s))
            "last_3f",            # (s))
            "odds",
            "popularity",
            "horse_weight",       # (kg)
            "horse_weight_diff",  # (kg)
            "finish_rank"
        ]
        self.registry.register_features(self.atomic_features)

    @staticmethod
    def _parse_time(time_str: str) -> float:
        if pd.isna(time_str) or not isinstance(time_str, str) or ":" not in time_str:
            return 0.0
        try:
            parts = time_str.split(":")
            return float(parts[0]) * 60.0 + float(parts[1])
        except Exception:
            return 0.0

    @staticmethod
    def _parse_gender_age(ga_str: str) -> Tuple[float, float]:
        if pd.isna(ga_str) or not isinstance(ga_str, str) or len(ga_str) < 2:
            return -1.0, 0.0
        gender_char = ga_str[0]
        age_str = ga_str[1:]
        gender_map = {"牝": 0.0, "牡": 1.0, "セ": 2.0, "騸": 2.0}
        g_code = gender_map.get(gender_char, -1.0)
        try:
            age = float(re.findall(r'\d+', age_str)[0])
        except Exception:
            age = 0.0
        return g_code, age

    def clean_race_results(self, df: pd.DataFrame) -> pd.DataFrame:
        # turn original dataframe into Float32
        print(" Analyzing time and text...")
        surface_map = {"芝": 0.0, "ダート": 1.0, "ダ": 1.0}
        direction_map = {"右": 0.0, "左": 1.0, "直": 2.0}
        weather_map = {"晴": 0.0, "曇": 1.0, "雨": 2.0, "雪": 3.0, "曇り": 1.0}
        cond_map = {"良": 0.0, "稍重": 1.0, "重": 2.0, "不良": 3.0}
        df["surface_code"] = df["surface"].map(surface_map).fillna(-1.0).astype(np.float32)
        df["direction_code"] = df["direction"].map(direction_map).fillna(-1.0).astype(np.float32)
        df["weather_code"] = df["weather"].map(weather_map).fillna(-1.0).astype(np.float32)
        df["condition_code"] = df["condition"].map(cond_map).fillna(-1.0).astype(np.float32)
        df["time_seconds"] = df["time"].apply(self._parse_time).astype(np.float32)
        ga_res = df["gender_age"].apply(self._parse_gender_age)
        df["gender_code"] = [x[0] for x in ga_res]
        df["age"] = [x[1] for x in ga_res]
        df["finish_rank"] = df.groupby("race_id").cumcount() + 1
        num_cols = ["distance", "weight_carried", "last_3f", "odds", "popularity", "horse_weight", "horse_weight_diff", "finish_rank"]

        for col in num_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0).astype(np.float32)
            
        return df

    def run(self, races_dir: str, payoff_dir: str):
        race_files = glob.glob(os.path.join(races_dir, "*.parquet"))
        if not race_files:
            raise FileNotFoundError(f"Finding no .parquet in {races_dir} !")
        print(f"Finding {len(race_files)} Parquet files for race results. Start loading...")
        df_races = pd.concat([pd.read_parquet(f) for f in race_files], ignore_index=True)
        df_clean = self.clean_race_results(df_races)
        print("Generating Race-Level Store (Race-level Tensor)...")
        race_index_records = []
        grouped_race = df_clean.groupby("race_id")
        
        for race_id, group in grouped_race:
            race_tensor = group[self.atomic_features].values.astype(np.float32)
            self.storage.save_race_tensor(str(race_id), race_tensor)
            race_index_records.append({
                "race_id": str(race_id),
                "date": group["date"].iloc[0],
                "field_size": len(group)
            })
        print("Generating Horse-Level Store (Horse Tensor)...")
        df_clean = df_clean.sort_values(by="date")
        grouped_horse = df_clean.groupby("horse_id")
        horse_index_records = []

        for horse_id, group in grouped_horse:
            horse_tensor = group[self.atomic_features].values.astype(np.float32)
            dates = group["date"].astype(str).tolist()
            self.storage.save_horse_data(str(horse_id), horse_tensor, dates)
            horse_index_records.append({
                "horse_id": str(horse_id),
                "total_races": len(group)
            })
        print("Generating global index (horse_index / race_index)...")
        pd.DataFrame(race_index_records).to_parquet(os.path.join(self.storage.metadata_dir, "race_index.parquet"))
        pd.DataFrame(horse_index_records).to_parquet(os.path.join(self.storage.metadata_dir, "horse_index.parquet"))
        payoff_files = glob.glob(os.path.join(payoff_dir, "*.parquet"))
        if payoff_files:
            print(f"Finding {len(payoff_files)} parquet files for payoff. Loading data...")
            df_payoff = pd.concat([pd.read_parquet(f) for f in payoff_files], ignore_index=True)
            df_payoff.to_parquet(os.path.join(self.storage.metadata_dir, "payouts_ledger.parquet"))
        else:
            print("Finding no parquet files under /payoff, skipping payoff tensors")
        print("\n" + "="*40 + "\n Data Ingestion Finishing processing!\n" + "="*40)


if __name__ == "__main__":
    STORE_ROOT_PATH = "../data/store"
    RACES_RAW_DIR = "../data/races"
    PAYOFF_RAW_DIR = "../data/payoff"
    pipeline = QuantDataIngestionPipeline(store_root=STORE_ROOT_PATH)
    pipeline.run(races_dir=RACES_RAW_DIR, payoff_dir=PAYOFF_RAW_DIR)