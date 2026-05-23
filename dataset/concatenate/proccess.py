import pandas as pd
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Tuple, List, Dict
from tqdm import tqdm
from loguru import logger
import fire


class RaceDataDumper:
    def __init__(self, input_file: str = "raw_race_data.parquet", horse_seq_file: str = "horse_sequence.parquet", race_feature_file: str = "race_feature.parquet", max_workers: int = 16):
        self.input_file = Path(input_file).expanduser()
        self.horse_seq_file = Path(horse_seq_file).expanduser()
        self.race_feature_file = Path(race_feature_file).expanduser()
        self.max_workers = max_workers

    def _load_data(self) -> pd.DataFrame:
        logger.info(f"Loading data from {self.input_file}...")
        if self.input_file.is_dir():
            files = sorted(self.input_file.glob("*.parquet"))
            df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        else:
            df = pd.read_parquet(self.input_file)
        numeric_cols = ["last_3f", "odds", "popularity", "horse_weight", "horse_weight_diff", "weight_carried", "margin"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        if "time" in df.columns:
            def time_to_seconds(t):
                try:
                    if ':' in str(t):
                        m, s = str(t).split(':')
                        return int(m) * 60 + float(s)
                    return float(t)
                except:
                    return None
            df["time_secs"] = df["time"].apply(time_to_seconds)

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["horse_id", "race_id", "date"])

        # Keep one row per horse-race pair to prevent duplicate sequence samples.
        df = df.drop_duplicates(subset=["horse_id", "race_id"], keep="last")

        # Fill only numeric features; keep text columns as-is to avoid artificial categorical collisions.
        existing_numeric_cols = [c for c in numeric_cols if c in df.columns]
        if existing_numeric_cols:
            df[existing_numeric_cols] = df[existing_numeric_cols].fillna(0)

        sort_cols = ["horse_id", "date"]
        if "race_id" in df.columns:
            sort_cols.append("race_id")
        df = df.sort_values(sort_cols)
        
        return df

    @staticmethod
    def _process_horse_group(group: pd.DataFrame) -> Tuple[List[Dict], List[Dict]]:
        sort_cols = ["date"]
        if "race_id" in group.columns:
            sort_cols.append("race_id")
        group = group.sort_values(sort_cols).reset_index(drop=True)
        horse_sequences = []
        race_features = []

        for idx, row in group.iterrows():
            history = []
            # Use positional history to include earlier same-day races (after sorting),
            # which avoids many identical histories caused by date-only filtering.
            past_races = group.iloc[:idx]
            
            for _, past in past_races.iterrows():
                history.append({
                    "race_date": past["date"].strftime("%Y-%m-%d"),
                    "distance": past["distance"],
                    "surface": past["surface"],
                    "condition": past["condition"],
                    "weather": past["weather"],
                    "direction":past["direction"],
                    "jockey": past["jockey"],
                    "gender_age":past["gender_age"],
                    "weight_carried": past["weight_carried"],
                    "horse_weight": past["horse_weight"],
                    "horse_weight_diff": past["horse_weight_diff"],
                    "last_3f": past["last_3f"],
                    "passing": past["passing"],
                    "margin": past["margin"],
                    "odds": past["odds"],
                    "popularity": past["popularity"],
                    "time":past["time"]
                })

            horse_sequences.append({
                "sample_id": f"horse{row['horse_id']}_race{row['race_id']}",
                "horse_id": row["horse_id"],
                "race_id_target": row["race_id"],
                "history": history,
                "label": None, 
            })

            past_last3f = [h["last_3f"] for h in history if isinstance(h["last_3f"], (int, float)) and h["last_3f"] > 0]
            past_last3f_avg = np.mean(past_last3f) if past_last3f else 0.0
            past_margin_avg = np.mean([h["margin"] for h in history if pd.notna(h["margin"])]) if history else 0

            race_features.append({
                "race_id": row["race_id"],
                "horse_id": row["horse_id"],
                "distance": row["distance"],
                "surface": row["surface"],
                "condition": row["condition"],
                "direction":row["direction"],
                "gender_age":row["gender_age"],
                "weather": row["weather"],
                "jockey": row["jockey"],
                "trainer": row["trainer"],
                "owner": row["owner"],
                "horse_weight": row["horse_weight"],
                "horse_weight_diff": row["horse_weight_diff"],
                "past_last3f_avg": past_last3f_avg,
                "past_margin_avg": past_margin_avg,
                "predicted_rank": None,
                "odds": row["odds"],
                "popularity": row["popularity"],
            })

        return horse_sequences, race_features

    def dump(self):
        df = self._load_data()
        horse_sequences_all = []
        race_features_all = []
        groups = [g for _, g in df.groupby("horse_id")]
        
        logger.info("Start processing horse groups in parallel...")
        with tqdm(total=len(groups)) as p_bar:
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [executor.submit(self._process_horse_group, g) for g in groups]
                for future in as_completed(futures):
                    try:
                        seq, feat = future.result()
                        # Keep only the last (longest-history) sample per horse.
                        if seq:
                            horse_sequences_all.append(seq[-1])
                        if feat:
                            race_features_all.append(feat[-1])
                    except Exception as e:
                        logger.error(f"Error processing group: {e}")
                    finally:
                        p_bar.update()

        logger.info("End of processing. Start saving parquet files...")
        
        self.horse_seq_file.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(horse_sequences_all).to_parquet(self.horse_seq_file, index=False)
        logger.info(f"Saved {self.horse_seq_file.name}, {len(horse_sequences_all)} rows.")
        self.race_feature_file.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(race_features_all).to_parquet(self.race_feature_file, index=False)
        logger.info(f"Saved {self.race_feature_file.name}, {len(race_features_all)} rows.")

    def __call__(self, *args, **kwargs):
        self.dump()


if __name__ == "__main__":
    fire.Fire({"dump_race": RaceDataDumper})
