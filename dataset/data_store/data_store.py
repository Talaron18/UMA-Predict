import os
import glob
import json
import re
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple

class FeatureRegistry:
    """Feature registry with persistent metadata."""
    def __init__(self, path: str):
        self.path = path
        self.feature_to_idx: Dict[str, int] = {}

    def register_features(self, features: List[str]):
        self.feature_to_idx = {feat: idx for idx, feat in enumerate(features)}
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.feature_to_idx, f, indent=2, ensure_ascii=False)
        print(f"[FeatureRegistry] Registered {len(features)} features")

    def get_idx(self, name: str) -> int:
        return self.feature_to_idx[name]

    def load(self):
        if os.path.exists(self.path):
            with open(self.path, "r", encoding="utf-8") as f:
                self.feature_to_idx = json.load(f)


class StorageEngine:
    """Tensor storage backend."""
    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self.metadata_dir = os.path.join(root_dir, "metadata")
        self.horse_dir = os.path.join(root_dir, "horse")
        self.race_dir = os.path.join(root_dir, "race")
        self.payoff_dir = os.path.join(root_dir, "payoff")

        for d in [self.metadata_dir, self.horse_dir, self.race_dir, self.payoff_dir]:
            os.makedirs(d, exist_ok=True)

    def save_horse_data(self, horse_id: str, data: np.ndarray, dates: List[str]):
        path = os.path.join(self.horse_dir, str(horse_id))
        os.makedirs(path, exist_ok=True)
        data = data.astype(np.float32)

        fp = np.memmap(os.path.join(path, "features.bin"), dtype="float32", mode="w+", shape=data.shape)
        fp[:] = data[:]
        fp.flush()

        with open(os.path.join(path, "meta.json"), "w") as f:
            json.dump({
                "shape": list(data.shape),
                "dtype": "float32",
                "num_features": data.shape[1] if len(data.shape) > 1 else 1
            }, f, indent=2)

        np.save(os.path.join(path, "dates.npy"), np.array(dates, dtype="datetime64[D]"))

    def save_race_tensor(self, race_id: str, data: np.ndarray):
        np.save(os.path.join(self.race_dir, f"{race_id}.npy"), data.astype(np.float32))

    def save_payoff_tensor(self, race_id: str, data: np.ndarray):
        np.save(os.path.join(self.payoff_dir, f"{race_id}.npy"), data.astype(np.float32))


class QuantDataIngestionPipeline:
    """Horse racing datastore builder."""
    def __init__(self, store_root: str):
        self.storage = StorageEngine(store_root)
        self.registry = FeatureRegistry(os.path.join(self.storage.metadata_dir, "feature_map.json"))
        
        self.atomic_features = [
            "surface_code", "direction_code", "distance", "weather_code", "condition_code",
            "gender_code", "age", "weight_carried", "time_seconds", "last_3f", "odds",
            "popularity", "horse_weight", "horse_weight_diff", "finish_rank", "bracket_num",
            "horse_num", "num_horses", "margin", "pace_first_half", "pace_second_half",
            "pace_diff", "prize", "career_races", "career_wins"
        ]
        self.registry.register_features(self.atomic_features)

        self.venue_to_direction = {
            "札幌": 0, "函館": 0, "福島": 0, "中山": 0, "阪神": 0, "小倉": 0,
            "新潟": 1, "東京": 1, "中京": 1, "京都": 1, "新潟直": 2
        }

    @staticmethod
    def _safe_float(x, default=0.0):
        try:
            return default if pd.isna(x) else float(x)
        except Exception:
            return default

    @staticmethod
    def _parse_time(time_str: str) -> float:
        if pd.isna(time_str) or not isinstance(time_str, str):
            return 0.0
        time_str = time_str.strip()
        if time_str == "":
            return 0.0
        try:
            parts = time_str.split(":")
            if len(parts) == 2:
                return float(parts[0]) * 60.0 + float(parts[1])
            return float(parts[0])
        except Exception:
            return 0.0

    @staticmethod
    def _parse_distance(distance_str: str) -> Tuple[float, float]:
        if pd.isna(distance_str) or not isinstance(distance_str, str):
            return -1.0, 0.0
        
        distance_str = distance_str.strip()
        surface_code = -1.0
        if "芝" in distance_str:
            surface_code = 0.0
        elif "ダ" in distance_str or "砂" in distance_str:
            surface_code = 1.0
        elif "障" in distance_str:
            surface_code = 2.0

        nums = re.findall(r"\d+", distance_str)
        return surface_code, float(nums[0]) if nums else 0.0

    @staticmethod
    def _parse_horse_weight(weight_str: str) -> Tuple[float, float]:
        if pd.isna(weight_str) or not isinstance(weight_str, str):
            return 0.0, 0.0
        weight_str = weight_str.strip()
        if weight_str == "":
            return 0.0, 0.0

        match = re.match(r"(\d+)\s*\(([-+\d]+)\)", weight_str)
        if match:
            return float(match.group(1)), float(match.group(2))

        nums = re.findall(r"\d+", weight_str)
        return float(nums[0]) if nums else 0.0, 0.0

    def _parse_direction(self, venue: str) -> float:
        if pd.isna(venue) or not isinstance(venue, str):
            return -1.0
        if "新潟" in venue and "直" in venue:
            return 2.0
        for k, v in self.venue_to_direction.items():
            if k in venue:
                return float(v)
        return -1.0

    @staticmethod
    def _parse_prize_money(money_str: str) -> float:
        if pd.isna(money_str) or not isinstance(money_str, str):
            return 0.0
        money_str = money_str.replace(",", "")
        total = 0.0

        oku_match = re.search(r"(\d+)億", money_str)
        if oku_match:
            total += float(oku_match.group(1)) * 10000
        man_match = re.search(r"(\d+)万", money_str)
        if man_match:
            total += float(man_match.group(1))
        return total

    @staticmethod
    def _parse_pace(pace_str: str):
        if pd.isna(pace_str) or not isinstance(pace_str, str):
            return 0.0, 0.0, 0.0
        pace_str = pace_str.strip()
        if "-" not in pace_str:
            return 0.0, 0.0, 0.0
        try:
            first, second = pace_str.split("-")
            first, second = float(first), float(second)
            return first, second, second - first
        except Exception:
            return 0.0, 0.0, 0.0

    def _load_horse_data(self, horse_dir: str) -> pd.DataFrame:
        parquet_files = glob.glob(os.path.join(horse_dir, "*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in {horse_dir}")
        print(f"Found {len(parquet_files)} horse parquet files")

        records = []
        for file_path in parquet_files:
            try:
                df = pd.read_parquet(file_path)
            except Exception as e:
                print(f"Failed reading {file_path}: {e}")
                continue

            for _, horse in df.iterrows():
                horse_id = horse.get("horse_id")
                if pd.isna(horse_id):
                    continue

                history_raw = horse.get("history", [])
                if isinstance(history_raw, str):
                    try: history = json.loads(history_raw)
                    except Exception: continue
                elif isinstance(history_raw, list):
                    history = history_raw
                else:
                    continue

                birth_date = str(horse.get("birth_date", ""))
                birth_year_match = re.search(r"(\d{4})", birth_date)
                birth_year = int(birth_year_match.group(1)) if birth_year_match else 0
                gender = str(horse.get("gender", ""))

                total_results = str(horse.get("total_results", ""))
                career_races_total, career_wins_total = 0, 0
                result_match = re.search(r"(\d+)戦(\d+)勝", total_results)
                if result_match:
                    career_races_total = int(result_match.group(1))
                    career_wins_total = int(result_match.group(2))

                total_prize = (
                    self._parse_prize_money(str(horse.get("prize_money_jra", "0万円"))) +
                    self._parse_prize_money(str(horse.get("prize_money_nar", "0万円")))
                )

                for race in history:
                    if not isinstance(race, dict):
                        continue
                    records.append({
                        "race_id": race.get("race_id", ""),
                        "horse_id": str(horse_id),
                        "date": race.get("date", ""),
                        "venue": race.get("venue", ""),
                        "weather": race.get("weather", ""),
                        "track_condition": race.get("track_condition", ""),
                        "distance_str": race.get("distance", ""),
                        "time": race.get("time", ""),
                        "last_3f": race.get("last_3f", ""),
                        "odds": race.get("odds", ""),
                        "popularity": race.get("popularity", ""),
                        "horse_weight_str": race.get("horse_weight", ""),
                        "finish_pos": race.get("finish_pos", ""),
                        "weight": race.get("weight", ""),
                        "margin": race.get("margin", ""),
                        "pace": race.get("pace", ""),
                        "prize": race.get("prize", "0"),
                        "bracket_num": race.get("bracket_num", ""),
                        "horse_num": race.get("horse_num", ""),
                        "num_horses": race.get("num_horses", ""),
                        "gender": gender,
                        "birth_year": birth_year,
                        "career_races_total": career_races_total,
                        "career_wins_total": career_wins_total,
                        "total_prize_money": total_prize,
                    })
        return pd.DataFrame(records)

    def clean_race_results(self, df: pd.DataFrame):
        weather_map = {"晴": 0.0, "曇": 1.0, "曇り": 1.0, "雨": 2.0, "小雨": 2.0, "雪": 3.0, "みぞれ": 3.0}
        cond_map = {"良": 0.0, "稍重": 1.0, "稍": 1.0, "重": 2.0, "不良": 3.0}
        gender_map = {"牝": 0.0, "牡": 1.0, "セ": 2.0, "騸": 2.0}

        df["weather_code"] = df["weather"].map(weather_map).fillna(-1.0).astype(np.float32)
        df["condition_code"] = df["track_condition"].map(cond_map).fillna(-1.0).astype(np.float32)
        df["gender_code"] = df["gender"].map(gender_map).fillna(-1.0).astype(np.float32)
        df["time_seconds"] = df["time"].apply(self._parse_time).astype(np.float32)

        dist_res = df["distance_str"].apply(self._parse_distance)
        df["surface_code"] = [x[0] for x in dist_res]
        df["distance"] = [x[1] for x in dist_res]
        df["direction_code"] = df["venue"].apply(self._parse_direction).astype(np.float32)

        df["date_dt"] = pd.to_datetime(df["date"], format="%Y/%m/%d", errors="coerce")
        df["date_clean"] = df["date_dt"].dt.strftime("%Y-%m-%d").fillna("1970-01-01")
        df["age"] = (df["date_dt"].dt.year - df["birth_year"]).fillna(0.0).astype(np.float32)

        hw_res = df["horse_weight_str"].apply(self._parse_horse_weight)
        df["horse_weight"] = [x[0] for x in hw_res]
        df["horse_weight_diff"] = [x[1] for x in hw_res]

        numeric_cols = ["weight", "last_3f", "odds", "popularity", "bracket_num", "horse_num", "num_horses", "margin", "prize", "finish_pos"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype(np.float32)

        df["weight_carried"] = df["weight"]
        df["finish_rank"] = df["finish_pos"]

        pace_res = df["pace"].apply(self._parse_pace)
        df["pace_first_half"] = [x[0] for x in pace_res]
        df["pace_second_half"] = [x[1] for x in pace_res]
        df["pace_diff"] = [x[2] for x in pace_res]

        df = df.sort_values(["horse_id", "date_dt"]).reset_index(drop=True)
        df["career_races"] = df.groupby("horse_id").cumcount().astype(np.float32)
        df["career_wins"] = df.groupby("horse_id")["finish_rank"].transform(
            lambda x: (x == 1.0).astype(np.float32).cumsum().shift(1, fill_value=0)
        ).astype(np.float32)

        return df

    @staticmethod
    def _parse_payoff_values(v):
        if pd.isna(v): return [0.0]
        if isinstance(v, (int, float)): return [float(v)]
        if not isinstance(v, str): return [0.0]
        
        v = v.strip()
        if v == "": return [0.0]
        out = []
        for item in v.split("|"):
            try: out.append(float(item))
            except Exception: continue
        return out if out else [0.0]

    def _build_payoff_tensor(self, payoff_row):
        payoff_fields = ["Win", "Place", "Bracket_Quinella", "Quinella", "Quinella_Place", "Exacta", "Trio", "Trifecta"]
        values = []
        for field in payoff_fields:
            parsed = self._parse_payoff_values(payoff_row.get(field, "0"))[:3]
            while len(parsed) < 3:
                parsed.append(0.0)
            values.extend(parsed)
        return np.array(values, dtype=np.float32)

    def _load_payoff_data(self, payoff_dir):
        parquet_files = glob.glob(os.path.join(payoff_dir, "*.parquet"))
        if not parquet_files:
            print("No payoff parquet found")
            return pd.DataFrame()

        dfs = []
        for f in parquet_files:
            try: dfs.append(pd.read_parquet(f))
            except Exception as e: print(f"Failed reading payoff file {f}: {e}")
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    def run(self, horse_dir: str, payoff_dir: str):
        print("=" * 60 + "\nBuilding Horse Racing Datastore\n" + "=" * 60)
        df = self._load_horse_data(horse_dir)
        if df.empty:
            raise ValueError("No race data loaded")
        print(f"Loaded {len(df)} race records")
        df = self.clean_race_results(df)

        print("\n[1] Building race-level tensors")
        race_index_records = []
        for race_id, group in df.groupby("race_id"):
            if pd.isna(race_id) or str(race_id).strip() == "":
                continue
            group = group.sort_values("horse_num")
            race_tensor = group[self.atomic_features].values.astype(np.float32)
            self.storage.save_race_tensor(str(race_id), race_tensor)
            race_index_records.append({
                "race_id": str(race_id),
                "date": group["date_clean"].iloc[0],
                "field_size": len(group),
            })

        print("\n[2] Building horse-level tensors")
        horse_index_records = []
        for horse_id, group in df.groupby("horse_id"):
            group = group.sort_values("date_dt")
            horse_tensor = group[self.atomic_features].values.astype(np.float32)
            self.storage.save_horse_data(
                horse_id=str(horse_id),
                data=horse_tensor,
                dates=group["date_clean"].tolist(),
            )
            horse_index_records.append({"horse_id": str(horse_id), "total_races": len(group)})

        print("\n[3] Processing payoff tensors")
        df_payoff = self._load_payoff_data(payoff_dir)
        payoff_records = []
        if not df_payoff.empty:
            race_id_set = set(r["race_id"] for r in race_index_records)
            for _, row in df_payoff.iterrows():
                race_id = str(row.get("race_id", ""))
                if race_id == "": continue
                payoff_tensor = self._build_payoff_tensor(row)
                self.storage.save_payoff_tensor(race_id, payoff_tensor)
                payoff_records.append({
                    "race_id": race_id,
                    "date": row.get("date", ""),
                    "has_race_data": (race_id in race_id_set),
                })

        print("\n[4] Saving metadata")
        metadata_dir = self.storage.metadata_dir
        pd.DataFrame(race_index_records).to_parquet(os.path.join(metadata_dir, "race_index.parquet"))
        pd.DataFrame(horse_index_records).to_parquet(os.path.join(metadata_dir, "horse_index.parquet"))
        
        race_horse_map = df.groupby("race_id")["horse_id"].apply(list).reset_index()
        race_horse_map.to_parquet(os.path.join(metadata_dir, "race_horse_map.parquet"))

        horse_static = df.groupby("horse_id").agg({
            "gender": "first", "birth_year": "first", "career_races_total": "max",
            "career_wins_total": "max", "total_prize_money": "max",
        }).reset_index()
        horse_static.to_parquet(os.path.join(metadata_dir, "horse_static.parquet"))

        if payoff_records:
            pd.DataFrame(payoff_records).to_parquet(os.path.join(metadata_dir, "payoff_index.parquet"))
        df.to_parquet(os.path.join(metadata_dir, "flattened_master_records.parquet"))

        print("\n" + "=" * 60 + f"\nDatastore Build Complete\nRace tensors : {len(race_index_records)}\nHorse tensors: {len(horse_index_records)}\nFeatures     : {len(self.atomic_features)}\n" + "=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--store-root", default="../../data/datastore1")
    parser.add_argument("--horse-dir", default="../../data/horse")
    parser.add_argument("--payoff-dir", default="../../data/payoff")
    args = parser.parse_args()

    pipeline = QuantDataIngestionPipeline(store_root=args.store_root)
    pipeline.run(horse_dir=args.horse_dir, payoff_dir=args.payoff_dir)