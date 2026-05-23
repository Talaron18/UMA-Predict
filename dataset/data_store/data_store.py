import os
import glob
import json
import re
import hashlib
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional

class FeatureTopology:
    FEATURE_METADATA = {
        "surface_code":       {"ns": "base",     "group": "track",    "deps": []},
        "direction_code":     {"ns": "base",     "group": "track",    "deps": []},
        "distance":           {"ns": "base",     "group": "track",    "deps": []},
        "weather_code":       {"ns": "base",     "group": "env",      "deps": []},
        "condition_code":     {"ns": "base",     "group": "env",      "deps": []},
        "gender_code":        {"ns": "base",     "group": "horse",    "deps": []},
        "age":                {"ns": "base",     "group": "horse",    "deps": []},
        "weight_carried":     {"ns": "base",     "group": "horse",    "deps": []},
        "time_seconds":       {"ns": "base",     "group": "performance", "deps": []},
        "last_3f":            {"ns": "base",     "group": "performance", "deps": []},
        "odds":               {"ns": "base",     "group": "market",   "deps": []},
        "popularity":         {"ns": "base",     "group": "market",   "deps": []},
        "horse_weight":       {"ns": "base",     "group": "horse",    "deps": []},
        "horse_weight_diff":  {"ns": "base",     "group": "horse",    "deps": []},
        "finish_rank":        {"ns": "base",     "group": "performance", "deps": []},
        "bracket_num":        {"ns": "base",     "group": "race_state", "deps": []},
        "horse_num":          {"ns": "base",     "group": "race_state", "deps": []},
        "num_horses":         {"ns": "base",     "group": "race_state", "deps": []},
        "margin":             {"ns": "base",     "group": "performance", "deps": []},
        "pace_first_half":    {"ns": "base",     "group": "performance", "deps": []},
        "pace_second_half":   {"ns": "base",     "group": "performance", "deps": []},
        "pace_diff":          {"ns": "base",     "group": "performance", "deps": []},
        "prize":              {"ns": "base",     "group": "market",   "deps": []},
        
        "sire_code":            {"ns": "eco", "group": "pedigree",  "deps": []},
        "dam_sire_code":        {"ns": "eco", "group": "pedigree",  "deps": []},
        "jockey_code":          {"ns": "eco", "group": "human",     "deps": []},
        "trainer_code":         {"ns": "eco", "group": "human",     "deps": []},
        "course_code":          {"ns": "eco", "group": "track",     "deps": []},
        "track_layout":         {"ns": "eco", "group": "track",     "deps": []},
        "corner_count":         {"ns": "eco", "group": "track",     "deps": []},
        "slope_type":           {"ns": "eco", "group": "track",     "deps": []},
        
        "days_since_last_race": {"ns": "eco", "group": "temporal",  "deps": []},
        "month":                {"ns": "eco", "group": "temporal",  "deps": []},
        "season":               {"ns": "eco", "group": "temporal",  "deps": ["month"]},
        
        "implied_prob":          {"ns": "derived", "group": "market",   "deps": ["odds"]},
        "favorite_gap":          {"ns": "derived", "group": "market",   "deps": ["implied_prob"]},
        "race_strategy_tendency":{"ns": "derived", "group": "macro",   "deps": ["pace_diff", "last_3f"]},
        "field_wear":            {"ns": "derived", "group": "macro",   "deps": ["course_code", "date_clean", "num_horses"]},
        "density":               {"ns": "derived", "group": "macro",   "deps": ["num_horses", "distance"]},
        "jockey_weight_min":     {"ns": "derived", "group": "human_static", "deps": ["jockey_code", "weight_carried"]}
    }

    @classmethod
    def get_features_by_ns(cls, ns: str) -> List[str]:
        return [k for k, v in cls.FEATURE_METADATA.items() if v["ns"] == ns]

    @classmethod
    def get_dependency_graph(cls) -> Dict[str, List[str]]:
        return {k: v["deps"] for k, v in cls.FEATURE_METADATA.items() if v["deps"]}


class FeatureRegistry:
    def __init__(self, path: str):
        self.path = path
        self.feature_to_idx: Dict[str, int] = {}

    def register_features(self, features: List[str]) -> None:
        self.feature_to_idx = {feat: idx for idx, feat in enumerate(features)}
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.feature_to_idx, f, indent=2, ensure_ascii=False)
        print(f"[FeatureRegistry] Registered {len(features)} features via Topology.")

    def load(self) -> None:
        if os.path.exists(self.path):
            with open(self.path, "r", encoding="utf-8") as f:
                self.feature_to_idx = json.load(f)


class StorageEngine:
    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self.metadata_dir = os.path.join(root_dir, "metadata")
        self.horse_dir = os.path.join(root_dir, "horse")
        self.race_dir = os.path.join(root_dir, "race")
        self.payoff_dir = os.path.join(root_dir, "payoff")
        for d in [self.metadata_dir, self.horse_dir, self.race_dir, self.payoff_dir]:
            os.makedirs(d, exist_ok=True)

    def save_horse_tensor(self, horse_id: str, data: np.ndarray, dates: List[str]) -> None:
        path = os.path.join(self.horse_dir, str(horse_id))
        os.makedirs(path, exist_ok=True)
        data = data.astype(np.float32)

        bin_path = os.path.join(path, "features.bin")
        fp = np.memmap(bin_path, dtype="float32", mode="w+", shape=data.shape)
        fp[:] = data[:]
        fp.flush()

        with open(os.path.join(path, "meta.json"), "w", encoding="utf-8") as f:
            json.dump({
                "shape": list(data.shape),
                "dtype": "float32",
                "num_features": data.shape[1] if len(data.shape) > 1 else 1
            }, f, indent=2)

        np.save(os.path.join(path, "dates.npy"), np.array(dates, dtype="datetime64[D]"))

    def save_race_tensor(self, race_id: str, data: np.ndarray) -> None:
        np.save(os.path.join(self.race_dir, f"{race_id}.npy"), data.astype(np.float32))

    def save_payoff_tensor(self, race_id: str, data: np.ndarray) -> None:
        np.save(os.path.join(self.payoff_dir, f"{race_id}.npy"), data.astype(np.float32))


class HashEncoder:
    def __init__(self, num_buckets: int = 5000):
        self.num_buckets = num_buckets

    def encode(self, value: Optional[str]) -> float:
        if pd.isna(value) or not value or str(value).strip().lower() in ["nan", "null", ""]:
            return 0.0  
        hasher = hashlib.md5(str(value).strip().encode("utf-8"))
        hash_val = int(hasher.hexdigest(), 16)
        
        return float((hash_val % self.num_buckets) + 1)

class QuantDataIngestionPipeline:

    VENUE_ATTRIBUTES = {
        "札幌":  {"course_id": 1, "track_layout": 0, "corner_count": 2, "slope_type": 0},
        "函館":  {"course_id": 2, "track_layout": 0, "corner_count": 2, "slope_type": 0},
        "福島":  {"course_id": 3, "track_layout": 0, "corner_count": 2, "slope_type": 0},
        "中山":  {"course_id": 4, "track_layout": 0, "corner_count": 2, "slope_type": 0},
        "東京":  {"course_id": 5, "track_layout": 0, "corner_count": 2, "slope_type": 0},
        "中京":  {"course_id": 6, "track_layout": 0, "corner_count": 2, "slope_type": 0},
        "京都":  {"course_id": 7, "track_layout": 0, "corner_count": 2, "slope_type": 0},
        "阪神":  {"course_id": 8, "track_layout": 0, "corner_count": 2, "slope_type": 0},
        "小倉":  {"course_id": 9, "track_layout": 0, "corner_count": 2, "slope_type": 0},
        "新潟":  {"course_id": 10, "track_layout": 0, "corner_count": 2, "slope_type": 0},
        "新潟直":{"course_id": 11, "track_layout": 1, "corner_count": 0, "slope_type": 0},
    }

    def __init__(self, store_root: str):
        self.storage = StorageEngine(store_root)
        self.registry = FeatureRegistry(os.path.join(self.storage.metadata_dir, "feature_map.json"))

        self.atomic_features = list(FeatureTopology.FEATURE_METADATA.keys())
        self.registry.register_features(self.atomic_features)

        self.metadata_columns = [
            "horse_name", "status", "sire_name", "grandsire_name", "great_grandsire_name", "dam_sire_name"
        ]

        self.venue_to_direction = {
            "札幌": 0, "函館": 0, "福島": 0, "中山": 0, "阪神": 0, "小倉": 0,
            "新潟": 1, "東京": 1, "中京": 1, "京都": 1, "新潟直": 2
        }

        self.hasher = HashEncoder(num_buckets=8000)

    @staticmethod
    def _safe_float(x, default=0.0) -> float:
        try: return default if pd.isna(x) else float(x)
        except: return default

    @staticmethod
    def _parse_time(time_str: str) -> float:
        if pd.isna(time_str) or not isinstance(time_str, str): return 0.0
        time_str = time_str.strip()
        try:
            parts = time_str.split(":")
            if len(parts) == 2:
                return float(parts[0]) * 60.0 + float(parts[1])
            return float(parts[0])
        except: return 0.0

    @staticmethod
    def _parse_distance(distance_str: str) -> Tuple[float, float]:
        if pd.isna(distance_str) or not isinstance(distance_str, str): return -1.0, 0.0
        distance_str = distance_str.strip()
        surface_code = -1.0
        if "芝" in distance_str: surface_code = 0.0
        elif "ダ" in distance_str or "砂" in distance_str: surface_code = 1.0
        elif "障" in distance_str: surface_code = 2.0
        nums = re.findall(r"\d+", distance_str)
        return surface_code, float(nums[0]) if nums else 0.0

    @staticmethod
    def _parse_horse_weight(weight_str: str) -> Tuple[float, float]:
        if pd.isna(weight_str) or not isinstance(weight_str, str): return 0.0, 0.0
        weight_str = weight_str.strip()
        match = re.match(r"(\d+)\s*\(([-+\d]+)\)", weight_str)
        if match: return float(match.group(1)), float(match.group(2))
        nums = re.findall(r"\d+", weight_str)
        return float(nums[0]) if nums else 0.0, 0.0

    def _parse_direction(self, venue: str) -> float:
        if pd.isna(venue) or not isinstance(venue, str): return -1.0
        if "新潟" in venue and "直" in venue: return 2.0
        for k, v in self.venue_to_direction.items():
            if k in venue: return float(v)
        return -1.0

    @staticmethod
    def _parse_prize_money(money_str: str) -> float:
        if pd.isna(money_str) or not isinstance(money_str, str): return 0.0
        money_str = money_str.replace(",", "")
        total = 0.0
        oku_match = re.search(r"(\d+)億", money_str)
        if oku_match: total += float(oku_match.group(1)) * 10000
        man_match = re.search(r"(\d+)万", money_str)
        if man_match: total += float(man_match.group(1))
        return total

    @staticmethod
    def _parse_pace(pace_str: str) -> Tuple[float, float, float]:
        if pd.isna(pace_str) or not isinstance(pace_str, str) or "-" not in pace_str:
            return 0.0, 0.0, 0.0
        try:
            first, second = pace_str.strip().split("-")
            return float(first), float(second), float(second) - float(first)
        except: return 0.0, 0.0, 0.0

    def _compute_macro_eco_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """集中计算战术大盘、赛道磨损及密度等高级生态特征"""
        
        df["density"] = (df["num_horses"] / (df["distance"] / 1000.0 + 1e-5)).fillna(0.0).astype(np.float32)

        race_pace_mean = df.groupby("race_id")["pace_diff"].transform("mean")
        df["race_strategy_tendency"] = np.where(race_pace_mean < -1.0, 0.0, np.where(race_pace_mean > 1.0, 2.0, 1.0)).astype(np.float32)
        df = df.sort_values(["course_code", "date_dt"]).reset_index(drop=True)
        race_horses = df.groupby(["course_code", "date_clean"])["num_horses"].first().reset_index()
        race_horses["wear_cum"] = race_horses.groupby("course_code")["num_horses"].cumsum()
        
        wear_map = dict(zip(race_horses["course_code"].astype(str) + "_" + race_horses["date_clean"], race_horses["wear_cum"]))
        df["field_wear"] = (df["course_code"].astype(str) + "_" + df["date_clean"]).map(wear_map).fillna(0.0).astype(np.float32)

        df["jockey_weight_min"] = df.groupby("jockey_code")["weight_carried"].transform("min").fillna(0.0).astype(np.float32)

        return df

    def clean_race_results(self, df: pd.DataFrame, pedigree_df: pd.DataFrame) -> pd.DataFrame:
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

        for col in ["weight", "last_3f", "odds", "popularity", "bracket_num", "horse_num",
                    "num_horses", "margin", "prize", "finish_pos"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype(np.float32)

        df["weight_carried"] = df["weight"]
        df["finish_rank"] = df["finish_pos"]

        pace_res = df["pace"].apply(self._parse_pace)
        df["pace_first_half"] = [x[0] for x in pace_res]
        df["pace_second_half"] = [x[1] for x in pace_res]
        df["pace_diff"] = [x[2] for x in pace_res]

        df["sire_code"] = df["horse_id"].astype(str).map(
            dict(zip(pedigree_df["horse_id"].astype(str), pedigree_df["sire_id"])) if not pedigree_df.empty else {}
        ).fillna("").apply(self.hasher.encode).astype(np.float32)

        df["dam_sire_code"] = df["horse_id"].astype(str).map(
            dict(zip(pedigree_df["horse_id"].astype(str), pedigree_df["dam_sire_id"])) if not pedigree_df.empty else {}
        ).fillna("").apply(self.hasher.encode).astype(np.float32)

        df["jockey_code"] = df["jockey"].astype(str).apply(self.hasher.encode).astype(np.float32)
        df["trainer_code"] = df["trainer"].astype(str).apply(self.hasher.encode).astype(np.float32)

        def _get_venue_base(v):
            if pd.isna(v): return ""
            if "新潟直" in v: return "新潟直"
            m = re.search(r"[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]+", v)
            return m.group(0) if m else ""

        venue_bases = df["venue"].apply(_get_venue_base)
        for attr in ["course_id", "track_layout", "corner_count", "slope_type"]:
            df[attr] = venue_bases.apply(lambda v: self.VENUE_ATTRIBUTES.get(v, {}).get(attr, -1)).astype(np.float32)

        course_map = {v["course_id"]: i for i, v in enumerate(self.VENUE_ATTRIBUTES.values())}
        df["course_code"] = df["course_id"].map(course_map).fillna(-1).astype(np.float32)

        df = df.sort_values(["horse_id", "date_dt"]).reset_index(drop=True)
        df["days_since_last_race"] = df.groupby("horse_id")["date_dt"].diff().dt.days.fillna(0).astype(np.float32)
        df["month"] = df["date_dt"].dt.month.fillna(0).astype(np.float32)
        season_map = {3:1, 4:1, 5:1, 6:2, 7:2, 8:2, 9:3, 10:3, 11:3, 12:4, 1:4, 2:4}
        df["season"] = df["month"].map(season_map).fillna(0).astype(np.float32)

        df["odds"] = df["odds"].replace(0, np.nan)
        df["implied_prob"] = (100.0 / df["odds"]).fillna(0.0).astype(np.float32)

        def compute_fav_gap(group):
            fav_odds = group["odds"].min()
            fav_prob = 100.0 / fav_odds if (pd.notna(fav_odds) and fav_odds > 0) else 0.0
            group["favorite_gap"] = group["implied_prob"] - fav_prob
            return group
        df = df.groupby("race_id", group_keys=False).apply(compute_fav_gap)
        df["favorite_gap"] = df["favorite_gap"].astype(np.float32)

        df["career_races"] = df.groupby("horse_id").cumcount().astype(np.float32)
        df["career_wins"] = df.groupby("horse_id")["finish_rank"].transform(
            lambda x: (x == 1.0).astype(np.float32).cumsum().shift(1, fill_value=0)
        ).astype(np.float32)

        df = self._compute_macro_eco_features(df)

        return df

    def run(self, horse_dir: str, payoff_dir: str, pedigree_dir: str):
        print("=" * 60 + "\nBuilding Advanced Tensor Datastore\n" + "=" * 60)

        df_horses = self._load_horse_data(horse_dir)
        if df_horses.empty: raise ValueError("No race data loaded")
        df_pedigree = self._load_pedigree(pedigree_dir)
        df_payoff = self._load_payoff_data(payoff_dir)

        df = self.clean_race_results(df_horses, df_pedigree)

        print("\n[1] Building race-level tensors")
        race_index_records = []
        for race_id, group in df.groupby("race_id"):
            if pd.isna(race_id) or str(race_id).strip() == "": continue
            group = group.sort_values("horse_num")
            race_tensor = group[self.atomic_features].values.astype(np.float32)
            self.storage.save_race_tensor(str(race_id), race_tensor)
            race_index_records.append({
                "race_id": str(race_id), "date": group["date_clean"].iloc[0], "field_size": len(group)
            })

        print("\n[2] Building horse-level tensors (Using continuous memmap)")
        horse_index_records = []
        for horse_id, group in df.groupby("horse_id"):
            group = group.sort_values("date_dt")
            horse_tensor = group[self.atomic_features].values.astype(np.float32)
            
            self.storage.save_horse_tensor(
                horse_id=str(horse_id), data=horse_tensor, dates=group["date_clean"].tolist()
            )
            horse_index_records.append({"horse_id": str(horse_id), "total_races": len(group)})

        print("\n[3] Processing payoff tensors")
        payoff_records = []
        if not df_payoff.empty:
            race_id_set = set(r["race_id"] for r in race_index_records)
            for _, row in df_payoff.iterrows():
                race_id = str(row.get("race_id", ""))
                if race_id == "": continue
                payoff_tensor = self._build_payoff_tensor(row)
                self.storage.save_payoff_tensor(race_id, payoff_tensor)
                payoff_records.append({
                    "race_id": race_id, "date": row.get("date", ""), "has_race_data": (race_id in race_id_set)
                })

        print("\n[4] Saving integrated metadata")
        meta_dir = self.storage.metadata_dir
        pd.DataFrame(race_index_records).to_parquet(os.path.join(meta_dir, "race_index.parquet"))
        pd.DataFrame(horse_index_records).to_parquet(os.path.join(meta_dir, "horse_index.parquet"))
        
        race_horse_map = df.groupby("race_id")["horse_id"].apply(list).reset_index()
        race_horse_map.to_parquet(os.path.join(meta_dir, "race_horse_map.parquet"))

        df.to_parquet(os.path.join(meta_dir, "flattened_master_records.parquet"))
        print("\n" + "=" * 60 + f"\nDatastore Build Complete (memmap mode)\nFeatures Total: {len(self.atomic_features)}\n" + "=" * 60)

    def _load_horse_data(self, horse_dir: str) -> pd.DataFrame:
        parquet_files = glob.glob(os.path.join(horse_dir, "*.parquet"))
        if not parquet_files: raise FileNotFoundError(f"No parquet files found in {horse_dir}")
        records = []
        for file_path in parquet_files:
            try: df = pd.read_parquet(file_path)
            except: continue
            for _, horse in df.iterrows():
                horse_id = horse.get("horse_id")
                if pd.isna(horse_id): continue
                history_raw = horse.get("history", [])
                if isinstance(history_raw, str):
                    try: history = json.loads(history_raw)
                    except: continue
                elif isinstance(history_raw, list): history = history_raw
                else: continue
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
                trainer_raw = str(horse.get("trainer", ""))
                trainer_name = trainer_raw.split(" (")[0] if "(" in trainer_raw else trainer_raw

                for race in history:
                    if not isinstance(race, dict): continue
                    records.append({
                        "race_id": race.get("race_id", ""), "horse_id": str(horse_id), "date": race.get("date", ""),
                        "venue": race.get("venue", ""), "weather": race.get("weather", ""), "track_condition": race.get("track_condition", ""),
                        "distance_str": race.get("distance", ""), "time": race.get("time", ""), "last_3f": race.get("last_3f", ""),
                        "odds": race.get("odds", ""), "popularity": race.get("popularity", ""), "horse_weight_str": race.get("horse_weight", ""),
                        "finish_pos": race.get("finish_pos", ""), "weight": race.get("weight", ""), "margin": race.get("margin", ""),
                        "pace": race.get("pace", ""), "prize": race.get("prize", "0"), "bracket_num": race.get("bracket_num", ""),
                        "horse_num": race.get("horse_num", ""), "num_horses": race.get("num_horses", ""), "jockey": str(race.get("jockey", "")),
                        "trainer": trainer_name, "gender": gender, "birth_year": birth_year, "career_races_total": career_races_total,
                        "career_wins_total": career_wins_total, "total_prize_money": total_prize,
                    })
        return pd.DataFrame(records)

    def _load_pedigree(self, pedigree_dir: str) -> pd.DataFrame:
        parquet_files = glob.glob(os.path.join(pedigree_dir, "*.parquet"))
        if not parquet_files: return pd.DataFrame(columns=["horse_id", "sire_id", "dam_sire_id"])
        records = []
        for f in parquet_files:
            df = pd.read_parquet(f)
            for _, row in df.iterrows():
                try: ped = json.loads(row["pedigree_json"])
                except: ped = []
                sire_id = ped[0].get("id", "") if len(ped) > 0 else ""
                half = len(ped) // 2
                dam_sire_id = ped[half].get("id", "") if 0 < half < len(ped) else ""
                records.append({"horse_id": str(row["horse_id"]), "sire_id": sire_id, "dam_sire_id": dam_sire_id})
        return pd.DataFrame(records)

    def _load_payoff_data(self, payoff_dir: str) -> pd.DataFrame:
        parquet_files = glob.glob(os.path.join(payoff_dir, "*.parquet"))
        dfs = [pd.read_parquet(f) for f in parquet_files if os.path.exists(f)]
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    def _build_payoff_tensor(self, payoff_row) -> np.ndarray:
        fields = ["Win", "Place", "Bracket_Quinella", "Quinella", "Quinella_Place", "Exacta", "Trio", "Trifecta"]
        values = []
        for f in fields:
            parsed = self._parse_payoff_values(payoff_row.get(f, "0"))[:3]
            while len(parsed) < 3: parsed.append(0.0)
            values.extend(parsed)
        return np.array(values, dtype=np.float32)

    @staticmethod
    def _parse_payoff_values(v) -> List[float]:
        if pd.isna(v): return [0.0]
        if isinstance(v, (int, float)): return [float(v)]
        if not isinstance(v, str): return [0.0]
        out = []
        for item in v.strip().split("|"):
            try: out.append(float(item))
            except: continue
        return out if out else [0.0]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--store-root", default="./data/datastore")
    parser.add_argument("--horse-dir", default="./data/horse")
    parser.add_argument("--payoff-dir", default="./data/payoff")
    parser.add_argument("--pedigree-dir", default="./data/pedigree_outputs")
    args = parser.parse_args()

    pipeline = QuantDataIngestionPipeline(store_root=args.store_root)
    pipeline.run(horse_dir=args.horse_dir, payoff_dir=args.payoff_dir, pedigree_dir=args.pedigree_dir)