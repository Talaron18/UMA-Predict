import os
import glob
import json
import re
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple

# ==========================================
# 1. 基础设施层 (遵循架构设计书规范)
# ==========================================

class FeatureRegistry:
    """特征注册表：管理原子特征名称与 Tensor 列索引的映射"""
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
    """高性能存储引擎：负责 Horse-Level bin/npy 和 Race-Level npy 的底层写入"""
    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self.metadata_dir = os.path.join(root_dir, "metadata")
        self.horse_dir = os.path.join(root_dir, "horse")
        self.race_dir = os.path.join(root_dir, "race")
        
        os.makedirs(self.metadata_dir, exist_ok=True)
        os.makedirs(self.horse_dir, exist_ok=True)
        os.makedirs(self.race_dir, exist_ok=True)

    def save_horse_data(self, horse_id: str, data: np.ndarray, dates: List[str]):
        """保存单匹马的时间序列数据 (num_races, num_features)"""
        path = os.path.join(self.horse_dir, str(horse_id))
        os.makedirs(path, exist_ok=True)
        
        # 写入 features.bin (mmap 格式)
        data = data.astype(np.float32)
        fp = np.memmap(os.path.join(path, "features.bin"), dtype='float32', mode='w+', shape=data.shape)
        fp[:] = data[:]
        fp.flush()
        
        # 写入 meta.json 记录 shape
        with open(os.path.join(path, "meta.json"), "w") as f:
            json.dump({"shape": list(data.shape)}, f)
            
        # 写入 dates.npy
        np.save(os.path.join(path, "dates.npy"), np.array(dates, dtype='datetime64[D]'))

    def save_race_tensor(self, race_id: str, data: np.ndarray):
        """保存整场比赛的横截面数据 (field_size, num_features)"""
        path = os.path.join(self.race_dir, f"{str(race_id)}.npy")
        np.save(path, data.astype(np.float32))


# ==========================================
# 2. 核心 ETL 转换与导入管道
# ==========================================

class QuantDataIngestionPipeline:
    def __init__(self, store_root: str):
        self.storage = StorageEngine(store_root)
        self.registry = FeatureRegistry(os.path.join(self.storage.metadata_dir, "feature_map.json"))
        
        # 统一定义全系统强制对齐的原子特征列表 (15维原子特征)
        self.atomic_features = [
            "surface_code",       # 芝=0, 砂/ダート=1
            "direction_code",     # 右=0, 左=1, 直=2
            "distance",           # 距离 (米)
            "weather_code",       # 晴=0, 曇=1, 雨=2, 雪=3
            "condition_code",     # 良=0, 稍重=1, 重=2, 不良=3
            "gender_code",        # 牝=0, 牡=1, 騸/セ=2
            "age",                # 年龄
            "weight_carried",     # 负重 (kg)
            "time_seconds",       # 完赛时间 (秒)
            "last_3f",            # 后三华里时间 (秒)
            "odds",               # 独赢赔率
            "popularity",         # 人气排名
            "horse_weight",       # 马体重
            "horse_weight_diff",  # 马体重增减
            "finish_rank"         # 冲线名次
        ]
        self.registry.register_features(self.atomic_features)

    @staticmethod
    def _parse_time(time_str: str) -> float:
        """将 '1:35.1' 解析为 95.1 秒"""
        if pd.isna(time_str) or not isinstance(time_str, str) or ":" not in time_str:
            return 0.0
        try:
            parts = time_str.split(":")
            return float(parts[0]) * 60.0 + float(parts[1])
        except Exception:
            return 0.0

    @staticmethod
    def _parse_gender_age(ga_str: str) -> Tuple[float, float]:
        """将 '牝3' 或 '牡4' 拆分为性别代码和年龄"""
        if pd.isna(ga_str) or not isinstance(ga_str, str) or len(ga_str) < 2:
            return -1.0, 0.0
        
        gender_char = ga_str[0]
        age_str = ga_str[1:]
        
        # 性别映射
        gender_map = {"牝": 0.0, "牡": 1.0, "セ": 2.0, "騸": 2.0}
        g_code = gender_map.get(gender_char, -1.0)
        
        # 年龄提取
        try:
            age = float(re.findall(r'\d+', age_str)[0])
        except Exception:
            age = 0.0
        return g_code, age

    def clean_race_results(self, df: pd.DataFrame) -> pd.DataFrame:
        """纯向量化清洗：将原始复杂字典/DataFrame 转化为标准的 float32 矩阵"""
        print("   🧼 正在解析分类文本与时间项...")
        
        # 1. 基础类别映射
        surface_map = {"芝": 0.0, "ダート": 1.0, "ダ": 1.0}
        direction_map = {"右": 0.0, "左": 1.0, "直": 2.0}
        weather_map = {"晴": 0.0, "曇": 1.0, "雨": 2.0, "雪": 3.0, "曇り": 1.0}
        cond_map = {"良": 0.0, "稍重": 1.0, "重": 2.0, "不良": 3.0}
        
        df["surface_code"] = df["surface"].map(surface_map).fillna(-1.0).astype(np.float32)
        df["direction_code"] = df["direction"].map(direction_map).fillna(-1.0).astype(np.float32)
        df["weather_code"] = df["weather"].map(weather_map).fillna(-1.0).astype(np.float32)
        df["condition_code"] = df["condition"].map(cond_map).fillna(-1.0).astype(np.float32)
        
        # 2. 解析时间与性别年龄
        df["time_seconds"] = df["time"].apply(self._parse_time).astype(np.float32)
        
        ga_res = df["gender_age"].apply(self._parse_gender_age)
        df["gender_code"] = [x[0] for x in ga_res]
        df["age"] = [x[1] for x in ga_res]
        
        # 3. 提取名次（如果数据已按冲线顺序排列，直接按组内序号+1生成真实名次）
        df["finish_rank"] = df.groupby("race_id").cumcount() + 1
        
        # 4. 强制其余数值列转为 float32
        num_cols = ["distance", "weight_carried", "last_3f", "odds", "popularity", "horse_weight", "horse_weight_diff", "finish_rank"]
        for col in num_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0).astype(np.float32)
            
        return df

    def run(self, races_dir: str, payoff_dir: str):
        """核心流：批量扫描、合并、清洗、按 Horse/Race 两层分流并构建索引"""
        # --- Step 1: 扫描并加载所有 races Parquet 文件 ---
        race_files = glob.glob(os.path.join(races_dir, "*.parquet"))
        if not race_files:
            raise FileNotFoundError(f"在 {races_dir} 下未找到任何 parquet 文件！")
        
        print(f"📚 找到 {len(race_files)} 个比赛结果 Parquet 文件，正在加载合并...")
        df_races = pd.concat([pd.read_parquet(f) for f in race_files], ignore_index=True)
        
        # 执行统一清洗
        df_clean = self.clean_race_results(df_races)
        
        # --- Step 2: 存储 Race-Level Tensors (横截面) ---
        print("📦 正在生成 Race-Level Store (横截面 Tensor)...")
        race_index_records = []
        grouped_race = df_clean.groupby("race_id")
        
        for race_id, group in grouped_race:
            # 严格按照 feature_map 的索引顺序抽取矩阵
            race_tensor = group[self.atomic_features].values.astype(np.float32)
            self.storage.save_race_tensor(str(race_id), race_tensor)
            
            # 记录 race 索引元数据
            race_index_records.append({
                "race_id": str(race_id),
                "date": group["date"].iloc[0],
                "field_size": len(group)
            })

        # --- Step 3: 存储 Horse-Level Store (历史时序) ---
        print("🐎 正在生成 Horse-Level Store (个体时序 Tensor)...")
        # 必须先按日期升序排序，确保时间序列从旧到新
        df_clean = df_clean.sort_values(by="date")
        grouped_horse = df_clean.groupby("horse_id")
        horse_index_records = []
        
        for horse_id, group in grouped_horse:
            horse_tensor = group[self.atomic_features].values.astype(np.float32)
            dates = group["date"].astype(str).tolist()
            self.storage.save_horse_data(str(horse_id), horse_tensor, dates)
            
            # 记录 horse 索引元数据
            horse_index_records.append({
                "horse_id": str(horse_id),
                "total_races": len(group)
            })

        # --- Step 4: 产生并保存全局快速索引文件 ---
        print("🗂️ 正在生成全局高速索引索引 (horse_index / race_index)...")
        pd.DataFrame(race_index_records).to_parquet(os.path.join(self.storage.metadata_dir, "race_index.parquet"))
        pd.DataFrame(horse_index_records).to_parquet(os.path.join(self.storage.metadata_dir, "horse_index.parquet"))

        # --- Step 5: 搬运并统一保存 payoff (派彩数据) ---
        payoff_files = glob.glob(os.path.join(payoff_dir, "*.parquet"))
        if payoff_files:
            print(f"💰 找到 {len(payoff_files)} 个派彩 Parquet 文件，正在合并建账...")
            df_payoff = pd.concat([pd.read_parquet(f) for f in payoff_files], ignore_index=True)
            # 留作回测模块（Backtest Ledger）专属账本
            df_payoff.to_parquet(os.path.join(self.storage.metadata_dir, "payouts_ledger.parquet"))
        else:
            print("⚠️ 提示: payoff 文件夹下未找到 Parquet 文件，跳过派彩数据建账。")

        print("\n" + "="*40 + "\n🎉 Data Ingestion 流程全部圆满完成！\n" + "="*40)


# ==========================================
# 3. 自动化运行入口
# ==========================================
if __name__ == "__main__":
    # 按照你的设计书要求，规定 Feature Store 数据落地的根目录位置
    STORE_ROOT_PATH = "../data/store"
    
    # 原始数据输入路径
    RACES_RAW_DIR = "../data/races"
    PAYOFF_RAW_DIR = "../data/payoff"
    
    # 初始化并一键启动管道
    pipeline = QuantDataIngestionPipeline(store_root=STORE_ROOT_PATH)
    pipeline.run(races_dir=RACES_RAW_DIR, payoff_dir=PAYOFF_RAW_DIR)