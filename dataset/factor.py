import os
import json
import numpy as np
import pandas as pd

def inspect_horse_bin_data(store_root: str, horse_id: str):
    """
    探测并打印指定马匹存储在底层 bin 文件中的原子特征数据
    """
    horse_path = os.path.join(store_root, "horse", str(horse_id))
    meta_path = os.path.join(horse_path, "meta.json")
    bin_path = os.path.join(horse_path, "features.bin")
    dates_path = os.path.join(horse_path, "dates.npy")
    map_path = os.path.join(store_root, "metadata", "feature_map.json")

    # 1. 安全检查
    if not os.path.exists(horse_path):
        print(f"❌ 未找到马匹 {horse_id} 的存储目录{horse_path}，请检查 ID 是否正确。")
        return

    # 2. 读取元数据和特征映射表，获取矩阵的 Shape 和列名
    with open(meta_path, "r") as f:
        meta = json.load(f)
    shape = tuple(meta["shape"])  # 例如: (5, 15) -> 5场比赛，15个原子特征

    with open(map_path, "r", encoding="utf-8") as f:
        feature_to_idx = json.load(f)
    # 将字典反转为 id -> name 的列表，确保列顺序严格对齐
    columns = [None] * len(feature_to_idx)
    for name, idx in feature_to_idx.items():
        columns[idx] = name

    # 3. 使用 mmap 模式极速映射二进制特征，加载日期序列
    features_mmap = np.memmap(bin_path, dtype='float32', mode='r', shape=shape)
    dates = np.load(dates_path, mmap_mode='r')

    # 4. 转换成 Pandas DataFrame 方便人类阅读
    # 注意：此时虽然转成了 DataFrame，但这只是为了在控制台漂亮地 print 出来
    df_inspect = pd.DataFrame(np.array(features_mmap), columns=columns)
    df_inspect.insert(0, "race_date", dates)  # 把日期插到第一列

    # 5. 打印可视化信息
    print("=" * 60)
    print(f"📊 马匹 ID: {horse_id} 的底层二进制数据分析")
    print(f"📈 矩阵维度 (比赛场数, 原子特征数): {shape}")
    print("=" * 60)
    print("💡 硬盘中的纯二进制数据流转化如下:")
    
    # 调整 pandas 打印配置，防止列被截断
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(df_inspect)
    print("=" * 60)

# ==========================================
# 测试查看入口
# ==========================================
if __name__ == "__main__":
    # 指向你的数据商店根目录
    MY_STORE_ROOT = "../data/store"
    
    # 填入你想查看的任意一个 horse_id (例如你原始数据里的 "2013106126")
    TARGET_HORSE = "1993104054" 
    
    inspect_horse_bin_data(store_root=MY_STORE_ROOT, horse_id=TARGET_HORSE)