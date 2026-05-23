import os
import re
import json
import time
import pandas as pd
import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# 伪装浏览器请求头
HEADERS = {
    'accept-language': 'zh-CN,zh;q=0.9,en-US;q=0.8,en;q=0.7,zh-TW;q=0.6,ja;q=0.5',
    'priority': 'u=0, i',
    'sec-ch-ua': '"Chromium";v="148", "Google Chrome";v="148", "Not/A)Brand";v="99"',
    'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/148.0.0.0 Safari/537.36',
}
# 多线程数据安全锁
data_lock = threading.Lock()


def extract_failed_horse_ids(log_file_path):
    """从错误日志文本中精准提取失败的马匹 ID"""
    if not os.path.exists(log_file_path):
        print(f"❌ 错误：找不到日志文件 {log_file_path}")
        return []
    with open(log_file_path, "r", encoding="utf-8", errors="ignore") as f:
        log_content = f.read()
    failed_ids = re.findall(r"采集马匹\s+(\d{10})", log_content)
    unique_ids = list(dict.fromkeys(failed_ids))
    return unique_ids


def parse_pedigree_table(soup):
    """精准提取 62 个祖先马匹的纯净 ID 和名称，完美过滤垃圾超链接"""
    table = soup.find("table", class_=["blood_table", "detail"])
    if not table:
        return []

    anchors = table.find_all("a", href=re.compile(r"/horse/\w+/"))
    unique_ancestors = {}
    for a in anchors:
        href = a.get("href", "")
        if any(keyword in href for keyword in ["/ped/", "/sire/", "/mare/"]):
            continue
            
        m = re.search(r"/horse/(\w+)/", href)
        if m:
            horse_id = m.group(1)
            name = " ".join(a.stripped_strings)
            if name and horse_id not in unique_ancestors:
                unique_ancestors[horse_id] = {"id": horse_id, "name": name}
                
    return list(unique_ancestors.values())


def process_single_pedigree(horse_id, max_retries=3):
    """单匹马血统抓取线程流"""
    ped_url = f"https://db.netkeiba.com/horse/ped/{horse_id}/"
    soup_ped = None
    
    for attempt in range(max_retries):
        try:
            res = requests.get(ped_url, headers=HEADERS, timeout=12)
            res.encoding = "EUC-JP"
            if "blood_table" in res.text:
                soup_ped = BeautifulSoup(res.text, "html.parser")
                break
        except requests.RequestException:
            time.sleep(1.5 * (attempt + 1))
            
    if not soup_ped:
        return {"horse_id": horse_id, "pedigree_json": "[]", "status": "failed"}

    ancestors = parse_pedigree_table(soup_ped)
    ped_json_str = json.dumps(ancestors, ensure_ascii=False)
    
    time.sleep(0.15)
    return {"horse_id": horse_id, "pedigree_json": ped_json_str, "status": "success"}


def run_pedigree_scraper_chunked(mode="index", input_index_path="./data/datastore1/metadata/horse_index.parquet", start_row=0, log_file_path=None, max_workers=4, chunk_size=20000):
    """
    支持单文件 50MB 限额保护的分卷全量采集器
    :param chunk_size: 💡 每积攒多少条数据就强制落盘成一个独立的 .parquet 文件（2万条纯文本血统数据大约几兆到十几兆，绝对安全低于50MB）
    """
    horse_ids = []

    # 1. 确定数据源
    if mode == "retry":
        if not log_file_path:
            print("❌ 错误：retry 模式下必须指定 log_file_path")
            return
        horse_ids = extract_failed_horse_ids(log_file_path)
        file_prefix = "horse_pedigree_recovered"
        print(f"🎬 [模式：网络错误精准补爬] 目标总量: {len(horse_ids)} 匹马")
    else:
        if not os.path.exists(input_index_path):
            print(f"❌ 错误：找不到大索引文件 {input_index_path}")
            return
        df_index = pd.read_parquet(input_index_path)
        sliced_df = df_index.iloc[start_row:]
        horse_ids = sliced_df["horse_id"].dropna().astype(str).unique().tolist()
        file_prefix = f"horse_pedigree_from_row_{start_row}"
        print(f"🎬 [模式：常规索引断点续爬] 跳过前 {start_row} 行，全量目标: {len(horse_ids)} 匹马")

    if not horse_ids:
        print("没有可执行的任务，程序安全退出。")
        return

    # 创建一个独立的文件夹来保存这些分卷 Parquet，避免污染根目录
    output_dir = "pedigree_outputs"
    os.makedirs(output_dir, exist_ok=True)

    # 2. 核心分卷调度控制
    current_chunk_data = []
    chunk_index = 1
    completed_count = 0

    print(f"📦 安全控制开启：每满 {chunk_size} 条数据将自动保存为一个独立的 Parquet 文件，确保单个文件远低于 50MB 限额。")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_horse = {executor.submit(process_single_pedigree, hid): hid for hid in horse_ids}
        
        for future in as_completed(future_to_horse):
            try:
                res_dict = future.result()
                if res_dict is not None:
                    with data_lock:
                        current_chunk_data.append(res_dict)
            except Exception:
                pass
                
            completed_count += 1
            
            # 💡 达到分卷阈值，自动打包落盘
            if len(current_chunk_data) >= chunk_size:
                with data_lock:
                    df_chunk = pd.DataFrame(current_chunk_data)
                    chunk_file = os.path.join(output_dir, f"{file_prefix}_part_{chunk_index}.parquet")
                    df_chunk.to_parquet(chunk_file, index=False, engine='pyarrow')
                    print(f"💾 【分卷自动保护】第 {chunk_index} 卷已写满，安全保存至: {chunk_file} (条数: {len(current_chunk_data)})")
                    # 清空当前缓冲区，迎接下一卷
                    current_chunk_data = []
                    chunk_index += 1

            if completed_count % 100 == 0 or completed_count == len(horse_ids):
                print(f"📊 总进度提示: 已完成 {completed_count} / {len(horse_ids)}")

        # 💡 抓取彻底结束后，把最后没攒够 chunk_size 的小尾巴数据存盘
        if current_chunk_data:
            df_chunk = pd.DataFrame(current_chunk_data)
            chunk_file = os.path.join(output_dir, f"{file_prefix}_part_{chunk_index}.parquet")
            df_chunk.to_parquet(chunk_file, index=False, engine='pyarrow')
            print(f"💾 【最终收尾】最后一卷安全保存至: {chunk_file} (条数: {len(current_chunk_data)})")

    print(f"\n🎉 全量采集彻底结束！所有纯净 Parquet 文件均已安全切片并保存在 `{output_dir}/` 文件夹中。")


if __name__ == "__main__":
    # 示例：从原有大索引第 2528 行往后开刷，每 20000 匹马自动切一个物理文件
    run_pedigree_scraper_chunked(
        mode="index",
        input_index_path="./data/datastore1/metadata/horse_index.parquet",
        start_row=0,
        max_workers=4,
        chunk_size=2000  # 💡 2万条纯文本 json 构成的 parquet 大约在 10MB~20MB 左右，非常稳地卡在 50MB 安全线下
    )