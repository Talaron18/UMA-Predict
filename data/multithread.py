import pandas as pd
from bs4 import BeautifulSoup
import requests
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from JRA.data.crawl import parse_race_page, parse_pay_page, HEADERS

data_lock = threading.Lock()

def process_single_day(day, mode):
    """处理单日逻辑的包装函数"""
    date_str = day.strftime("%Y%m%d")
    url = f"https://db.netkeiba.com/race/list/{date_str}/"
    day_results = []
    
    try:
        res = requests.get(url, headers=HEADERS, timeout=10)
        res.encoding = "EUC-JP"
        soup = BeautifulSoup(res.text, "html.parser")
        
        graded_ids = []
        for a in soup.find_all("a", href=re.compile(r"^/race/\d+/$")):
            if any(g in a.get_text() for g in ["(G1)","(G2)","(G3)","(GI)","(GII)","(GIII)"]):
                m = re.search(r"/race/(\d+)/", a["href"])
                if m: graded_ids.append(m.group(1))
        
        graded_ids = list(set(graded_ids))
        if not graded_ids:
            return []

        for rid in graded_ids:
            r_res = requests.get(f"https://db.netkeiba.com/race/{rid}/", headers=HEADERS, timeout=10)
            r_res.encoding = "EUC-JP"
            if mode == "race":
                df = parse_race_page(r_res.text, rid, date_str)
            elif mode == "pay":
                df = parse_pay_page(r_res.text, rid, date_str)
            
            day_results.append(df)
            time.sleep(0.5) 
            
        print(f"[{date_str}] 完成，抓取到 {len(graded_ids)} 场重赏")
        return day_results

    except Exception as e:
        print(f"Error on {date_str}: {e}")
        return []

def run_batch_scrape_multithreaded(start_year, end_year, mode, max_workers=8):
    all_race_data = []
    all_days = []
    
    for year in range(start_year, end_year + 1):
        dates = pd.date_range(f"{year}-01-01", f"{year}-12-31")
        all_days.extend([d for d in dates if d.weekday() >= 5])

    print(f"开始多线程爬取，总计 {len(all_days)} 个日期，线程数: {max_workers}")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交任务
        future_to_day = {executor.submit(process_single_day, day, mode): day for day in all_days}
        
        for future in as_completed(future_to_day):
            result = future.result()
            if result:
                with data_lock:
                    all_race_data.extend(result)

    # 保存逻辑
    if all_race_data:
        final_df = pd.concat(all_race_data, ignore_index=True)
        final_df = final_df.astype({"race_id": str, "date": str})
        output_file = f"{'graded_races' if mode == 'race' else 'payoff'}_{start_year}_{end_year}.parquet"
        final_df.to_parquet(output_file, index=False, engine='pyarrow')
        print(f"\n🎉 成功保存至: {output_file}，共 {len(final_df)} 条记录")
    else:
        print("未抓取到任何重赏数据")

if __name__ == "__main__":
    mode = input("crawling option (pay/race): ").strip()
    if mode not in ("pay", "race"):
        print("Invalid Input!")
        exit()
    st = int(input("year start: "))
    en = int(input("year end: "))
    run_batch_scrape_multithreaded(st, en, mode, max_workers=5)