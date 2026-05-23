import pandas as pd
from bs4 import BeautifulSoup
import requests
import time
import os
import re
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

HEADERS = {
    'accept-language': 'zh-CN,zh;q=0.9,en-US;q=0.8,en;q=0.7,zh-TW;q=0.6,ja;q=0.5',
    'priority': 'u=0, i',
    'sec-ch-ua': '"Chromium";v="148", "Google Chrome";v="148", "Not/A)Brand";v="99"',
    'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/148.0.0.0 Safari/537.36',
}

data_lock = threading.Lock()

def parse_basic_info(soup, horse_id):
    horse_data = {
        "horse_id": horse_id, "horse_name": "", "eng_name": "", "status": "", 
        "gender": "", "color": "", "birth_date": "", "trainer": "", "owner": "", 
        "breeder": "", "birthplace": "", "prize_money_jra": "", "prize_money_nar": "", "total_results": ""
    }
    try:
        title_div = soup.select_one(".horse_title, .horse_title_name, .db_head_name")
        if title_div:
            name_tag = title_div.find("h1")
            if name_tag: horse_data["horse_name"] = name_tag.get_text(strip=True)
            txt_01_tag = title_div.select_one(".txt_01")
            if txt_01_tag:
                info_parts = txt_01_tag.get_text(strip=True).split()
                if len(info_parts) >= 3:
                    horse_data["status"], horse_data["gender"], horse_data["color"] = info_parts[0], info_parts[1], info_parts[2]
        
        eng_tag = soup.select_one(".eng_name")
        if eng_tag: horse_data["eng_name"] = eng_tag.get_text(strip=True)

        prof_table = soup.select_one(".db_prof_table")
        if prof_table:
            for row in prof_table.find_all("tr"):
                th, td = row.find("th"), row.find("td")
                if th and td:
                    key, val = th.get_text(strip=True), " ".join(td.stripped_strings)
                    if "生年月日" in key: horse_data["birth_date"] = val
                    elif "調教師" in key: horse_data["trainer"] = val
                    elif "馬主" in key: horse_data["owner"] = val
                    elif "生産者" in key: horse_data["breeder"] = val
                    elif "産地" in key: horse_data["birthplace"] = val
                    elif "獲得賞金" in key and "中央" in key: horse_data["prize_money_jra"] = val
                    elif "獲得賞金" in key and "地方" in key: horse_data["prize_money_nar"] = val
                    elif "通算成績" in key: horse_data["total_results"] = val
    except Exception as e:
        print(f"[Basic Parse Error] Failed to analyze {horse_id}: {e}")
    return horse_data

def extract_failed_horse_ids(log_file_path):
    if not os.path.exists(log_file_path):
        print(f"Error: Can't find log at {log_file_path}")
        return []
        
    with open(log_file_path, "r", encoding="utf-8", errors="ignore") as f:
        log_content = f.read()
    failed_ids = re.findall(r"采集马匹\s+(\d{10})", log_content)
    unique_ids = list(dict.fromkeys(failed_ids))
    print(f"Succeeding getting  {len(unique_ids)} ids that failed. ")
    return unique_ids

def parse_history_table(soup):
    table = soup.find("table", class_=["db_h_race_results", "nk_tb_common"])
    if not table:
        return []

    history_list = []
    rows = table.find_all("tr")[1:]  
    for row in rows:
        for hidden in row.find_all(class_="disp_none"):
            hidden.decompose()
        tds = row.find_all("td")
        if len(tds) < 15: 
            continue
        race_record = {}
        try:
            race_record["date"] = " ".join(tds[0].stripped_strings)
            race_record["venue"] = " ".join(tds[1].stripped_strings)
            race_record["weather"] = " ".join(tds[2].stripped_strings)
            race_record["race_num"] = " ".join(tds[3].stripped_strings)
            race_record["race_name"] = " ".join(tds[4].stripped_strings)
            a_tag = tds[4].find("a")
            if a_tag and "race" in a_tag.get("href", ""):
                match = re.search(r"/race/(\d+)/", a_tag["href"])
                race_record["race_id"] = match.group(1) if match else ""
            else:
                race_record["race_id"] = ""
            jockey_td_idx = None
            for idx, td in enumerate(tds):
                if td.find("a", href=re.compile(r"/jockey/")):
                    jockey_td_idx = idx
                    break
            if jockey_td_idx is not None:
                race_record["num_horses"]   = " ".join(tds[jockey_td_idx - 6].stripped_strings) if jockey_td_idx - 6 >= 0 else ""
                race_record["bracket_num"]  = " ".join(tds[jockey_td_idx - 5].stripped_strings) if jockey_td_idx - 5 >= 0 else ""
                race_record["horse_num"]    = " ".join(tds[jockey_td_idx - 4].stripped_strings) if jockey_td_idx - 4 >= 0 else ""
                race_record["odds"]         = " ".join(tds[jockey_td_idx - 3].stripped_strings) if jockey_td_idx - 3 >= 0 else ""
                race_record["popularity"]   = " ".join(tds[jockey_td_idx - 2].stripped_strings) if jockey_td_idx - 2 >= 0 else ""
                race_record["finish_pos"]   = " ".join(tds[jockey_td_idx - 1].stripped_strings) if jockey_td_idx - 1 >= 0 else ""
                
                race_record["jockey"]       = " ".join(tds[jockey_td_idx].stripped_strings)
                race_record["weight"]       = " ".join(tds[jockey_td_idx + 1].stripped_strings) if jockey_td_idx + 1 < len(tds) else ""
                race_record["distance"]     = " ".join(tds[jockey_td_idx + 2].stripped_strings) if jockey_td_idx + 2 < len(tds) else ""
                race_record["track_condition"] = " ".join(tds[jockey_td_idx + 3].stripped_strings) if jockey_td_idx + 3 < len(tds) else ""
            else:
                continue

            all_texts = [" ".join(td.stripped_strings) for td in tds]
            all_texts = [t for t in all_texts if t != ""]  
            
            race_record["time"] = ""
            race_record["margin"] = ""
            race_record["passing_pos"] = ""
            race_record["pace"] = ""
            race_record["last_3f"] = ""
            race_record["horse_weight"] = ""
            race_record["winner_runner_up"] = ""
            race_record["prize"] = ""

            if len(all_texts) >= 1:
                race_record["prize"] = all_texts[-1]
            if len(all_texts) >= 2:
                if not re.search(r"\d+:", all_texts[-2]):  
                    race_record["winner_runner_up"] = all_texts[-2]

            for t_val in all_texts:
                if re.match(r"^\d+:\d+\.\d+$", t_val):
                    race_record["time"] = t_val
                    continue
                if re.match(r"^\d+\(\s*[\+\-]?\d+\s*\)$", t_val):
                    race_record["horse_weight"] = t_val
                    continue
                if re.match(r"^\d+-\d+(-\d+)*$", t_val):
                    race_record["passing_pos"] = t_val
                    continue
                if re.match(r"^\d+\.\d+-\d+\.\d+$", t_val):
                    race_record["pace"] = t_val
                    continue
                if re.match(r"^\d+\.\d+$", t_val):
                    f_val = float(t_val)
                    if 28.0 <= f_val <= 48.0:
                        race_record["last_3f"] = t_val
                        continue

            if race_record["time"]:
                try:
                    time_idx = all_texts.index(race_record["time"])
                    if time_idx + 1 < len(all_texts):
                        potential_margin = all_texts[time_idx + 1]
                        if re.match(r"^\d+\.\d+$", potential_margin) and potential_margin != race_record["last_3f"]:
                            race_record["margin"] = potential_margin
                except ValueError:
                    pass

            history_list.append(race_record)
        except Exception as row_err:
            print(f"[Row Parse Warning] Getting unknown lable, skipping: {row_err}")
            continue
            
    return history_list


def process_single_horse(horse_id):
    base_url = f"https://db.netkeiba.com/horse/{horse_id}/"
    result_url = f"https://db.netkeiba.com/horse/result/{horse_id}/"
    try:
        res_base = requests.get(base_url, headers=HEADERS, timeout=10)
        res_base.encoding = "EUC-JP"
        if "db_prof_table" not in res_base.text:
            print(f"[Skip] Horse ID {horse_id} data damaged, skipping current page")
            return None
            
        soup_base = BeautifulSoup(res_base.text, "html.parser")
        horse_dict = parse_basic_info(soup_base, horse_id)
        time.sleep(0.2)
        res_result = requests.get(result_url, headers=HEADERS, timeout=10)
        res_result.encoding = "EUC-JP"
        if "db_h_race_results" not in res_result.text:
            horse_dict["history"] = json.dumps([], ensure_ascii=False)
            return pd.DataFrame([horse_dict])
        soup_result = BeautifulSoup(res_result.text, "html.parser")
        history_list = parse_history_table(soup_result)
        horse_dict["history"] = json.dumps(history_list, ensure_ascii=False)
        time.sleep(0.3) 
        return pd.DataFrame([horse_dict])
    except Exception as e:
        print(f"[Error] Failed when collecting data of {horse_id}: {e}")
        return None


def run_batch_scrape_horses(input_parquet_path, start_row=0, max_workers=4):
    all_horse_data = []
    if not os.path.exists(input_parquet_path):
        print(f"Error: Failed to locate {input_parquet_path}")
        return
        
    index_df = pd.read_parquet(input_parquet_path)
    sliced_df = index_df.iloc[start_row:]
    horse_ids = sliced_df["horse_id"].dropna().astype(str).unique().tolist()
    print(f"Successfully loaded! Skipping {start_row} lines. Starting from {start_row} ...")
    print(f"Number of horses to collect: {len(horse_ids)}")

    completed_count = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_horse = {executor.submit(process_single_horse, hid): hid for hid in horse_ids}
        for future in as_completed(future_to_horse):
            result = future.result()
            if result is not None and not result.empty:
                with data_lock:
                    all_horse_data.append(result)
            completed_count += 1
            if completed_count % 20 == 0:
                print(f"Progress: {completed_count} / {len(horse_ids)}")

    if all_horse_data:
        final_df = pd.concat(all_horse_data, ignore_index=True)
        output_file = f"horse_perfect_from_row_{start_row}.parquet"
        final_df.to_parquet(output_file, index=False, engine='pyarrow')
        print(f"\n Data collection accomplished. Saved in: {output_file}")
    else:
        print("Failed to collect effective data")


if __name__ == "__main__":
    run_batch_scrape_horses(
        input_parquet_path="horse_index.parquet", 
        start_row=2528, 
        max_workers=4
    )