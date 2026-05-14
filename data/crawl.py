import pandas as pd
from bs4 import BeautifulSoup
import requests
import re
import time

HEADERS = {
    'accept-language': 'zh-CN,zh;q=0.9,en-US;q=0.8,en;q=0.7,zh-TW;q=0.6,ja;q=0.5',
    'priority': 'u=0, i',
    'sec-ch-ua': '"Chromium";v="148", "Google Chrome";v="148", "Not/A)Brand";v="99"',
    'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/148.0.0.0 Safari/537.36',
}

def safe_float(x):
    try: return float(x.replace(',', ''))
    except: return None

def safe_int(x):
    try: return int(x.replace(',', ''))
    except: return None

def parse_course(course_text):
    surface = "芝" if "芝" in course_text else ("ダート" if "ダ" in course_text else "障害")
    direction = "左" if "左" in course_text else ("右" if "右" in course_text else "直线")
    dist_m = re.search(r"(\d+)m", course_text)
    distance = safe_int(dist_m.group(1)) if dist_m else None
    return surface, direction, distance

def extract_id(text, pattern):
    if text:
        m = re.search(pattern, text)
        return m.group(1) if m else None
    return None

def parse_race_page(html_text, race_id, race_date):
    soup = BeautifulSoup(html_text, "html.parser")
    # 提取环境信息
    intro = soup.find("div", class_="data_intro")
    surface, direction, distance, weather, track_cond = [None]*5
    if intro and intro.find("span"):
        txt = intro.find("span").get_text(" ", strip=True)
        parts = [p.strip() for p in txt.split("/")]
        if len(parts) >= 1: surface, direction, distance = parse_course(parts[0])
        if len(parts) >= 2: weather = parts[1].replace("天候 :", "").strip()
        if len(parts) >= 3: track_cond = parts[2].split(":")[-1].strip()

    table = soup.find("table", class_=re.compile(r"race_table"))
    if not table: return pd.DataFrame()

    rows = table.find_all("tr")
    headers = [th.get_text(strip=True) for th in rows[0].find_all(["th", "td"])]
    col_map = {name: i for i, name in enumerate(headers)}

    data = []
    for tr in rows[1:]:
        cols = tr.find_all("td")
        if not cols: continue
        def g(k1, k2=None):
            idx = col_map.get(k1) or col_map.get(k2)
            return cols[idx].get_text(strip=True) if idx is not None else None

        hw_txt = g("馬体重", "马体重")
        h_w, h_w_diff = None, None
        if hw_txt:
            m = re.search(r"(\d+)\(([-+]?\d+)\)", hw_txt)
            if m: h_w, h_w_diff = safe_int(m.group(1)), safe_int(m.group(2))
        horse_id = None
        horse_idx = col_map.get("馬名") or col_map.get("马名")
        if horse_idx is not None:
            horse_tag = cols[horse_idx].find("a")
            if horse_tag:
                horse_id = extract_id(horse_tag.get("href"), r"/horse/([^/]+)/")

        data.append({
            "race_id": race_id, "date": race_date,
            "surface": surface, "direction": direction, "distance": distance,
            "weather": weather, "condition": track_cond,
            "horse_id":horse_id,
            "gender_age": g("性齢", "性龄"),
            "weight_carried": safe_float(g("斤量")),
            "jockey": g("騎手", "骑手"),
            "time": g("タイム", "时间"),
            "margin": g("着差"),
            "passing": g("通過", "通过"),
            "last_3f": safe_float(g("上り")),
            "odds": safe_float(g("単勝", "单胜")),
            "popularity": safe_int(g("人気", "人气")),
            "horse_weight": h_w, "horse_weight_diff": h_w_diff,
            "trainer": g("調教師", "调教师"), "owner": g("馬主", "马主")
        })
    return pd.DataFrame(data)

def parse_pay_page(html_text, race_id, race_date):
    soup = BeautifulSoup(html_text, "html.parser")
    tables = soup.find_all("table", class_=re.compile(r"pay_table"))
    if not tables:
        return pd.DataFrame()

    pay_data = {
        "race_id": race_id,
        "date": race_date,
        "Win": None,              # 単勝
        "Place": None,            # 複勝
        "Bracket_Quinella": None, # 枠連
        "Quinella": None,         # 馬連
        "Quinella_Place": None,   # ワイド
        "Exacta": None,           # 馬単
        "Trio": None,             # 3連複
        "Trifecta": None          # 3連単
    }
    mapping = {
        "単勝": "Win", "複勝": "Place", "枠連": "Bracket_Quinella",
        "馬連": "Quinella", "ワイド": "Quinella_Place", "馬単": "Exacta",
        "三連複": "Trio", "三連単": "Trifecta"
    }

    for table in tables:
        rows = table.find_all("tr")
        for tr in rows:
            th = tr.find("th")
            tds = tr.find_all("td")
            
            if th and tds:
                label = th.get_text(strip=True)
                field = None
                for key, value in mapping.items():
                    if key in label: 
                        field = value
                        break
                if field:
                    raw_val = tds[1].get_text("|", strip=True)
                    clean_val = raw_val.replace(",", "").replace("円", "")
                    if pay_data[field]:
                        pay_data[field] += f" | {clean_val}"
                    else:
                        pay_data[field] = clean_val

    return pd.DataFrame([pay_data])

def run_batch_scrape(start_year, end_year, mode):
    all_race_data = []
    
    for year in range(start_year, end_year + 1):
        dates = pd.date_range(f"{year}-01-01", f"{year}-12-31")
        for day in dates:
            if day.weekday() < 5: continue # 只看周六日
            
            date_str = day.strftime("%Y%m%d")
            url = f"https://db.netkeiba.com/race/list/{date_str}/"
            try:
                res = requests.get(url, headers=HEADERS, timeout=10)
                res.encoding = "EUC-JP"
                soup = BeautifulSoup(res.text, "html.parser")
                
                # 筛选重赏赛 ID
                graded_ids = []
                for a in soup.find_all("a", href=re.compile(r"^/race/\d+/$")):
                    if any(g in a.get_text() for g in ["(G1)","(G2)","(G3)","(GI)","(GII)","(GIII)"]):
                        graded_ids.append(re.search(r"/race/(\d+)/", a["href"]).group(1))
                
                graded_ids = list(set(graded_ids))
                if not graded_ids: continue
                
                print(f"[{date_str}] 发现 {len(graded_ids)} 场重赏")
                for rid in graded_ids:
                    r_res = requests.get(f"https://db.netkeiba.com/race/{rid}/", headers=HEADERS)
                    r_res.encoding = "EUC-JP"
                    if mode == "race":
                        df = parse_race_page(r_res.text, rid, date_str)
                    elif mode == "pay":
                        df = parse_pay_page(r_res.text, rid, date_str)

                    all_race_data.append(df)
                    time.sleep(1.5)
                    
            except Exception as e:
                print(f"Error on {date_str}: {e}")

    if all_race_data:
        final_df = pd.concat(all_race_data, ignore_index=True)
        final_df = final_df.astype({"race_id": str, "date": str})
        if mode == "race":
            output_file = f"graded_races_{start_year}_{end_year}.parquet"
        elif mode == "pay":
            output_file = f"payoff_{start_year}_{end_year}.parquet"
        final_df.to_parquet(output_file, index=False, engine='pyarrow')
        print(f"成功保存至: {output_file}，共 {len(final_df)} 条记录")
    else:
        print("未抓取到任何重赏数据")

if __name__ == "__main__":
    mode = input("crawling option: ")
    if mode not in ("pay", "race"):
        print("Invalid Input!")
        exit()
    st = int(input("year start: "))
    en = int(input("year end: "))
    run_batch_scrape(start_year = st, end_year = en, mode=mode)
