import pandas as pd
import random
import datetime
import calendar
from datetime import datetime, timedelta
from config import (
    TEMP_SHIFT_PATH, YEAR, MONTH, DAYS_IN_MONTH, SHIFT_TYPES, NURSES, FULL_OFF_SHIFTS, HALF_OFF_SHIFTS, TARGET_REST_SCORE, is_japanese_holiday
)

KUBO_NAME = "久保"
CT_SHIFT_NAME = "CT"
CT_CANDIDATES = ["三好", "前野"]
OUTPUT_TEMP_PATH = "output/temp_shift_2.csv"

start_date = datetime(YEAR, MONTH - 1, 21)
dates = [start_date + timedelta(days=i) for i in range(DAYS_IN_MONTH)]
weekday_list = [calendar.day_name[d.weekday()] for d in dates]

# --- 1. データ読み込み ---
df = pd.read_csv("output/temp_shift.csv", index_col=0)

# --- 2. 対象の日付列のみ抽出 ---
date_cols = df.columns.tolist()

rest_score_full = ["休"]
rest_score_half = ["休/", "/休"]
rest_shifts_all = rest_score_full + rest_score_half + ["1/", "2/", "3/", "4/", "/訪"]

# 3. 久保の現在の休みスコアをカウント（休:2点、半休:1点）
kubos_current_rest_score = 0
for shift in df.loc[KUBO_NAME, date_cols]:
    if shift in rest_score_full:
        kubos_current_rest_score += 2
    elif shift in rest_score_half or shift in ["1/", "2/", "3/", "4/", "/訪"]:
        kubos_current_rest_score += 1

needed_rest_score = TARGET_REST_SCORE * 2 - kubos_current_rest_score
print(f"✅ 現在の休みスコア: {kubos_current_rest_score/2}, 追加必要: {needed_rest_score/2}")

# 4. 久保が休んでいない日からランダムに追加割り当て
non_rest_days = [day for day in date_cols if df.at[KUBO_NAME, day] not in rest_shifts_all]
random.shuffle(non_rest_days)

for day in non_rest_days:
    if needed_rest_score < 0:
        break
    if needed_rest_score >= 0:
        df.at[KUBO_NAME, day] = "休"
        needed_rest_score -= 2

#
# --- 5. 久保が休みの日にはCTを三好・前野に交互で割り当て ---
ct_alternator = 0
for i, day in enumerate(date_cols):
    if df.at[KUBO_NAME, day] == "休":
        weekday = weekday_list[i]
        if weekday in ["Thursday", "Sunday"] or is_japanese_holiday(dates[i]):
            continue
        candidate = CT_CANDIDATES[ct_alternator % len(CT_CANDIDATES)]
        df.at[candidate, day] = CT_SHIFT_NAME
        ct_alternator += 1

# --- 6. 出力 ---
df.to_csv(OUTPUT_TEMP_PATH, encoding='utf-8-sig')
print(f"✅ 出力完了: {OUTPUT_TEMP_PATH}")
