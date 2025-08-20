from ortools.sat.python import cp_model
import pandas as pd
import calendar
from datetime import datetime, timedelta
from config import (
    TEMP_SHIFT_PATH, YEAR, MONTH, DAYS_IN_MONTH, SHIFT_TYPES, HOLIDAY_MAP, INPUT_CSV, NURSES, HOLIDAY_NO_WORKERS, is_japanese_holiday
)

start_date = datetime(YEAR, MONTH - 1, 21)
dates = [start_date + timedelta(days=i) for i in range(DAYS_IN_MONTH)]
weekday_list = [calendar.day_name[d.weekday()] for d in dates]

# ==========モデル定義=========
model = cp_model.CpModel()
x = {}
for n in NURSES:
    for d in range(DAYS_IN_MONTH):
        for s in SHIFT_TYPES:
            x[n, d, s] = model.NewBoolVar(f'x_{n}_{d}_{s}')


# 休日フラグを作成（木曜・日曜はfull、土曜はhalf）
holiday_flags = []
for d in dates:
    if is_japanese_holiday(d):
        # 祝日は木曜・日曜扱い（full）
        holiday_flags.append('full')
    elif d.weekday() == 3 or d.weekday() == 6:
        # 木曜または日曜
        holiday_flags.append('full')
    elif d.weekday() == 5:
        # 土曜
        holiday_flags.append('half')
    else:
        holiday_flags.append('none')

# 日付をindex番号に変換する関数を定義
def date_to_index(day):
    return day - 21 if day >= 21 else day + 10


# ==========Strict制約==========
# Strict1: 木日は特定看護師は休み(久保は第2木曜日は/訪(訪看日))
thursday_indices = [i for i, wd in enumerate(weekday_list) if wd == 'Thursday']
second_thu = thursday_indices[1] if len(thursday_indices) >= 2 else None

for d in range(DAYS_IN_MONTH):
    if holiday_flags[d] in ['full']:
        for n in HOLIDAY_NO_WORKERS:
            if n == '久保' and second_thu is not None and d == second_thu:
                model.Add(x['久保', d, '/訪'] == 1)
            else:
                model.Add(x[n, d, '休'] == 1)


# Strict2: 希望休をStrictに反映する
# 各看護師ごとに「休み希望日」と「希望種別」をセットで格納
day_cols = [col for col in INPUT_CSV.columns if str(col).isdigit()]
req_dayoff = {}
for nurse in NURSES:
    reqs = []
    for col in day_cols:
        val = INPUT_CSV.loc[INPUT_CSV['日付'] == nurse, col]
        typ = str(val.values[0]).strip() if col in INPUT_CSV.columns and not val.empty else ''
        if typ in HOLIDAY_MAP:
            day = int(col)
            reqs.append((day, typ))  # (日付, 希望種別)
    req_dayoff[nurse] = reqs

shift_types = {
    '①': '休',
    '②': '休',
    '③': '休/',
    '④': '/休',
    '⑤': '/休'
}

for nurse, reqs in req_dayoff.items():
    for day, typ in reqs:
        idx = date_to_index(day)
        if typ in ['①', '②']:
            shift = '休'
        elif typ == '③':
            shift = '休/'
        elif typ in ['④', '⑤']:
            shift = '/休'
        else:
            continue
        model.Add(x[nurse, idx, shift] == 1)
        for s in SHIFT_TYPES:
            if s != shift:
                model.Add(x[nurse, idx, s] == 0)


# Strict3: 夜勤を各日に必ず1人、かつ均等に入れる
YAKIN_WORKERS = ['樋渡', '中山', '川原田', '友枝', '奥平', '前野', '森園'] # 「御書」「三好」「久保」「三好」「前野」「田浦」は夜勤無し

for d in range(DAYS_IN_MONTH):
    model.Add(sum(x[n, d, '夜'] for n in YAKIN_WORKERS) == 1)

total_night_days = sum(1 for d in range(DAYS_IN_MONTH))
base, rem = divmod(total_night_days, len(YAKIN_WORKERS))
night_counts = [model.NewIntVar(base, base + (1 if i < rem else 0), f'{n}_night_count') for i, n in enumerate(YAKIN_WORKERS)]
for i, n in enumerate(YAKIN_WORKERS):
    model.Add(night_counts[i] == sum(x[n, d, '夜'] for d in range(DAYS_IN_MONTH)))

# Strict3.1: 夜勤を同じ人に連続して2日おきに入れない（夜→×→夜→×の防止）
for n in YAKIN_WORKERS:
    for d in range(DAYS_IN_MONTH - 3):
        # 例: day d に夜勤 → d+1に× → d+2に夜勤 → d+3に× は避けたい
        model.AddBoolOr([
            x[n, d, '夜'].Not(),
            x[n, d + 2, '夜'].Not()
        ])

# Strict4: 夜勤の翌日は必ず「×」
for n in YAKIN_WORKERS:
    for d in range(DAYS_IN_MONTH - 1):
        model.Add(x[n, d + 1, '×'] == x[n, d, '夜'])
        for s in SHIFT_TYPES:
            if s != '×':
                model.Add(x[n, d, '夜'] + x[n, d + 1, s] <= 1)


# ==========最適化==========
solver = cp_model.CpSolver()
status = solver.Solve(model)

if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
    result = []
    for n in NURSES:
        row = []
        for d in range(DAYS_IN_MONTH):
            shift = ''
            for s in SHIFT_TYPES:
                if solver.Value(x[n, d, s]):
                    shift = s
                    break
            row.append(shift)
        result.append([n] + row)
    columns = ['nurse'] + [f'day_{i}' for i in range(DAYS_IN_MONTH)]
    df = pd.DataFrame(result, columns=columns)
    df.to_csv(TEMP_SHIFT_PATH, index=False, encoding='utf-8-sig')
    print(f"✅ シフトCSVを {TEMP_SHIFT_PATH} に保存しました。")
else:
    print("❌ 解が見つかりませんでした。")