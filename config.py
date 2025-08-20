from datetime import date, datetime, timedelta
import calendar
import pandas as pd

# 初期設定
REQ_SHIFT_PATH = 'data/req_shift_8.csv'
# REQ_SHIFT_PATH = 'data/req_shift_t1.csv'
TEMPLATE_PATH = 'data/shift_template.xlsx'
TEMP_SHIFT_PATH = 'output/temp_shift.csv'

YEAR = 2025
MONTH = 8
DAYS_IN_MONTH = 31
SHIFT_TYPES = [
    '休', '夜', '早', '残', '〇', '1', '2', '3', '4', '×',
    '/訪', 'CT', '早日', '残日', '1/', '2/', '3/', '4/', '/休', '休/'
]
HOLIDAY_MAP = {
    '①': ['休'],
    '②': ['休', '×'],
    '③': ['休/'],
    '④': ['/休'],
    '⑤': ['/休', '×'],
}

INPUT_CSV = pd.read_csv(REQ_SHIFT_PATH)
NURSES = [n for n in INPUT_CSV['日付'].dropna().tolist() if n != '曜日']
HOLIDAY_NO_WORKERS = ['御書', '久保', '小嶋', '久保（千）', '田浦']
HOLIDAY_WORKERS = [n for n in NURSES if n not in HOLIDAY_NO_WORKERS]
FULL_OFF_SHIFTS = ['休']
HALF_OFF_SHIFTS = ['休/', '/休', '1/', '2/', '3/', '4/', '/訪']
TARGET_REST_SCORE = 13  # 各看護師が取得したい休みの目標

start_date = datetime(YEAR, MONTH - 1, 21)
dates = [start_date + timedelta(days=i) for i in range(DAYS_IN_MONTH)]
weekday_list = [calendar.day_name[d.weekday()] for d in dates]

def japanese_holidays(year: int) -> list[date]:
    """指定した年の日本の祝日を返します（1980–2099年対応）。"""
    holidays = []

    # 固定祝日
    fixed = [
        (1, 1), (2, 11), (2, 23),
        (4, 29), (5, 3), (5, 4), (5, 5),
        (8, 11), (11, 3), (11, 23),
    ]
    for m, d in fixed:
        holidays.append(date(year, m, d))

    # Happy Monday 制度
    def nth_monday(month: int, nth: int) -> date:
        d0 = date(year, month, 1)
        offset = (0 - d0.weekday()) % 7  # 0=Mon
        return d0 + timedelta(days=offset + 7 * (nth - 1))
    holidays.append(nth_monday(1, 2))   # 成人の日
    holidays.append(nth_monday(7, 3))   # 海の日
    holidays.append(nth_monday(9, 3))   # 敬老の日
    holidays.append(nth_monday(10, 2))  # スポーツの日

    # 春分の日・秋分の日（近似式）
    vernal = int(20.8431 + 0.242194 * (year - 1980) - ((year - 1980) // 4))
    autumn = int(23.2488 + 0.242194 * (year - 1980) - ((year - 1980) // 4))
    holidays.append(date(year, 3, vernal))
    holidays.append(date(year, 9, autumn))

    # 国民の休日（祝日に挟まれた平日）
    holiday_set = set(holidays)
    added = True
    while added:
        added = False
        sorted_h = sorted(holiday_set)
        for i in range(len(sorted_h) - 1):
            if (sorted_h[i + 1] - sorted_h[i]).days == 2:
                mid = sorted_h[i] + timedelta(days=1)
                if mid.weekday() != 6 and mid not in holiday_set:
                    holiday_set.add(mid)
                    added = True
                    break

    # 振替休日（日曜の祝日の翌平日）
    sorted_h = sorted(holiday_set)
    for h in sorted_h:
        if h.weekday() == 6:  # 日曜
            next_day = h + timedelta(days=1)
            while next_day in holiday_set:
                next_day += timedelta(days=1)
            holiday_set.add(next_day)

    return sorted(holiday_set)

def is_japanese_holiday(dt) -> bool:
    """datetime.datetime から祝日かどうかを返す。"""
    return dt.date() in japanese_holidays(dt.year)