# =============================
# 1. 必要ライブラリのインポート
# =============================
import pandas as pd
import random
import calendar
from datetime import datetime, timedelta
from config import (
    YEAR, MONTH, DAYS_IN_MONTH, FULL_OFF_SHIFTS, HALF_OFF_SHIFTS, TARGET_REST_SCORE, is_japanese_holiday
)

# =============================
# 2. データ読み込みと初期化
# =============================
def load_and_initialize():
    df = pd.read_csv("output/temp_shift_2.csv", index_col=0)
    fixed_mask = df.notna()
    nurse_names = df.index.tolist()
    date_cols = df.columns.tolist()
    start_date = datetime(YEAR, MONTH - 1, 21)
    days = [col for col in df.columns if col.startswith('day_')]
    dates = [start_date + timedelta(days=i) for i in range(DAYS_IN_MONTH)]
    weekday_list = [calendar.day_name[d.weekday()] for d in dates]
    return df, fixed_mask, date_cols, nurse_names, dates, weekday_list

# =============================
# 3. 休みスコア計算
# =============================
def calculate_rest_scores(df, date_cols, nurse_names):
    scores = {}
    for n in nurse_names:
        total = 0
        for d in date_cols:
            shift = df.at[n, d]
            if shift in FULL_OFF_SHIFTS:
                total += 1
            elif shift in HALF_OFF_SHIFTS:
                total += 0.5
        scores[n] = total
    return scores

    
def initialize_rest_score():
    """
    current_rest_score を初期化する。各看護師の休みスコア（休=2, 休/=1, それ以外=0）を計算。
    グローバル変数 current_rest_score, df, date_cols, nurse_names を利用。
    """
    global current_rest_score, df, date_cols, nurse_names
    current_rest_score = {}
    for n in nurse_names:
        score = 0
        for d in date_cols:
            shift = df.at[n, d]
            if shift == '休':
                score += 2
            elif shift in ['休/', '/休', '/訪']:
                score += 1
        current_rest_score[n] = score


def initialize_shift_counts():
    """
    shift_counts_weekday, shift_counts_saturday を初期化する。
    各看護師ごとに各シフトの割り当て回数を0で初期化。
    グローバル変数 shift_counts_weekday, shift_counts_saturday, nurse_names を利用。
    """
    global shift_counts_weekday, shift_counts_saturday, nurse_names
    weekday_shifts = ["1", "2", "3", "4", "早", "残", "〇", "CT", "2・CT"]
    saturday_shifts = ["1/", "2/", "3/", "4/", "早", "残", "〇"]
    shift_counts_weekday = {n: {s: 0 for s in weekday_shifts} for n in nurse_names}
    shift_counts_saturday = {n: {s: 0 for s in saturday_shifts} for n in nurse_names}

# =============================
# 4. 割り当て処理：日別に関数化
# =============================
def assign_weekday_shift(d, col):
    """
    平日（月火水金）のシフト割り当て（外来4人、病棟3人、夜勤CT）
    グローバル変数を参照し、d, colのみ引数で受け取る
    """
    global df, weekday_list, nurse_names, fixed_mask, shift_counts_weekday
    weekday = weekday_list[d]
    assigned_nurses = set()
    busy_shifts = ['休', '休/', '/休', '夜', '×', '/訪']
    available_nurses = [
        n for n in nurse_names if df.at[n, col] not in busy_shifts and not fixed_mask.at[n, col]
    ]

    n_to_assign = 8 if len(available_nurses) >= 8 else 7

    # CT or 2・CT 割り当て
    if n_to_assign == 8:
        if not fixed_mask.at["久保", col]: # 平日 & 久保出勤 & 8人以上出勤可能
            df.at["久保", col] = "CT"
            shift_counts_weekday['久保']['CT'] += 1
            assigned_nurses.add('久保')
    else:
        if not fixed_mask.at["久保", col]: # 平日 & 久保出勤 & 7人
            df.at["久保", col] = "2・CT"
            shift_counts_weekday['久保']['2・CT'] += 1
            assigned_nurses.add('久保')

    # 外来（1〜4）割り当て
    if n_to_assign == 8: # 平日 & 8人以上出勤可能 → 外来['1', '2', '3', '4']
        gai_shift = random.sample(['1', '2', '3', '4'], k=4)
    else: # 平日 & 7人 → 外来['1', '3', '4']
        gai_shift = random.sample(['1', '3', '4'], k=3)
    # 外来メイン看護師に優先して割り当て
    gai_nurses = [n for n in ['小嶋', '久保（千）', '田浦'] if n in available_nurses]
    assigned_gai = set()
    for s in gai_shift:
        if gai_nurses:
            count_dict = {n: shift_counts_weekday[n][s] for n in gai_nurses if n not in assigned_gai}
            if count_dict:
                min_count = min(count_dict.values())
                candidates = [n for n, c in count_dict.items() if c == min_count]
                assign = sorted(candidates)[0]
                if not fixed_mask.at[assign, col]:
                    df.at[assign, col] = s
                    shift_counts_weekday[assign][s] += 1
                    assigned_nurses.add(assign)
                    assigned_gai.add(assign)
    remain = len(gai_shift) - len(assigned_gai)
    if remain > 0:
        other_candidates = [
            n
            for n in available_nurses
            if n not in assigned_nurses
            and n != '御書'
            and not fixed_mask.at[n, col]
        ]
        for s in gai_shift[len(assigned_gai):]:
            filtered_candidates = [n for n in other_candidates if n != '御書']
            if filtered_candidates:
                min_count = min(shift_counts_weekday[n][s] for n in filtered_candidates)
                assign = min([n for n in filtered_candidates if shift_counts_weekday[n][s] == min_count])
                if not fixed_mask.at[assign, col]:
                    df.at[assign, col] = s
                    shift_counts_weekday[assign][s] += 1
                    assigned_nurses.add(assign)
                    other_candidates.remove(assign)

    # 病棟（早・残・〇）
    byoto_shifts = ['早', '残', '〇']
    remain_candidates = [n for n in available_nurses if n not in assigned_nurses]
    for s in byoto_shifts:
        if remain_candidates:
            min_count = min(shift_counts_weekday[n][s] for n in remain_candidates)
            assign = min([n for n in remain_candidates if shift_counts_weekday[n][s] == min_count])
            if not fixed_mask.at[assign, col]:
                df.at[assign, col] = s
                shift_counts_weekday[assign][s] += 1
                assigned_nurses.add(assign)
                remain_candidates.remove(assign)

    # 余った人に休みを割当
    remain_nurses = [
        n for n in nurse_names
        if (df.at[n, col] == '' or pd.isna(df.at[n, col]))
        and n in available_nurses
        and not fixed_mask.at[n, col]
    ]
    assign_rest_shifts(remain_nurses, col)

def assign_holiday_shift(d, col):
    """
    木曜・日曜・祝日（B日程）のシフト割り当て。
    - 「早日」「残日」を1人ずつ均等に割り当て
    - 残りの空白には休み希望に応じて「休」または「休/」を割り当て
    グローバル変数を活用し、d, colのみ引数で受け取る
    """
    global df, weekday_list, nurse_names, fixed_mask
    weekday = weekday_list[d]
    busy_shifts = ['休', '休/', '/休', '×', '夜', '/訪']

    # 候補：勤務可能かつ未固定の看護師
    candidates = [
        n for n in nurse_names
        if df.at[n, col] not in busy_shifts and not fixed_mask.at[n, col]
    ]

    # 早日割当
    early_counts = {n: (df == '早日').loc[n].sum() for n in candidates}
    assign_early = None
    if early_counts:
        min_early = min(early_counts.values())
        early_candidates = [n for n in candidates if early_counts[n] == min_early]
        assign_early = sorted(early_candidates)[0]
        df.at[assign_early, col] = '早日'
        candidates.remove(assign_early)

    # 残日割当
    late_counts = {n: (df == '残日').loc[n].sum() for n in candidates}
    assign_late = None
    if late_counts:
        min_late = min(late_counts.values())
        late_candidates = [n for n in candidates if late_counts[n] == min_late]
        assign_late = sorted(late_candidates)[0]
        df.at[assign_late, col] = '残日'
        candidates.remove(assign_late)

    # 残りの候補に休み割当
    assign_rest_shifts(candidates, col)

    # まだ空欄の人（全看護師から）の補填
    remain_nurses = [
        n for n in nurse_names
        if (df.at[n, col] == '' or pd.isna(df.at[n, col]))
        and df.at[n, col] not in busy_shifts
        and not fixed_mask.at[n, col]
    ]
    assign_rest_shifts(remain_nurses, col)

def assign_saturday_shift(d, col):
    """
    土曜（C日程）のシフト割り当て。
    - 「久保」が出勤の場合「2/」優先
    - 土曜担当外来優先
    - 病棟シフト割り当て
    - 休み割り当て
    グローバル変数 df, nurse_names, fixed_mask, shift_counts_saturday, 土曜担当 を参照。
    """
    global df, nurse_names, fixed_mask, shift_counts_saturday, 土曜担当
    assigned_nurses = set()
    busy_shifts = ['休', '休/', '/休', '×', '夜']
    gai_candidates = []
    gai_shift = []
    土曜_gai = []

    # 1. 外来シフト割当
    if '久保' in nurse_names and df.at['久保', col] not in busy_shifts and not fixed_mask.at['久保', col]:
        df.at['久保', col] = '2/'
        shift_counts_saturday['久保']['2/'] += 1
        assigned_nurses.add('久保')
        gai_shift = ['1/', '3/', '4/']
        土曜_gai = [n for n in 土曜担当 if n != '久保']
    else:
        gai_shift = ['1/', '2/', '3/', '4/']
        土曜_gai = list(土曜担当)

    # 外来：土曜担当優先で割当
    gai_assigned = []
    shift_to_nurse = {}
    for s, nurse in zip(gai_shift, 土曜_gai):
        if (
            nurse in nurse_names
            and df.at[nurse, col] not in busy_shifts
            and not fixed_mask.at[nurse, col]
            and nurse != '御書'
        ):
            df.at[nurse, col] = s
            shift_counts_saturday[nurse][s] += 1
            assigned_nurses.add(nurse)
            gai_assigned.append(nurse)
            shift_to_nurse[s] = nurse

    # 残り外来シフトが埋まっていなければ、他の看護師から均等割当
    remain_gai = [s for s in gai_shift if s not in shift_to_nurse]
    remain_num = len(remain_gai)
    if remain_num > 0:
        other_candidates = [
            n for n in nurse_names
            if n not in assigned_nurses
            and n != '御書'
            and not fixed_mask.at[n, col]
            and df.at[n, col] not in busy_shifts
        ]
        for s in remain_gai:
            if other_candidates:
                min_count = min(shift_counts_saturday[n][s] for n in other_candidates)
                assign = min([n for n in other_candidates if shift_counts_saturday[n][s] == min_count])
                df.at[assign, col] = s
                shift_counts_saturday[assign][s] += 1
                assigned_nurses.add(assign)
                other_candidates.remove(assign)

    # 2. 病棟シフト（早・残・〇）も最小割当順に選出
    byoto_shifts = ['早', '残', '〇']
    remain_candidates = [
        n for n in nurse_names
        if n not in assigned_nurses
        and df.at[n, col] not in busy_shifts
        and not fixed_mask.at[n, col]
    ]
    for s in byoto_shifts:
        if remain_candidates:
            # 病棟シフトの割当回数が最も少ない人を選ぶ
            count_dict = {n: (df.loc[n] == s).sum() for n in remain_candidates}
            min_count = min(count_dict.values())
            assign = min([n for n in remain_candidates if count_dict[n] == min_count])
            df.at[assign, col] = s
            assigned_nurses.add(assign)
            remain_candidates.remove(assign)

    # 3. 残り空欄には休みを割当
    remain_nurses = [
        n for n in nurse_names
        if (df.at[n, col] == '' or pd.isna(df.at[n, col]))
        and df.at[n, col] not in busy_shifts
        and not fixed_mask.at[n, col]
    ]
    assign_rest_shifts(remain_nurses, col)

def assign_fallback_rest(d, col):
    """
    その他の日に、空欄セルの看護師に対して休みシフト（休、休/など）を割り当てる。
    - 空欄かつbusy_shiftsでないかつ未固定の看護師を抽出
    - assign_rest_shifts()で休みスコアに基づき割り当て
    グローバル変数 df, nurse_names, fixed_mask を明示的に使用
    """
    global df, nurse_names, fixed_mask
    # 休みや夜勤など既に割り当て済みシフト
    busy_shifts = ['休', '休/', '/休', '夜', '×', '/訪']
    # 空欄セルかつbusy_shiftsでなく、未固定の看護師を取得
    remain_nurses = [
        n for n in nurse_names
        if (df.at[n, col] == '' or pd.isna(df.at[n, col]))
        and df.at[n, col] not in busy_shifts
        and not fixed_mask.at[n, col]
    ]
    # 空欄セルがあれば休み割り当て関数を呼び出し
    if remain_nurses:
        assign_rest_shifts(remain_nurses, col)

def assign_rest_shifts(nurses, col):
    """
    看護師ごとの休み不足スコアに応じて、空いているシフトに
    「休」または「半休（休/）」を優先的に割り当てる。
    current_rest_score は 2点満点方式（休=2, 半休=1）を使用。
    グローバル変数 df, fixed_mask, current_rest_score, TARGET_REST_SCORE を参照。
    """
    global df, fixed_mask, current_rest_score, TARGET_REST_SCORE

    # 1. 割り当て対象者の休み不足スコアを計算（目標値2点×TARGET_REST_SCOREとの差分）
    need = {}
    for n in nurses:
        if fixed_mask.at[n, col]:
            continue  # すでに確定している場合はスキップ
        # 目標点との差分（2点制: 休=2, 休/=1）
        need_score = TARGET_REST_SCORE * 2 - current_rest_score.get(n, 0)
        need[n] = need_score

    # 2. 休み不足の多い順に並べて割り当て
    sorted_nurses = sorted(need, key=lambda n: need[n], reverse=True)
    for n in sorted_nurses:
        if fixed_mask.at[n, col]:
            continue  # すでに確定している場合はスキップ
        remaining = need[n]
        if remaining <= 0:
            continue  # すでに目標を満たしている場合は割り当て不要

        # 休みの割り当て方針
        # 休み不足が2点以上→「休」を割り当て
        # 1点だけ不足→「休/」を割り当て
        # 0点以下は割り当て不要
        if remaining >= 2:
            df.at[n, col] = '休'
            current_rest_score[n] = current_rest_score.get(n, 0) + 2
        elif remaining == 1:
            df.at[n, col] = '休/'
            current_rest_score[n] = current_rest_score.get(n, 0) + 1
        # それ以外（0以下）は何もしない

# =============================
# 5. 休み最適化処理（共通）
# =============================
def balance_rest_days():
    """
    各看護師の休日日数に偏りがある場合（差が2日以上）、
    出勤日と休みをスワップしてバランスを取る。
    - 夜勤の翌日の×は対象外
    - すでに休みまたは半休の日にはスワップしない
    """
    global df, nurse_names, date_cols, fixed_mask, FULL_OFF_SHIFTS, HALF_OFF_SHIFTS

    # 1. 現在の休日日数をカウント
    totals = {}
    for n in nurse_names:
        total = 0
        for d in date_cols:
            shift = df.at[n, d]
            if shift in FULL_OFF_SHIFTS:
                total += 1
            elif shift in HALF_OFF_SHIFTS:
                total += 0.5
        totals[n] = total

    # 2. 休日日数の最大と最小の差が2日以上なら調整
    while max(totals.values()) - min(totals.values()) > 2:
        high = max(totals, key=totals.get)  # 休みが多い人
        low = min(totals, key=totals.get)   # 休みが少ない人
        moved = False

        for idx, col in enumerate(date_cols):
            high_shift = df.at[high, col]
            low_shift = df.at[low, col]

            if fixed_mask.at[high, col] or fixed_mask.at[low, col]:
                continue

            # 夜勤明けの「×」は除外
            if (
                high_shift == '×'
                and idx > 0
                and df.at[high, date_cols[idx - 1]] == '夜'
            ):
                continue
            if (
                low_shift == '×'
                and idx > 0
                and df.at[low, date_cols[idx - 1]] == '夜'
            ):
                continue

            # 夜勤そのものは動かさない
            if high_shift == '夜' or low_shift == '夜':
                continue

            # 高休日日の休みを、低休日日の出勤と交換できるか？
            if high_shift in FULL_OFF_SHIFTS and low_shift not in FULL_OFF_SHIFTS + HALF_OFF_SHIFTS:
                df.at[high, col], df.at[low, col] = low_shift, '休'
                totals[high] -= 1
                totals[low] += 1
                moved = True
                break

        if not moved:
            break  # これ以上動かせないなら終了

def ensure_min_rest_days_balanced():
    """
    各看護師がTARGET_REST_SCORE（例: 13日）に達していない場合、
    出勤に余裕がある日から「休」または「休/」を割り当てる。
    - 夜勤明け（×）の翌日は休みにしない
    - 出勤が7人以下になる日は避ける
    """
    global df, nurse_names, date_cols, fixed_mask
    rest_totals = {}

    # 1. 各看護師の現在の休日日数（1日休み=1、半休=0.5）を算出
    for n in nurse_names:
        total = 0
        for d in date_cols:
            shift = df.at[n, d]
            if shift in FULL_OFF_SHIFTS:
                total += 1
            elif shift in HALF_OFF_SHIFTS:
                total += 0.5
        rest_totals[n] = total

    # 2. 各日付の出勤者数をカウント（非休み者）
    work_count_per_day = {}
    for d in date_cols:
        count = 0
        for n in nurse_names:
            if df.at[n, d] not in FULL_OFF_SHIFTS + HALF_OFF_SHIFTS:
                count += 1
        work_count_per_day[d] = count

    # 3. 出勤者が多い日から優先して休みを割り当てる
    sorted_days = sorted(date_cols, key=lambda d: work_count_per_day[d], reverse=True)

    # 4. 各看護師の不足分を埋める
    for n in nurse_names:
        while rest_totals[n] < TARGET_REST_SCORE:
            inserted = False
            for d in sorted_days:
                idx = date_cols.index(d)
                shift = df.at[n, d]

                if fixed_mask.at[n, d]:
                    continue
                if shift in FULL_OFF_SHIFTS + HALF_OFF_SHIFTS:
                    continue
                if shift == '×' and idx > 0 and df.at[n, date_cols[idx - 1]] == '夜':
                    continue
                if work_count_per_day[d] <= 7:
                    continue

                remaining = TARGET_REST_SCORE - rest_totals[n]

                # 残りスコアに応じて休み or 半休を入れる
                if remaining >= 1:
                    df.at[n, d] = '休'
                    rest_totals[n] += 1
                    work_count_per_day[d] -= 1
                    inserted = True
                    break
                elif remaining >= 0.5:
                    df.at[n, d] = '休/'
                    rest_totals[n] += 0.5
                    work_count_per_day[d] -= 1
                    inserted = True
                    break
            if not inserted:
                break  # 追加できないならループを抜ける

def prevent_seven_day_streaks():
    """
    各看護師について、7日以上の連勤が発生しないように強制的に「休」を入れる。
    - 夜勤とその翌日の「×」には干渉しない
    - 固定されたシフトには変更を加えない
    """
    global df, nurse_names, date_cols, fixed_mask
    off_codes = FULL_OFF_SHIFTS + HALF_OFF_SHIFTS

    for n in nurse_names:
        streak = 0
        for i, col in enumerate(date_cols):
            shift = df.at[n, col]

            if shift in off_codes:
                streak = 0
                continue

            streak += 1

            if streak >= 7:
                # 直近7日のうちで変更可能な日を探して「休」を入れる
                changed = False
                for j in range(i, i - 7, -1):
                    if j < 0:
                        break
                    col_j = date_cols[j]
                    shift_j = df.at[n, col_j]

                    if fixed_mask.at[n, col_j]:
                        continue
                    if shift_j == "夜":
                        continue
                    if (
                        shift_j == "×"
                        and j > 0
                        and df.at[n, date_cols[j - 1]] == "夜"
                    ):
                        continue

                    df.at[n, col_j] = "休"
                    streak = 0
                    changed = True
                    break

                # 上記で変更できなければ、その日自体を「休」に変更（最終手段）
                if not changed and not fixed_mask.at[n, col] and shift != "夜":
                    df.at[n, col] = "休"
                    streak = 0

def prevent_four_day_rest_streaks():
    """
    各看護師の4日以上連続した休み（休、休/、/休）を検出し、
    固定されていない休みを出勤日と交換することで分散させる。
    - 夜勤・夜勤明け「×」には干渉しない
    - 連続休みの中心から変更を試みる
    """
    global df, nurse_names, date_cols, fixed_mask
    off_codes = FULL_OFF_SHIFTS + HALF_OFF_SHIFTS

    for n in nurse_names:
        streak = 0
        for i, col in enumerate(date_cols):
            shift = df.at[n, col]

            if shift in off_codes:
                streak += 1
                if streak >= 4:
                    swapped = False
                    # 直近4日の中で、固定されていない休みを探す
                    for j in range(i, i - 4, -1):
                        if j < 0:
                            continue
                        col_j = date_cols[j]
                        shift_j = df.at[n, col_j]

                        if fixed_mask.at[n, col_j] or shift_j not in off_codes:
                            continue

                        # 休みを出勤日と交換できるか？
                        for k, col_k in enumerate(date_cols):
                            if i - 3 <= k <= i:
                                continue  # 連続休み範囲は除く
                            shift_k = df.at[n, col_k]

                            if fixed_mask.at[n, col_k]:
                                continue
                            if shift_k in off_codes or shift_k == "夜":
                                continue
                            if (
                                shift_k == "×"
                                and k > 0
                                and df.at[n, date_cols[k - 1]] == "夜"
                            ):
                                continue

                            # 入れ替え実行
                            df.at[n, col_j], df.at[n, col_k] = shift_k, shift_j
                            swapped = True
                            streak = 0
                            break
                        if swapped:
                            break

                    if not swapped:
                        streak = 0  # 分散できなければ連続カウントだけリセット
            else:
                streak = 0

# =============================
# 6. シフト割り当てメインループ
# =============================
def assign_all_shifts():
    global df, date_cols, weekday_list, dates, fixed_mask, nurse_names
    for d, col in enumerate(date_cols):
        weekday = weekday_list[d]
        if weekday in ['Monday', 'Tuesday', 'Wednesday', 'Friday'] and not is_japanese_holiday(dates[d]):
            assign_weekday_shift(d, col)
        elif weekday in ['Thursday', 'Sunday'] or is_japanese_holiday(dates[d]):
            assign_holiday_shift(d, col)
        elif weekday == 'Saturday':
            assign_saturday_shift(d, col)
        else:
            assign_fallback_rest(d, col)

# =============================
# 7. 出力処理
# =============================
def export_outputs(df, date_cols, nurse_names):
    # NaN→休
    for d in date_cols:
        df[d] = df[d].apply(lambda x: '休' if pd.isna(x) or x == '' else x)

    # 1〜4をint化
    for d in date_cols:
        df[d] = df[d].apply(lambda x: int(x) if x in ['1', '2', '3', '4'] else x)

    df.to_csv("output/shift_final.csv", encoding="utf-8-sig")
    print("✅ シフト割り振りを shift_final.csv に保存しました。")

    # 休み合計列追加して保存
    df_with_rest = df.copy()
    df_with_rest['休み合計'] = [calculate_rest_scores(df, date_cols, [n])[n] for n in nurse_names]
    df_with_rest.to_csv("output/shift_summary.csv", encoding="utf-8-sig")
    print("✅ 休み合計列付きのシフトCSVを shift_summary.csv に保存しました。")

# =============================
# 8. メイン実行関数
# =============================
def main():
    global df, fixed_mask, date_cols, nurse_names, dates, weekday_list
    global current_rest_score, shift_counts_weekday, shift_counts_saturday, 土曜担当
    df, fixed_mask, date_cols, nurse_names, dates, weekday_list = load_and_initialize()
    initialize_rest_score()
    initialize_shift_counts()
    土曜担当 = ["小嶋", "田浦", "久保（千）"]
    assign_all_shifts()
    balance_rest_days()
    prevent_seven_day_streaks()
    ensure_min_rest_days_balanced()
    # prevent_four_day_rest_streaks()
    export_outputs(df, date_cols, nurse_names)

if __name__ == "__main__":
    main()