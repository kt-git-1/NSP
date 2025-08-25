# =============================
# 1. 必要ライブラリのインポート
# =============================
import pandas as pd
import random
import calendar
from datetime import datetime, timedelta
from ortools.sat.python import cp_model
from config import (
    YEAR, MONTH, DAYS_IN_MONTH, FULL_OFF_SHIFTS, HALF_OFF_SHIFTS, TARGET_REST_SCORE, is_japanese_holiday
)

# =============================
# 2. データ読み込みと初期化
# =============================
def load_and_initialize():
    df = pd.read_csv("output/temp_shift_2.csv", index_col=0)
    # 空文字をNaNに正規化してからfixed_maskを作成（空欄は固定扱いにしない）
    df.replace('', pd.NA, inplace=True)
    fixed_mask = df.notna()
    nurse_names = df.index.tolist()
    date_cols = df.columns.tolist()
    start_date = datetime(YEAR, MONTH - 1, 21)
    days = [col for col in df.columns if col.startswith('day_')]
    dates = [start_date + timedelta(days=i) for i in range(DAYS_IN_MONTH)]
    weekday_list = [calendar.day_name[d.weekday()] for d in dates]
    return df, fixed_mask, date_cols, nurse_names, dates, weekday_list

# =============================
# 追加: 空欄セル判定ヘルパー（先頭に配置）
# =============================
def _is_empty_cell(nm, c):
    v = df.at[nm, c]
    return pd.isna(v) or v == ''

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
    # 御書を優先的に病棟シフト（早・残・〇）へ割り当て
    if '御書' in remain_candidates and not fixed_mask.at['御書', col]:
        # 御書の各病棟シフトの割当回数を確認し、最も少ないシフトを優先
        counts_for_gosho = {s: shift_counts_weekday['御書'][s] for s in byoto_shifts}
        preferred = min(counts_for_gosho, key=counts_for_gosho.get)
        df.at['御書', col] = preferred
        shift_counts_weekday['御書'][preferred] += 1
        assigned_nurses.add('御書')
        remain_candidates.remove('御書')
        # すでに御書に割り当てたシフトは残りの候補から外す
        byoto_shifts = [s for s in byoto_shifts if s != preferred]
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
        if _is_empty_cell(n, col)
        and n in available_nurses
        and not fixed_mask.at[n, col]
    ]
    assign_rest_shifts(remain_nurses, col)
    df.to_csv('output/gairai_temp_shift.csv', encoding='utf-8-sig')

def assign_holiday_shift(d, col):
    """
    木曜・日曜・祝日（B日程）のシフト割り当てを最適化して決定する。
    OR-Tools の CP-SAT を用いて「早日」と「残日」を1人ずつ選び、
    残りは休みシフトを割り当てる。
    """
    global df, nurse_names, fixed_mask
    busy_shifts = ['休', '休/', '/休', '1/', '2/', '3/', '4/', '×', '夜', '/訪']

    for n in nurse_names:
        if df.at[n, col] in ['夜', '×']:
            df.at[n, col] = pd.NA
            fixed_mask.at[n, col] = False

    candidates = [
        n for n in nurse_names
        if (pd.isna(df.at[n, col]) or df.at[n, col] not in busy_shifts) and not fixed_mask.at[n, col] and n != '御書'
    ]
    if not candidates:
        return

    model = cp_model.CpModel()
    early = {n: model.NewBoolVar(f'early_{n}') for n in candidates}
    late = {n: model.NewBoolVar(f'late_{n}') for n in candidates}

    for n in candidates:
        model.Add(early[n] + late[n] <= 1)
    model.Add(sum(early[n] for n in candidates) == 1)
    model.Add(sum(late[n] for n in candidates) == 1)

    early_counts = {n: int((df == '早日').loc[n].sum()) for n in candidates}
    late_counts = {n: int((df == '残日').loc[n].sum()) for n in candidates}
    model.Minimize(sum((early_counts[n] + late_counts[n]) * (early[n] + late[n]) for n in candidates))

    solver = cp_model.CpSolver()
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(f"No feasible assignment for {col}")

    rest_candidates = []
    for n in candidates:
        if solver.BooleanValue(early[n]):
            df.at[n, col] = '早日'
        elif solver.BooleanValue(late[n]):
            df.at[n, col] = '残日'
        else:
            rest_candidates.append(n)

    assign_rest_shifts(rest_candidates, col)

    remain_nurses = [
        n for n in nurse_names
        if _is_empty_cell(n, col) and not fixed_mask.at[n, col]
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
    # 御書を優先的に土曜の病棟シフト（早・残・〇）へ割り当て
    if '御書' in remain_candidates and not fixed_mask.at['御書', col]:
        counts_for_gosho = {s: (df.loc['御書'] == s).sum() for s in byoto_shifts}
        preferred = min(counts_for_gosho, key=counts_for_gosho.get)
        df.at['御書', col] = preferred
        assigned_nurses.add('御書')
        remain_candidates.remove('御書')
        byoto_shifts = [s for s in byoto_shifts if s != preferred]
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
        if _is_empty_cell(n, col)
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
        if _is_empty_cell(n, col)
        and not fixed_mask.at[n, col]
    ]
    # 空欄セルがあれば休み割り当て関数を呼び出し
    if remain_nurses:
        assign_rest_shifts(remain_nurses, col)

def assign_rest_shifts(nurses, col):
    """
    看護師ごとの休み不足スコアに応じて、空いているセルにのみ
    「休」または「半休（休/）」を割り当てる。
    既に何らかのシフトが入っているセルは**絶対に上書きしない**。
    current_rest_score は 2点満点方式（休=2, 休/=1）を使用。
    グローバル変数 df, fixed_mask, current_rest_score, TARGET_REST_SCORE を参照。
    """
    global df, fixed_mask, current_rest_score, TARGET_REST_SCORE

    def _is_empty(nm, c):
        v = df.at[nm, c]
        return pd.isna(v) or v == ''

    # 1. 対象者の休み不足スコアを計算（2点制: 休=2, 休/=1）
    need = {}
    for n in nurses:
        # 固定セル or すでに何か入っているセルは対象外
        if fixed_mask.at[n, col]:
            continue
        if not _is_empty(n, col):
            continue
        need_score = TARGET_REST_SCORE * 2 - current_rest_score.get(n, 0)
        need[n] = need_score

    if not need:
        return

    # 2. 休み不足が大きい順に割り当て
    sorted_nurses = sorted(need, key=lambda n: need[n], reverse=True)
    for n in sorted_nurses:
        # 再確認：この時点でも空欄・未固定のみを割り当て
        if fixed_mask.at[n, col]:
            continue
        if not _is_empty(n, col):
            continue

        remaining = need[n]
        if remaining <= 0:
            continue

        # 休みの割り当て方針
        # 2点以上不足 → 「休」
        # 1点不足       → 「休/」
        if remaining >= 2:
            if not fixed_mask.at[n, col] and _is_empty_cell(n, col):
                df.at[n, col] = '休'
                current_rest_score[n] = current_rest_score.get(n, 0) + 2
        elif remaining == 1:
            if not fixed_mask.at[n, col] and _is_empty_cell(n, col):
                df.at[n, col] = '休/'
                current_rest_score[n] = current_rest_score.get(n, 0) + 1
        # 0以下は割り当てなし

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

    # 1) 現在の休日日数（1日=1, 半休=0.5）を集計
    totals = {}
    for n in nurse_names:
        t = 0
        for d in date_cols:
            s = df.at[n, d]
            if pd.isna(s):
                continue
            if s in FULL_OFF_SHIFTS:
                t += 1
            elif s in HALF_OFF_SHIFTS:
                t += 0.5
        totals[n] = t

    # 2) 差が2以上のあいだ、同一日の安全スワップで均し
    changed = True
    while changed and (max(totals.values()) - min(totals.values()) > 2):
        high = max(totals, key=totals.get)
        low = min(totals, key=totals.get)
        changed = False

        # highが休み、lowが出勤の同じ日を探してスワップ
        for col in date_cols:
            if fixed_mask.at[high, col] or fixed_mask.at[low, col]:
                continue
            s_high = df.at[high, col]
            s_low = df.at[low, col]
            if s_high not in (FULL_OFF_SHIFTS + HALF_OFF_SHIFTS):
                continue
            if s_low in (FULL_OFF_SHIFTS + HALF_OFF_SHIFTS):
                continue

            # 半休か1日休みかで、totalsの増減を決める
            off_delta_high = 1 if s_high in FULL_OFF_SHIFTS else 0.5

            if _swap_between_nurses_if_safe(col, nurse_off=high, nurse_work=low):
                totals[high] -= off_delta_high
                totals[low] += off_delta_high
                changed = True
                break

        # スワップできなければ終了（これ以上壊さずに均せない）
        if not changed:
            break

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
                # 空欄セルのみ対象（上書き禁止）
                if not _is_empty_cell(n, d):
                    continue
                if shift == '×' and idx > 0 and df.at[n, date_cols[idx - 1]] == '夜':
                    continue
                if work_count_per_day[d] <= 7:
                    continue

                remaining = TARGET_REST_SCORE - rest_totals[n]

                # 残りスコアに応じて休み or 半休を入れる
                if remaining >= 1:
                    if not fixed_mask.at[n, d] and _is_empty_cell(n, d):
                        df.at[n, d] = '休'
                        rest_totals[n] += 1
                        work_count_per_day[d] -= 1
                        inserted = True
                        break
                elif remaining >= 0.5:
                    if not fixed_mask.at[n, d] and _is_empty_cell(n, d):
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

                    if fixed_mask.at[n, col_j] or shift_j == '夜':
                        continue
                    if shift_j == '×' and j > 0 and df.at[n, date_cols[j - 1]] == '夜':
                        continue

                    # 同じ日でOFFの誰かと安全スワップを試す
                    for m in nurse_names:
                        if m == n:
                            continue
                        sm = df.at[m, col_j]
                        if fixed_mask.at[m, col_j]:
                            continue
                        if sm not in (FULL_OFF_SHIFTS + HALF_OFF_SHIFTS):
                            continue
                        if _swap_between_nurses_if_safe(col_j, nurse_off=m, nurse_work=n):
                            streak = 0
                            changed = True
                            break
                    if changed:
                        break

                # 上記で変更できなければ、その日自体を「休」に変更（最終手段）
                if not changed and not fixed_mask.at[n, col] and shift != "夜":
                    old = df.at[n, col]
                    df.at[n, col] = "休"
                    if not _valid_day_configuration(col):
                        # revert if breaks the day
                        df.at[n, col] = old
                    else:
                        streak = 0

# === Safety helpers for swap validation ===
def _is_off(shift):
    return shift in (FULL_OFF_SHIFTS + HALF_OFF_SHIFTS)

def _rest_points(shift):
    if shift == '休':
        return 2
    return 1 if shift in ['休/', '/休', '/訪'] else 0

def _day_kind(d):
    # Returns 'weekday', 'saturday', or 'holiday' for column index d
    wd = weekday_list[d]
    if wd in ['Thursday', 'Sunday'] or is_japanese_holiday(dates[d]):
        return 'holiday'
    if wd == 'Saturday':
        return 'saturday'
    return 'weekday'

def _valid_day_configuration(col):
    """Validate that required shifts for the day type are still satisfied in column `col`.
    This is a *minimal* validator that checks presence of mandatory roles without over-constraining counts.
    - Weekday (non-holiday): either CT day -> {1,2,3,4,早,残,〇,CT} present,
                             or 2・CT day -> {1,3,4,早,残,〇,2・CT} present.
    - Saturday: must include {'早','残','〇'} and either {1/,3/,4/} (when 久保が2/) or {1/,2/,3/,4/}.
    - Holiday (Thu/Sun/祝): must include {'早日','残日'}.
    Returns True if constraints are met.
    """
    global df, date_cols
    d = date_cols.index(col)
    kind = _day_kind(d)
    vals = set(df[col].dropna().tolist())

    if kind == 'weekday':
        if 'CT' in vals:
            required = {'1', '2', '3', '4', '早', '残', '〇', 'CT'}
            return required.issubset(vals)
        # 2・CT 体制
        if '2・CT' in vals:
            required = {'1', '3', '4', '早', '残', '〇', '2・CT'}
            return required.issubset(vals)
        # 平日なのにどちらもない場合は不可
        return False

    if kind == 'saturday':
        base = {'早', '残', '〇'}
        has_base = base.issubset(vals)
        if not has_base:
            return False
        # 外来が 1/,3/,4/（久保が2/を取る構成）か、1/,2/,3/,4/ のいずれか
        alt1 = {'1/', '3/', '4/'}
        alt2 = {'1/', '2/', '3/', '4/'}
        return (alt1.issubset(vals) and not ('2/' in vals and not alt2.issubset(vals))) or alt2.issubset(vals)

    if kind == 'holiday':
        return {'早日', '残日'}.issubset(vals)

    return True

def _apply_swap_if_safe(n, col_from, col_to):
    """Try swapping df.at[n, col_from] (off) with df.at[n, col_to] (work) safely.
    - Keep fixed_mask untouched
    - Do not move 夜 or × relative to 夜勤翌日
    - Maintain day configuration validity for BOTH columns
    - Respect 御書 constraints (only 早/残/〇 or rest on working days)
    - Update current_rest_score if present
    Returns True if swap applied, else False.
    """
    global df, fixed_mask, current_rest_score

    if fixed_mask.at[n, col_from] or fixed_mask.at[n, col_to]:
        return False

    d_from = date_cols.index(col_from)
    d_to = date_cols.index(col_to)

    s_from = df.at[n, col_from]
    s_to = df.at[n, col_to]

    # Must be swapping OFF <- > WORK
    if not _is_off(s_from):
        return False
    if _is_off(s_to):
        return False

    # Do not touch night or post-night ×
    if s_to == '夜':
        return False
    if s_from == '×' or s_to == '×':
        return False
    if (d_to > 0 and df.at[n, date_cols[d_to - 1]] == '夜'):
        return False
    if (d_from > 0 and df.at[n, date_cols[d_from - 1]] == '夜'):
        return False

    # 御書の制約: 御書が勤務日に移るなら病棟シフト（早・残・〇）のみ可
    if n == '御書' and s_to not in ['早', '残', '〇']:
        return False

    # Tentatively apply in-memory
    old_from, old_to = s_from, s_to
    df.at[n, col_from], df.at[n, col_to] = s_to, s_from

    ok = _valid_day_configuration(col_from) and _valid_day_configuration(col_to)

    if not ok:
        # revert
        df.at[n, col_from], df.at[n, col_to] = old_from, old_to
        return False

    # Update rest score bookkeeping if present
    try:
        if isinstance(current_rest_score, dict):
            current_rest_score[n] = current_rest_score.get(n, 0) - _rest_points(old_from) + _rest_points(s_from)
    except NameError:
        pass

    return True


def prevent_four_day_rest_streaks():
    """
    各看護師の4日以上連続した休み（休、休/、/休）を検出し、
    **日別の必要シフト構成を壊さない**範囲で、
    固定されていない別日の出勤と安全にスワップして分散させる。
    - 夜勤・夜勤明け「×」には干渉しない
    - 御書は病棟（早・残・〇）以外の勤務に移さない
    - 交換後に当日/交換日の必須シフト構成が満たされているか検証
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
                    # 直近4日の範囲から、固定でない休みセルを1つ選ぶ
                    for j in range(i, i - 4, -1):
                        if j < 0:
                            continue
                        col_j = date_cols[j]
                        shift_j = df.at[n, col_j]
                        if fixed_mask.at[n, col_j] or shift_j not in off_codes:
                            continue

                        # 交換候補: 連続休み区間外の"出勤"日に限定
                        for k, col_k in enumerate(date_cols):
                            if i - 3 <= k <= i:
                                continue
                            shift_k = df.at[n, col_k]
                            if fixed_mask.at[n, col_k]:
                                continue
                            if _is_off(shift_k) or shift_k == '夜':
                                continue
                            # 夜勤翌日の×は触らない
                            if (k > 0 and df.at[n, date_cols[k - 1]] == '夜'):
                                continue

                            # 安全スワップ適用
                            if _apply_swap_if_safe(n, col_j, col_k):
                                swapped = True
                                break
                        if swapped:
                            break

                    # スワップできなければカウントをリセットして次へ
                    if not swapped:
                        streak = 0
                continue
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


def _swap_between_nurses_if_safe(col, nurse_off, nurse_work):
    """Swap shifts between two different nurses on the SAME day column `col` safely.
    - nurse_off must currently be OFF (休/休// /休) on `col`
    - nurse_work must currently be WORK (not off) on `col`
    - Keep fixed_mask untouched
    - Do not touch 夜 and post-night ×
    - Respect 御書 constraints: if 御書 moves to WORK, it must be one of ['早','残','〇']
    - Validate day configuration after swap with _valid_day_configuration(col)
    Returns True if swap applied, else False.
    """
    global df, fixed_mask, date_cols, current_rest_score

    if fixed_mask.at[nurse_off, col] or fixed_mask.at[nurse_work, col]:
        return False

    d = date_cols.index(col)
    s_off = df.at[nurse_off, col]
    s_work = df.at[nurse_work, col]

    # pre-conditions
    if s_off not in (FULL_OFF_SHIFTS + HALF_OFF_SHIFTS):
        return False
    if s_work in (FULL_OFF_SHIFTS + HALF_OFF_SHIFTS):
        return False
    if s_work == '夜':
        return False
    if s_work == '×' or s_off == '×':
        return False
    if (d > 0 and (df.at[nurse_off, date_cols[d-1]] == '夜' or df.at[nurse_work, date_cols[d-1]] == '夜')):
        return False

    # 御書制約
    if nurse_off == '御書' and s_work not in ['早','残','〇']:
        return False

    # tentative swap
    old_off, old_work = s_off, s_work
    df.at[nurse_off, col], df.at[nurse_work, col] = s_work, s_off

    if not _valid_day_configuration(col):
        # revert
        df.at[nurse_off, col], df.at[nurse_work, col] = old_off, old_work
        return False

    # update current_rest_score bookkeeping if available (2pt scale)
    try:
        if isinstance(current_rest_score, dict):
            current_rest_score[nurse_off] = current_rest_score.get(nurse_off, 0) - _rest_points(old_off) + _rest_points(s_off)
            current_rest_score[nurse_work] = current_rest_score.get(nurse_work, 0) - _rest_points(old_work) + _rest_points(old_off)
    except NameError:
        pass

    return True

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
    df.fillna('', inplace=True)
    balance_rest_days()
    prevent_seven_day_streaks()
    prevent_four_day_rest_streaks()
    ensure_min_rest_days_balanced()
    export_outputs(df, date_cols, nurse_names)

if __name__ == "__main__":
    main()