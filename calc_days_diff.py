import os
import glob
import re
import pandas as pd
from datetime import datetime

from analyze_step3_results import normalize_date_answer
from results_helper import extract_model_name_from_filename, extract_prompt_type_from_filename

def parse_date(date_str):
    if not date_str or date_str == "0":
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    return None

def compare_csv_dates(csv_file, output_csv=None):
    df = pd.read_csv(csv_file)

    # normalize dates
    df["model_norm"] = df["model_answer"].apply(
        lambda x: normalize_date_answer(str(x)) if pd.notna(x) else None
    )
    df["expected_norm"] = df["expected_answer"].apply(
        lambda x: normalize_date_answer(str(x)) if pd.notna(x) else None
    )

    # calculate differences in days and months
    def compute_diff(row):
        m_norm = row.get("model_norm")
        e_norm = row.get("expected_norm")
        # skip empty or zero answers
        if not e_norm or e_norm == "0" or not m_norm:
            return pd.Series({"days_diff": None, "months_diff": None})
        # year-only case: treat each year as 365 days and 12 months
        if re.fullmatch(r"\d{4}", str(m_norm)) and re.fullmatch(r"\d{4}", str(e_norm)):
            year_diff = abs(int(m_norm) - int(e_norm))
            days_diff = year_diff * 365
            months_diff = year_diff * 12
            return pd.Series({"days_diff": days_diff, "months_diff": months_diff})
        # parse full or year-month dates
        model_date = parse_date(m_norm)
        expected_date = parse_date(e_norm)
        if model_date is None or expected_date is None:
            return pd.Series({"days_diff": None, "months_diff": None})
        delta = abs(model_date - expected_date)
        days_diff = delta.days
        months_diff = days_diff // 30
        return pd.Series({"days_diff": days_diff, "months_diff": months_diff})

    df = df.join(df.apply(compute_diff, axis=1))

    def model_in_regex(row):
        if pd.isna(row.get("model_norm")) or pd.isna(row.get("regex_matched_dates")):
            return False
        try:
            dates = eval(row["regex_matched_dates"])
            normalized_dates = set(
                normalize_date_answer(str(d)) for d in dates if d and d != "0"
            )
            return row["model_norm"] in normalized_dates
        except Exception:
            return False

    df["model_answer_in_regex_matches"] = df.apply(model_in_regex, axis=1)

    if output_csv:
        df[[
            "expected_answer", "model_answer", "model_norm", "expected_norm",
            "days_diff", "months_diff", "model_answer_in_regex_matches"
        ]].to_csv(output_csv, index=False)

    return df

def process_csv_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    csv_files = glob.glob(os.path.join(input_folder, "*.csv"))
    for csv_file in csv_files:
        base_name = os.path.basename(csv_file)
        if not (base_name.startswith("step3")):
            continue
        file_name, ext = os.path.splitext(base_name)
        output_file = os.path.join(output_folder, file_name + "_days_diff.csv")
        compare_csv_dates(csv_file, output_file)

def summarize_results(processed_folder, summary_csv):
    summary_list = []
    processed_files = glob.glob(os.path.join(processed_folder, "*_days_diff.csv"))

    for csv_file in processed_files:
        df = pd.read_csv(csv_file)
        base_name = os.path.basename(csv_file)

        # Modell- und Prompt-Typ extrahieren
        try:
            model_name = extract_model_name_from_filename(base_name, step=3)
            prev_model_name = extract_model_name_from_filename(base_name, step=2)
        except Exception:
            model_name = None
        try:
            prompt_type = extract_prompt_type_from_filename(base_name)
        except Exception:
            prompt_type = None

        num_questions = len(df)

        # 1) NaN-Antworten (kein days_diff berechnet)
        num_na_answers = df["days_diff"].isna().sum()

        # 2) Exakt gleiche Antworten (days_diff == 0)
        num_equal_answers = df["days_diff"].apply(
            lambda x: pd.notna(x) and x == 0
        ).sum()

        # 3) Unterschiedliche Antworten (days_diff > 0)
        num_diff_answers = df["days_diff"].apply(
            lambda x: pd.notna(x) and x > 0
        ).sum()

        # 4) Unterschiedliche Antworten, die nicht per Regex validiert wurden
        num_diff_not_in_regex = df.apply(
            lambda row: pd.notna(row["days_diff"])
                        and row["days_diff"] > 0
                        and not row.get("model_answer_in_regex_matches", False),
            axis=1
        ).sum()

        # Prozentwerte (optional)
        pct_na    = num_na_answers    / num_questions * 100 if num_questions else 0
        pct_equal = num_equal_answers / num_questions * 100 if num_questions else 0
        pct_diff  = num_diff_answers  / num_questions * 100 if num_questions else 0

        # Für detaillierte Statistiken nur die unterschiedlichen Antworten betrachten
        diff_df = df[df["days_diff"].apply(lambda x: pd.notna(x) and x > 0)]

        days_series = pd.to_numeric(diff_df["days_diff"], errors="coerce").dropna()
        months_series = pd.to_numeric(diff_df["months_diff"], errors="coerce").dropna()

        days_stats = days_series.describe() if not days_series.empty else {}
        months_stats = months_series.describe() if not months_series.empty else {}

        # Stats für die falsch-Regex-geprüften
        wrong_not_in_regex_df = diff_df[~diff_df["model_answer_in_regex_matches"]]
        days_stats_wrong_not_in_regex = (
            pd.to_numeric(wrong_not_in_regex_df["days_diff"], errors="coerce")
              .describe()
            if not wrong_not_in_regex_df.empty else {}
        )
        months_stats_wrong_not_in_regex = (
            pd.to_numeric(wrong_not_in_regex_df["months_diff"], errors="coerce")
              .describe()
            if not wrong_not_in_regex_df.empty else {}
        )

        # Zusammenfassung für dieses Modell/prompt
        summary_list.append({
            "model": model_name,
            "prompt type": prompt_type,
            "num_questions": num_questions,
            "prev_model": prev_model_name,

            "num_na_answers": num_na_answers,
            "pct_na": pct_na,

            "num_equal_answers": num_equal_answers,
            "pct_equal": pct_equal,

            "num_diff_answers": num_diff_answers,
            "pct_diff": pct_diff,

            "num_diff_not_in_regex": num_diff_not_in_regex,

            "days_mean": days_stats.get("mean", None),
            "days_min": days_stats.get("min", None),
            "days_max": days_stats.get("max", None),

            "months_mean": months_stats.get("mean", None),
            "months_min": months_stats.get("min", None),
            "months_max": months_stats.get("max", None),

            "days_mean_wrong_not_in_regex": days_stats_wrong_not_in_regex.get("mean", None),
            "days_min_wrong_not_in_regex": days_stats_wrong_not_in_regex.get("min", None),
            "days_max_wrong_not_in_regex": days_stats_wrong_not_in_regex.get("max", None),

            "months_mean_wrong_not_in_regex": months_stats_wrong_not_in_regex.get("mean", None),
            "months_min_wrong_not_in_regex": months_stats_wrong_not_in_regex.get("min", None),
            "months_max_wrong_not_in_regex": months_stats_wrong_not_in_regex.get("max", None),
        })

    # DataFrame erstellen, sortieren und als CSV speichern
    summary_df = pd.DataFrame(summary_list)
    summary_df.sort_values(by=["model", "prompt type"], inplace=True)
    summary_df.to_csv(summary_csv, index=False)

if __name__ == "__main__":
    input_folder = "./output"
    processed_folder = "./processed_csvs"

    process_csv_folder(input_folder, processed_folder)

    summary_csv = os.path.join(processed_folder, "summary_results.csv")
    summarize_results(processed_folder, summary_csv)