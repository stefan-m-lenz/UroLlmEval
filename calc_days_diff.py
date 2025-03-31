import os
import glob
import re
import pandas as pd
from datetime import datetime

from analyze_step3_results import normalize_date_answer
from results_helper import extract_model_name_from_filename, extract_prompt_type_from_filename, extract_step_from_filename

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

    # normalise dates
    df["model_norm"] = df["model_answer"].apply(
        lambda x: normalize_date_answer(str(x)) if pd.notna(x) else None
    )
    df["expected_norm"] = df["expected_answer"].apply(
        lambda x: normalize_date_answer(str(x)) if pd.notna(x) else None
    )

    # calc differnece in days and months
    def compute_diff(row):
        if row["expected_norm"] == "0":
            return pd.Series({"days_diff": None, "months_diff": None})
        model_date = parse_date(row["model_norm"])
        expected_date = parse_date(row["expected_norm"])
        if model_date is None or expected_date is None:
            return pd.Series({"days_diff": None, "months_diff": None})
        delta = abs(model_date - expected_date)
        days_diff = delta.days
        # Grobe Umrechnung in Monate (30 Tage pro Monat)
        months_diff = days_diff // 30
        return pd.Series({"days_diff": days_diff, "months_diff": months_diff})

    df = df.join(df.apply(compute_diff, axis=1))

    if output_csv:
        df[["expected_answer", "model_answer", "model_norm", "expected_norm", "days_diff", "months_diff"]].to_csv(output_csv, index=False)

    return df

def process_csv_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    csv_files = glob.glob(os.path.join(input_folder, "*.csv"))
    for csv_file in csv_files:
        base_name = os.path.basename(csv_file)
        if not (base_name.startswith("step3") and "step1" in base_name):
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
        try:
            model_name = extract_model_name_from_filename(base_name, 1)
        except Exception as e:
            model_name = None
        try:
            prompt_type = extract_prompt_type_from_filename(base_name)
        except Exception as e:
            prompt_type = None

        days_series = pd.to_numeric(df["days_diff"], errors="coerce").dropna()
        months_series = pd.to_numeric(df["months_diff"], errors="coerce").dropna()

        days_stats = days_series.describe() if not days_series.empty else {}
        months_stats = months_series.describe() if not months_series.empty else {}

        summary_list.append({
            "model": model_name,
            "prompt type": prompt_type,
            "diff_answers_count": days_stats.get("count", 0),
            "days_mean": days_stats.get("mean", None),
            "days_min": days_stats.get("min", None),
            "days_max": days_stats.get("max", None),
            "months_mean": months_stats.get("mean", None),
            "months_min": months_stats.get("min", None),
            "months_max": months_stats.get("max", None),
        })

    summary_df = pd.DataFrame(summary_list)
    summary_df.sort_values(by=["model", "prompt type"], inplace=True)
    summary_df.to_csv(summary_csv, index=False)

if __name__ == "__main__":
    input_folder = "./output"
    processed_folder = "./processed_csvs"

    process_csv_folder(input_folder, processed_folder)

    summary_csv = os.path.join(processed_folder, "summary_results.csv")
    summarize_results(processed_folder, summary_csv)
