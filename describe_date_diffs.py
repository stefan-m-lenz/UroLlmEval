import os
import glob
import pandas as pd
import ast
from datetime import datetime
from analyze_step3_results import normalize_date_answer
from results_helper import extract_model_name_from_filename, extract_prompt_type_from_filename


def load_and_compute_failure_rate(csv_file):
    df = pd.read_csv(csv_file)

    df["model_norm"] = df["model_answer"].apply(
        lambda x: normalize_date_answer(str(x)) if pd.notna(x) else None
    )
    df["expected_norm"] = df["expected_answer"].apply(
        lambda x: normalize_date_answer(str(x)) if pd.notna(x) else None
    )

    def model_in_regex(row):
        model_norm = row.get("model_norm")
        raw_matches = row.get("regex_matched_dates")

        if pd.isna(model_norm) or pd.isna(raw_matches) or model_norm in ("0", ""):
            return False

        try:
            matched_dates = ast.literal_eval(raw_matches)  # safer than eval
            normalized_matches = {
                normalize_date_answer(str(d)) for d in matched_dates
                if d and normalize_date_answer(str(d)) is not None
            }
            return model_norm in normalized_matches
        except Exception as e:
            print(f"Failed to parse regex match list: {raw_matches} | Error: {e}")
            return False

    # Use only rows where there is a date answer and the model answered with a valid date
    valid_rows = df[
        (df["expected_answer"] != "0") &
        (df["model_answer"] != "Nein") &
        (df["expected_answer"].notna()) &
        (df["model_answer"].notna())
    ].copy()

        # if extract_model_name_from_filename(csv_file, step=3) == "levenshtein-regex" and extract_model_name_from_filename(csv_file, step=2) == "meta-llama_Meta-Llama-3.1-8B-Instruct":
    #     print("hallo")
    #     x = valid_rows[["expected_answer", "model_answer", "model_answer_in_regex_matches"]]

    valid_rows["model_answer_in_regex_matches"] = valid_rows.apply(model_in_regex, axis=1)

    if valid_rows["expected_norm"].isna().any():
        raise ValueError(
            f"NaN found in expected_norm or model_norm after normalization "
            f"in file {os.path.basename(csv_file)}. This indicates a logic or format issue."
        )

    # Count failures: model answer is different and not in regex
    num_diff_not_in_regex = valid_rows[
        (valid_rows["model_norm"].notna()) &
        (valid_rows["expected_norm"] != valid_rows["model_norm"]) &
        (~valid_rows["model_answer_in_regex_matches"])
    ].shape[0]

    total_valid = len(valid_rows)
    pct_failures = (num_diff_not_in_regex / total_valid * 100) if total_valid else 0.0

    return {
        "file": os.path.basename(csv_file),
        "model": extract_model_name_from_filename(csv_file, step=3),
        "prompt_type": extract_prompt_type_from_filename(csv_file),
        "total_answered": total_valid,
        "diff_not_in_regex": num_diff_not_in_regex,
        "pct_diff_not_in_regex": pct_failures
    }


def parse_with_assumed_middle(date_str):
    """
    Parse a normalized date string ('YYYY', 'YYYY-MM', or 'YYYY-MM-DD')
    and return a datetime object, using the middle of the year/month as needed.
    """
    if not date_str or date_str == "0":
        return None
    try:
        if len(date_str) == 4:
            # Only year
            return datetime.strptime(date_str, "%Y").replace(month=7, day=1)
        elif len(date_str) == 7:
            # Year + month
            return datetime.strptime(date_str, "%Y-%m").replace(day=15)
        elif len(date_str) == 10:
            # Full date
            return datetime.strptime(date_str, "%Y-%m-%d")
    except Exception:
        return None
def load_and_compute_date_diffs(csv_file):
    df = pd.read_csv(csv_file)

    df = df[
        (df["expected_answer"] != "0") &
        (df["model_answer"] != "Nein") &
        (df["expected_answer"].notna()) &
        (df["model_answer"].notna())
    ].copy()

    df["model_norm"] = df["model_answer"].apply(lambda x: normalize_date_answer(str(x)))
    df["expected_norm"] = df["expected_answer"].apply(lambda x: normalize_date_answer(str(x)))

    df["model_parsed"] = df["model_norm"].apply(parse_with_assumed_middle)
    df["expected_parsed"] = df["expected_norm"].apply(parse_with_assumed_middle)

    diff_rows = df[
        (df["model_parsed"].notna()) &
        (df["expected_parsed"].notna()) &
        (df["model_norm"] != df["expected_norm"])
    ].copy()

    if diff_rows.empty:
        return pd.DataFrame()

    diff_rows["days_diff"] = (diff_rows["model_parsed"] - diff_rows["expected_parsed"]).abs().dt.days
    diff_rows["file"] = os.path.basename(csv_file)
    diff_rows["model"] = extract_model_name_from_filename(csv_file, step=3)
    diff_rows["prompt_type"] = extract_prompt_type_from_filename(csv_file)

    return diff_rows[[
        "file", "model", "prompt_type", "expected_answer", "model_answer",
        "expected_norm", "model_norm", "days_diff"
    ]]


def get_median_pct_date_not_in_regex(input_folder):
    csv_files = glob.glob(os.path.join(input_folder, "step3*.csv"))
    summary = [load_and_compute_failure_rate(file) for file in csv_files]
    df_summary = pd.DataFrame(summary)
    df_summary.to_csv("output/summary/date_diffs.csv", index=False)

    # Compute median percentage per prompt type
    statistics = (
        df_summary
        .groupby("prompt_type")["pct_diff_not_in_regex"]
        .agg(
            q25_pct_diff_not_in_regex=lambda x: x.quantile(0.25),
            median_pct_diff_not_in_regex="median",
            q75_pct_diff_not_in_regex=lambda x: x.quantile(0.75),
            max_pct_diff_not_in_regex="max",
        )
        .reset_index()
        .rename(columns={"pct_diff_not_in_regex": "median_pct_diff_not_in_regex"})
    )

    print("\nFor each prompt type: median % of wrong dates that are not in the dates extracted by regex:")
    print(statistics.to_string(index=False))

    return statistics


def main():
    input_folder = "output"
    #get_median_pct_date_not_in_regex(input_folder)

    csv_files = glob.glob(os.path.join(input_folder, "step3*.csv"))
    csv_files = [file for file in csv_files if "levenshtein-regex" not in file]
    all_diffs = [load_and_compute_date_diffs(file) for file in csv_files]
    df_diffs = pd.concat(all_diffs, ignore_index=True)

    # Now compute overall quantiles directly
    q25 = df_diffs["days_diff"].quantile(0.25)
    q50 = df_diffs["days_diff"].quantile(0.50)  # median
    q75 = df_diffs["days_diff"].quantile(0.75)

    print("\nQuantiles of date differences across all mismatches:")
    print(f"  25th percentile: {q25:.2f} days")
    print(f"  50th percentile (median): {q50:.2f} days")
    print(f"  75th percentile: {q75:.2f} days")


if __name__ == "__main__":
    main()
