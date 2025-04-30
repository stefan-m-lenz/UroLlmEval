import os
import glob
import pandas as pd
import ast
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


def main(input_folder="./output"):
    csv_files = glob.glob(os.path.join(input_folder, "step3*.csv"))
    summary = [load_and_compute_failure_rate(file) for file in csv_files]
    df_summary = pd.DataFrame(summary)
    df_summary.to_csv("output/summary/date_diffs.csv", index=False)

    # Compute median percentage per prompt type
    statistics = (
        df_summary
        .groupby("prompt_type")["pct_diff_not_in_regex"]
        .agg(
            median_pct_diff_not_in_regex="median",
            max_pct_diff_not_in_regex="max",
        )
        .reset_index()
        .rename(columns={"pct_diff_not_in_regex": "median_pct_diff_not_in_regex"})
    )

    print("\nFor each prompt type: median % of wrong dates that are not in the dates extracted by regex:")
    print(statistics.to_string(index=False))

if __name__ == "__main__":
    main()
