import pandas as pd
import argparse
import numpy as np
import ast
import os
import glob
from results_helper import normalize_icd10_answers, count_correct, output_analysis_result, count_dict

def normalized_icd_answer_to_bool(row):
    if row.model_answer_normalized:
        return row.model_answer_normalized in row.expected_answer
    else:
        return np.nan


def combine_model_answers_for_snippet(df):
    """
    Combines `model_answer_normalized` values per `snippet_id` into lists,
    ignoring `None` values, and returns a new DataFrame.

    Args:
    - df (pd.DataFrame): Input DataFrame containing columns "snippet_id", "expected_answer", "model_answer_normalized".

    Returns:
    - pd.DataFrame: A DataFrame with columns "snippet_id", "expected_answer", "model_answers_combined".
    """
    # Dictionary to hold the combined results
    combined_dict = {}

    # Iterate over unique snippet_ids
    for snippet_id in df["snippet_id"].unique():
        # Filter rows for the current snippet_id
        snippet_rows = df[df["snippet_id"] == snippet_id]

        # Extract expected_answer (it's the same for all rows with the same snippet_id)
        expected_answer = snippet_rows["expected_answer"].iloc[0]

        # Extract non-None model_answer_normalized values
        model_answers_combined = set(
            answer for answer in snippet_rows["model_answer_normalized"] if pd.notna(answer)
        )

        # Add the results to the dictionary
        combined_dict[snippet_id] = {
            "snippet_id": snippet_id,
            "expected_answer": set(ast.literal_eval(expected_answer)),
            "model_answers_combined": model_answers_combined
        }

    # Convert the dictionary to a DataFrame
    combined_df = pd.DataFrame(combined_dict.values())

    return combined_df.reset_index(drop=True)


SECONDARY_NEOPLASMS = {"C77", "C78", "C79"}

def run_analysis(file_path):
    results_df = pd.read_csv(file_path)

    results_df = normalize_icd10_answers(results_df)

    results_df["model_answer_bool"] = results_df.apply(normalized_icd_answer_to_bool, axis=1)
    results_df["model_answer_bool"] = results_df["model_answer_bool"].astype(float)
    results_df.loc[results_df["model_answer_normalized"].isin(SECONDARY_NEOPLASMS), "model_answer_bool"] = np.nan

    # interesting: percent all diagnosis correctly identified per snippet

    ps = count_correct(results_df["model_answer_bool"])
    ret = {"results_file": os.path.splitext(os.path.basename(file_path))[0]}
    # we don't know exactly whether the mapping is correct as we only have the data per snippet
    ret["p_potentially_correct_diagnoses"] = ps["p_correct"]
    ret["p_na"] = ps["p_na"]
    ret["p_C77_79"] = count_correct(results_df["model_answer_normalized"].isin(SECONDARY_NEOPLASMS))["p_correct"]

    results_df_snippets_combined = combine_model_answers_for_snippet(results_df)
    results_df_snippets_combined["all_diagnoses_found"] = results_df_snippets_combined.apply(
        lambda row: row.expected_answer <= row.model_answers_combined, axis=1)

    rows_with_diagnoses = results_df_snippets_combined["expected_answer"].apply(bool)
    all_diagnoses_found_in_snippets_with_diagnoses = results_df_snippets_combined["all_diagnoses_found"][rows_with_diagnoses]
    ret["p_all_diagnoses_found_in_snippets_with_diagnoses"] = count_correct(all_diagnoses_found_in_snippets_with_diagnoses)["p_correct"]

    results_df_snippets_combined["not_more_diagnoses_found"] = results_df_snippets_combined.apply(
        lambda row: row.model_answers_combined <= ( row.expected_answer | SECONDARY_NEOPLASMS ), axis=1)

    ret["p_no_other_diagnoses_found_for_snippet"] = count_correct(results_df_snippets_combined["not_more_diagnoses_found"])["p_correct"]

    ret["p_snippets_correct"] = count_correct(
        results_df_snippets_combined["all_diagnoses_found"] & results_df_snippets_combined["not_more_diagnoses_found"])["p_correct"]

    other_diagnoses = results_df_snippets_combined.apply(lambda row: row.model_answers_combined - row.expected_answer, axis=1)
    ret["other_diagnoses"] = count_dict(diagnosis for diagnoses in other_diagnoses for diagnosis in diagnoses)

    ret["na_values"] = count_dict(results_df[results_df['model_answer_bool'].isna()]['model_answer'].tolist())

    return ret


def run_analyses_on_dir(dir_path):
    step2files = glob.glob(os.path.join(dir_path, "step2_*.csv"))
    results = [run_analysis(file) for file in step2files]
    return results


def analyze_step2_results(results_path, output):
    if os.path.isdir(results_path):
        result = run_analyses_on_dir(results_path)
    else:
        result = run_analysis(results_path)
    output_analysis_result(result, output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze results from evaluation step 2")
    parser.add_argument("path", help="The path to a file containing the results of step 2")
    parser.add_argument("--output", help="Output file. May end with .json or .csv", default="stdout")

    args = parser.parse_args()

    analyze_step2_results(results_path=args.path, output=args.output)
