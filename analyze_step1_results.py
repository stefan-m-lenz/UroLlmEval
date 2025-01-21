import pandas as pd
import json
import argparse
import numpy as np
import os
import glob
from results_helper import calculate_metrics_for_bool_answer, parse_json_array, output_analysis_result, count_dict


def json_array_answer_to_bool(str):
    try:
        json_array = parse_json_array(str)
        return len(json_array) > 0
    except json.JSONDecodeError:
        return np.nan


def run_analysis(file_path):
    answers_df = pd.read_csv(file_path)
    answers_df["model_answer_bool"] = answers_df.apply(lambda row: json_array_answer_to_bool(row['model_answer']), axis=1)

    result = {"results_file" : os.path.splitext(os.path.basename(file_path))[0],
              **calculate_metrics_for_bool_answer(answers_df["expected_answer"], answers_df["model_answer_bool"]),
              "na_values": count_dict(answers_df[answers_df['model_answer_bool'].isna()]['model_answer'])}
    if "query_execution_time" in answers_df.columns:
        result = {**result,
                  "mean_query_execution_time": answers_df["query_execution_time"].mean()}
    return result


def run_analyses_on_dir(dir_path):
    step1files = glob.glob(os.path.join(dir_path, "step1_*.csv"))
    results = [run_analysis(file) for file in step1files]
    return results


def analyze_step1_results(results_path, output):
    if os.path.isdir(results_path):
        result = run_analyses_on_dir(results_path)
    else:
        result = run_analysis(results_path)
    output_analysis_result(result, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze results from evaluation step 1")
    parser.add_argument("path", help="The path to a file containing the results of step 1")
    parser.add_argument("--output", help="Output file. May end with .json or .csv", default="stdout")

    args = parser.parse_args()

    analyze_step1_results(results_path=args.path, output=args.output)