import os
import argparse
import re
import numpy as np
import glob
import pandas as pd
from create_table3d import render_table3d
from results_helper import (
    extract_model_name_from_filename, extract_prompt_type_from_filename, extract_step_from_filename,
    ensure_directory, get_metric_label, get_model_label, get_paper_models)


def extract_table(file_path):
    with open(file_path, "r") as file:
        content = file.read()
        # Regular expression to find the first <table>...</table>
        match = re.search(r"<table.*?>.*?</table>", content, re.DOTALL)
        if match:
            return match.group(0)
        else:
            return ""

def create_combined_table(table1_path, table2_path, output_file_path):
    table1 = extract_table(table1_path)
    table2 = extract_table(table2_path)
    combined_html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Combined Tables</title>
</head>
<body>
    {table1}
    <p style="margin:24pt"/>
    {table2}
</body>
</html>
"""
    with open(output_file_path, "w") as file:
        file.write(combined_html)


def generate_summary_table(analysis_df, output_file_path, prompt_type_order, metric_order):

    df = pd.melt(analysis_df, id_vars=["group", "prompt_type"],
                 value_vars=metric_order,
                 var_name="metric", value_name="value")

    value_vars_labels = [get_metric_label(metric) for metric in metric_order]

    df["subgroup"] = pd.Categorical(df["metric"].apply(get_metric_label),
                                    categories=value_vars_labels, ordered=True)
    df["category"] = pd.Categorical(df["prompt_type"],
                                    categories=prompt_type_order, ordered=True)

    render_table3d(df, output_file_path)


def find_model_without_other_model_in_previous_step(analysis_df):
    # the only model with no other model in the previous step is the best model from the previous step
    models_with_other_model_in_previous_step = analysis_df.loc[analysis_df["model"] != analysis_df["model_prev_step"], "model"].unique()
    all_models = analysis_df["model"].unique()
    model_without_other_model_in_previous_step = np.setdiff1d(all_models, models_with_other_model_in_previous_step)

    if len(model_without_other_model_in_previous_step) == 1:
        return model_without_other_model_in_previous_step[0]
    elif len(model_without_other_model_in_previous_step) == 0:
        raise ValueError("No entry in 'model' where previous step .")
    else:
        raise ValueError("Multiple entries in 'model' with no other model in previous step.")

def model_category_order(models_and_suffixes, models):
    model_labels = [get_model_label(model) for model in models]

    all_items = set(models_and_suffixes)
    category_order = []
    for base in model_labels:
        category_order.append(base)
        category_order.extend([item for item in all_items if item.startswith(base + " ")])

    return category_order

def create_tables_for_analysis_file(analysis_file, output_path, models):
    step = extract_step_from_filename(analysis_file)
    analysis_df = pd.read_csv(analysis_file)
    analysis_df["model"] = analysis_df["results_file"].apply(lambda filename: extract_model_name_from_filename(filename, step))

    if models:
        if models[0] == "PAPER_MODELS":
            models = get_paper_models(step)
        analysis_df = analysis_df[analysis_df["model"].isin(models)]

    analysis_df["group"] = analysis_df["model"].apply(get_model_label)

    if step in [2,3]:
        # group is the model + the info whether the previous steps were using the same model or other models
        analysis_df["model_prev_step"] = analysis_df["results_file"].apply(lambda filename: extract_model_name_from_filename(filename, step-1))

        # for subsequent steps there are two variants: one with the same model and one with the best model before
        different_model_prev_step = analysis_df["model_prev_step"] != analysis_df["model"]
        if step == 2:
            additional_info = f" - using results from best model in Step 1"
        elif step == 3:
            additional_info = f" - using results from best models in previous steps"
        analysis_df.loc[different_model_prev_step, "group"] += additional_info
        analysis_df.loc[analysis_df["model"] == find_model_without_other_model_in_previous_step(analysis_df), "group"] += (
            f" (the best model in step {step-1})"
        )

    create_tables_for_analysis_step(step, analysis_df, output_path, models)


def create_tables_for_analysis_step(step, analysis_df, output_path, models):

    analysis_df["prompt_type"] = analysis_df['results_file'].apply(extract_prompt_type_from_filename)
    output_file_path = os.path.join(output_path, f"step{step}_summary_table.html")
    reduced_output_file_path = os.path.join(output_path, f"step{step}_summary_table_reduced.html")

    if models:
        analysis_df["group"] = pd.Categorical(analysis_df["group"],
                                            categories=model_category_order(analysis_df["group"], models),
                                            ordered=True)

    if step == 1:
        generate_summary_table(analysis_df, output_file_path,
                               prompt_type_order = ["0",
                                                    "2-gyn", "2-uro",
                                                    "4-gyn", "4-uro",
                                                    "6-gyn", "6-uro"],
                               metric_order=["p_correct_total", "p_na", "recall", "specificity"])

    elif step == 2:
        generate_summary_table(analysis_df, output_file_path,
                               prompt_type_order = ["0-uro", "0-gyn",
                                                    "0-uro-ctx", "0-gyn-ctx",
                                                    "2-uro", "2-gyn",
                                                    "2-uro-ctx", "2-gyn-ctx"],
                               metric_order=["p_all_diagnoses_found_in_snippets_with_diagnoses",
                                             "p_no_other_diagnoses_found_for_snippet",
                                             "p_snippets_correct",
                                             "p_na",
                                             "p_C77_79"])

        generate_summary_table(analysis_df, reduced_output_file_path,
                               prompt_type_order = ["0-uro", "0-gyn",
                                                    "0-uro-ctx", "0-gyn-ctx",
                                                    "2-uro", "2-gyn",
                                                    "2-uro-ctx", "2-gyn-ctx"],
                               metric_order=["p_snippets_correct"])

    elif step == 3:
        output_file_path_llms = os.path.join(output_path, f"step{step}_summary_table_llms.html")
        reduced_output_file_path_llms = os.path.join(output_path, f"step{step}_summary_table_llms_reduced.html")

        prompt_type_order = ["0", "0-dates", "3", "3-dates",
                             "0-verify", "4-verify"]
        analysis_df_llms = analysis_df[analysis_df["prompt_type"].isin(prompt_type_order)]
        generate_summary_table(analysis_df_llms,
                               output_file_path_llms,
                               prompt_type_order=prompt_type_order,
                               metric_order=["p_correct", "p_correct_total", "p_na"])
        generate_summary_table(analysis_df_llms,
                               reduced_output_file_path_llms,
                               prompt_type_order=prompt_type_order,
                               metric_order=["p_correct_total"])

        output_file_path_heuristics = os.path.join(output_path, f"step{step}_summary_table_heuristics.html")
        prompt_type_order = ["regex-match-same-line", "regex-match-line-dist-1", "regex-match-line-dist-2"]
        generate_summary_table(analysis_df[analysis_df["prompt_type"].isin(prompt_type_order)],
                               output_file_path_heuristics,
                               prompt_type_order=prompt_type_order,
                               metric_order=["p_correct", "p_correct_total"])

        create_combined_table(table1_path=output_file_path_llms,
                              table2_path=output_file_path_heuristics,
                              output_file_path=output_file_path)

    else:
        raise RuntimeError(f"Step {step} not implemented")


def create_tables_for_dir(input_dir, output_dir, models):
    analysis_files = glob.glob(os.path.join(input_dir, "analysis_step*.csv"))
    for analysis_file in analysis_files:
        create_tables_for_analysis_file(analysis_file, output_dir, models)


def create_summary_tables(input, output_dir, models=[]):
    ensure_directory(output_dir)

    if os.path.isfile(input):
        create_tables_for_analysis_file(analysis_file=input, output_path=output_dir, models=models)
    else: # it must be a directory
        create_tables_for_dir(input_dir=input, output_dir=output_dir, models=models)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation with specified step and model.")

    parser.add_argument("input", type=str, help="File or directory with analysis results used as input")
    parser.add_argument("--output-dir", type=str, help="Output path for tables", default=None)
    parser.add_argument("--models", nargs='+', default=[], help="Names of models to display in the plot.")

    args = parser.parse_args()
    if not args.output_dir:
        args.output_dir = os.path.join(args.input, "summary")

    create_summary_tables(input=args.input, output_dir=args.output_dir, models=args.models)
