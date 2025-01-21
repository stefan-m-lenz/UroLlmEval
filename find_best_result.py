import pandas as pd
import os
import argparse
from results_helper import extract_step_from_filename, extract_model_name_from_filename, ensure_directory

RELEVANT_COLUMN = {
    1: "p_correct_total",
    2: "p_snippets_correct",
    3: "p_correct_total"
}

# Finds the best model/prompt combination.
# If a dictionary that maps steps to model names is specified in the models argument,
# it will only search for model/prompt combinations that use this model in the respective steps.
def find_best_result_in_df(analysis_df, models=None):

    # assume that all entries have the same step, use step from first row
    step = extract_step_from_filename(analysis_df.iloc[0]["results_file"])

    if models: # use only results from the specified models in the steps
        # go through all previous steps and find out whether the model has been used for the step in the models dict
        uses_models = pd.Series(True, index=analysis_df.index)

        for step_i in models.keys():
            model = models[step_i]
            model = model.replace("/", "_")
            model = model.replace(":", "_")
            model_in_step = analysis_df["results_file"].apply(lambda filename: extract_model_name_from_filename(filename=filename,
                                                                                                                step=step_i))
            uses_models = uses_models & (model_in_step == model)

        # filter analysis df and get only rows where the models have been used
        analysis_df = analysis_df.loc[uses_models]

    best_row_idx = analysis_df[RELEVANT_COLUMN[step]].idxmax()
    best_result = analysis_df.loc[best_row_idx]["results_file"]

    return best_result


def find_best_result(analysis_file, models=None):
    analysis_df = pd.read_csv(analysis_file)
    return find_best_result_in_df(analysis_df=analysis_df, models=models)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation with specified step and model.")
    parser.add_argument("analysis_file", type=argparse.FileType("r"))

    args = parser.parse_args()

    print(find_best_result(analysis_file=args.analysis_file))
