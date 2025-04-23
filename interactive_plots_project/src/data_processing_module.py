import re
import pandas as pd
import numpy as np


def extract_step_from_filename(filename):
    step = re.search(pattern="step([1-3])", string=filename)
    if not step:
        raise ValueError(f"Step could not be determined from filename {filename}")
    return int(step.group(1))

# given a filename and a step number. extract the model name
def extract_model_name_from_filename(filename, step):
    if step == 1:
        pattern=f"step{step}_(.*?)(__.*)?(.csv|$)"
    else:
        pattern=f"step{step}_(.*?)(__.*)?(step{step-1}|.csv|$)"

    match = re.search(pattern=pattern, string=filename)

    if match:
        possible_model = match.group(1)
        if possible_model.startswith("_"):
            # this means, there is no model name here, only for a previous step
            # capture file name for step + 1
            if step < 3:
                return extract_model_name_from_filename(filename, step + 1)
            else:
                raise RuntimeError(f"Model name could not be identified from filename '{filename}'")
        else:
            return possible_model
    else:
        raise ValueError(f"Filename '{filename}' does not contain model info about step{step}")


def extract_prompt_type_from_filename(filename):
    pattern = r"__([^._]*)" # the prompt type comes after the first occurence of two underscores
    match = re.search(pattern=pattern, string=filename)
    if match:
        return match.group(1)
    else:
        raise ValueError(f"Filename '{filename}' does not contain model info about the prompt type")

METRIC_LABELS = {
      "recall": "Sensitivity",
      "specificity": "Specificity",
      "p_correct_total": "Accuracy (NA=wrong)",
      "p_correct": "Accuracy (excl. NA)",
      "p_na": "NA",
      "p_C77_79": "C77-C79 answers",
      "p_all_diagnoses_found_in_snippets_with_diagnoses": "All diagnoses found",
      "p_no_other_diagnoses_found_for_snippet": "No incorrect diagnosis",
      "p_snippets_correct": "Snippet correct"
    }

MODEL_LABELS = {
      "meta-llama_Llama-3.2-1B-Instruct": "LLama 3.2 1B",
      "utter-project_EuroLLM-1.7B-Instruct": "EuroLLM 1.7B",
      "meta-llama_Llama-3.2-3B-Instruct": "LLama 3.2 3B",
      "LeoLM_leo-hessianai-7b-chat": "LeoLM 7B Chat",
      "BioMistral_BioMistral-7B": "BioMistral-7B",
      "mistralai_Mistral-7B-Instruct-v0.3": "Mistral 7B v0.3",
      "meta-llama_Meta-Llama-3.1-8B-Instruct": "Llama 3.1 8B",
      "VAGOsolutions_Llama-3.1-SauerkrautLM-8b-Instruct": "Llama 3.1 SauerkrautLM 8B",
      "mistralai_Mistral-Nemo-Instruct-2407": "Mistral NeMo 12B",
      "mistralai_Mixtral-8x7B-Instruct-v0.1": "Mixtral 8x7B",
      "meta-llama_Meta-Llama-3.1-70B-Instruct": "Llama 3.1 70B",
      "levenshtein-regex": "Levenshtein/Regex heuristics"
    }

def get_model_label(model_id):
    if model_id in MODEL_LABELS:
        return MODEL_LABELS[model_id]
    else:
        return model_id

def get_metric_label(metric_col):
    return METRIC_LABELS[metric_col]

def order_models(models):
    """
    Orders a list of model IDs based on their sequence in MODEL_LABELS.

    Args:
        models (list): List of model IDs to be ordered.

    Returns:
        list: Ordered list of model IDs.
    """
    # Get the order of models based on MODEL_LABELS
    model_order = {model: idx for idx, model in enumerate(MODEL_LABELS.keys())}

    # Sort the models based on their order in MODEL_LABELS
    ordered_models = sorted(models, key=lambda model: model_order.get(model, float('inf')))

    return ordered_models

PAPER_MODELS_STEP_1 = [model for model in MODEL_LABELS.keys() if model != "levenshtein-regex"]

PAPER_MODELS_STEP_2 = PAPER_MODELS_STEP_1

PAPER_MODELS_STEP_3 = list(MODEL_LABELS.keys())

def get_paper_models(step):
    if step == 1:
        return PAPER_MODELS_STEP_1
    elif step == 2:
        return PAPER_MODELS_STEP_2
    elif step == 3:
        return PAPER_MODELS_STEP_3

def preprocess_data(analysis_file_path, models_to_display):
    analysis_df = pd.read_csv(analysis_file_path)

    # fügt Spalte 'step', 'model' mit Modelnamen und 'prompt_type' hinzu
    analysis_df['step'] = analysis_df['results_file'].apply(extract_step_from_filename)
    if models_to_display and models_to_display[0] == "PAPER_MODELS":
        models_to_display = get_paper_models(analysis_df["step"][0])[::-1]
    analysis_df['model'] = analysis_df.apply(lambda row: extract_model_name_from_filename(row['results_file'], row['step']), axis=1)
    analysis_df['prompt_type'] = analysis_df['results_file'].apply(extract_prompt_type_from_filename)

    #definiert Modelnamen
    unique_models = order_models(analysis_df['model'].unique())
    if models_to_display:
        for model in models_to_display:
            if model not in unique_models:
                raise ValueError(f'{model} is not in csv. Check correct model name.')
        unique_models = models_to_display
    analysis_df = analysis_df[analysis_df['model'].isin(unique_models)]
    analysis_df.sort_values(by=['model'], inplace=True)

    return analysis_df, unique_models

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