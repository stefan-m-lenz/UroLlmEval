import numpy as np
import json
import re
import os
import csv
import time
from datetime import datetime
from itertools import chain
from collections import Counter
import pandas as pd
import datasets
from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix


EVALDATA=None

def get_evaldata():
    global EVALDATA
    if EVALDATA is None:
        EVALDATA = datasets.load_dataset("stefan-m-lenz/UroLlmEvalSet", split="eval")
        EVALDATA = [row for row in EVALDATA] # convert to simple list
    return EVALDATA


def no_linebreaks(text):
    return re.sub(r"\n(?!\n)", " ", text)


def parse_json_array(str):
    # find matching parenthesis.
    # llama3 outputs nested JSONs for some reason)
    str = str[(str.find('[')+1):]
    n_open_brackets = 1

    i = 0
    while i < len(str) and n_open_brackets != 0:
        nextchar = str[i]
        if re.match(r"\s", nextchar):
            pass
        elif nextchar == "[":
            n_open_brackets += 1
        elif nextchar == "]":
            n_open_brackets -= 1
        i += 1

    jsonstr = "[" + str[0:i]

    # replace german quotation marks if used
    jsonstr = jsonstr.replace("“", '"')
    jsonstr = jsonstr.replace("”", '"')

    return json.loads(jsonstr)

def flatten_list(nested_list):
    # Keep flattening while there are any sublists left
    while any(isinstance(elem, list) for elem in nested_list):
        nested_list = list(chain.from_iterable(elem if isinstance(elem, list) else [elem] for elem in nested_list))
    return nested_list

def extract_diagnosis_texts(string_with_json_array):
    try:
        diagnoses_list = parse_json_array(string_with_json_array)

    except json.JSONDecodeError:
        return []

    diagnoses_list = flatten_list(diagnoses_list)
    return diagnoses_list


# Return a dictionary with the keys as the unique elements of the collection and the values as their counts.
# The elements are sorted before they are put into the dictionary,
# first by count, then by element (ascending) as a tiebreaker.
# (This way the order is deterministic across different runs.)
def count_dict(collection):
    counter = Counter(collection)
    sorted_counter = dict(sorted(counter.items(), key=lambda item: (-item[1], item[0])))
    return sorted_counter


# Calculates several metrics from the confusion matrics of two pandas series, v_true and v_pred
# that contain boolean values
def calculate_metrics_for_bool_answer(v_true, v_pred):

    if len(v_pred) != len(v_true):
        raise ValueError("Length of vectors must agree.")

    nan_mask = v_pred.isna()

    n_correct_total = int(np.sum((v_pred == v_true) & ~nan_mask))
    p_correct_total = n_correct_total / len(v_pred)

    p_na = nan_mask.sum() / len(v_pred)
    v_pred = v_pred[~ nan_mask]
    # convert to numpy array as otherwise y_pred may be of type pandas.core.arrays.boolena.BooleanArray,
    # which scikit cannot handle

    v_pred = np.asarray(v_pred).astype(bool)
    v_true = v_true[~ nan_mask]
    v_true = np.asarray(v_true).astype(bool)

    precision = precision_score(v_true, v_pred, zero_division=0)
    recall = recall_score(v_true, v_pred, zero_division=0)
    f1 = f1_score(v_true, v_pred, zero_division=0)

    cm = confusion_matrix(v_true, v_pred, labels=[0, 1])

    # cm is in the form:
    # [[TN, FP],
    #  [FN, TP]]

    # Extract TN and FP
    TN, FP = cm[0, 0], cm[0, 1]

    # Calculate Specificity
    if TN == 0 and FP == 0:
        specificity = np.nan
    else:
        specificity = TN / (TN + FP)

    return {
        'precision': precision,
        'specificity': specificity,
        'recall': recall,
        'f1': f1,
        'p_na': p_na,
        'n_correct_total': n_correct_total,
        'p_correct_total': p_correct_total
    }


def count_correct(v):
    # Check whether v is a boolean vector. If not, transform it to one.
    if v.dtype == bool:
        pass
    elif v.dtype == float or v.dtype == object:
        # Check for values that are not 0.0, 1.0, or NaN
        if not v.isin([0.0, 1.0, np.nan]).all():
            raise ValueError("The float vector contains values other than 0.0, 1.0, or NaN.")
        # Convert float series to boolean
        v = v.map({1.0: True, 0.0: False}) # this mapping respects the NAs
    else:
        raise(ValueError("Vector with boolean values required"))

    nan_mask = v.isna()
    p_na = v.isna().sum() / len(v)

    n_correct_total = int(np.sum(v & ~nan_mask))
    p_correct_total = n_correct_total / len(v)

    v = v[~ nan_mask]
    if len(v) > 0:
        p_correct = sum(v) / len(v)
    else:
        p_correct = np.nan
    ret = {
        'p_correct': p_correct,
        'p_correct_total': p_correct_total,
        'p_na': p_na
    }
    return ret


def normalize_icd10_answers(results_df, model_answer_column="model_answer"):
    """
    Adds a column f"{model_answer_column}_normalized" to the results_df that contains ICD-10-Codes
    that can be mapped to values in the XML file.

    Parameters:
        df (pd.DataFrame): Input DataFrame containing a 'model_answer' column.
    """
    def normalize_and_truncate(text):
        match = re.search(r"\b([CD]{1}[0-9]{2})\b", string=text)
        if match:
            return match.group(1)
        else:
            return None

    # Apply the normalization and truncation function to the 'model_answer' column
    results_df[f"{model_answer_column}_normalized"] = results_df[model_answer_column].apply(normalize_and_truncate)
    return results_df


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


def create_output_filename(step, model, prompt_type, input_filename):
    model = model.replace("/", "_")
    model = model.replace(":", "_")
    if step > 1:
        # get only the name of the file without file ending
        input_filename = os.path.splitext(os.path.basename(input_filename))[0]
        model_name_previous_step = extract_model_name_from_filename(filename=input_filename, step=step-1)
    else:
        if prompt_type:
            return f"step1_{model}__{prompt_type}.csv"
        else:
            return f"step1_{model}.csv"

    if prompt_type:
        prompt_type = f"_{prompt_type}"
    else:
        prompt_type = ""
    if model_name_previous_step == model:
        filename_without_model = re.sub(f"_{model}", "", input_filename)
    else:
        filename_without_model = input_filename

    return f"step{step}_{model}_{prompt_type}_{filename_without_model}.csv"


def ensure_directory(path):
    # Check if the path exists
    if os.path.exists(path):
        # Check if the path is a directory
        if not os.path.isdir(path):
            raise NotADirectoryError(f"The path '{path}' exists but is not a directory.")
    else:
        # If the path does not exist, create the directory
        os.makedirs(path)


def create_output_path(step, model, prompt_type, input_filename, output_dir):
    output_filename = create_output_filename(step, model, prompt_type=prompt_type, input_filename=input_filename)
    return os.path.abspath(os.path.join(output_dir, output_filename))


def create_results_file(df, step, model, input_filename=None, prompt_type=None, output_dir="."):
    output_path = create_output_path(step, model, prompt_type=prompt_type, input_filename=input_filename, output_dir=output_dir)
    df.to_csv(output_path)
    output_path

def create_results_file_and_log_time(df, step, model, start_time, input_filename=None, prompt_type=None, output_dir="."):
    end_time = time.time()
    output_filename = create_output_filename(step, model, prompt_type=prompt_type, input_filename=input_filename)
    output_path = os.path.abspath(os.path.join(output_dir, output_filename))
    df.to_csv(output_path)
    log_time(output_dir=output_dir, step=step, model=model, prompt_type=prompt_type, results_file=output_filename,
             elapsed_seconds=(end_time - start_time))


def log_time(output_dir, step, model, prompt_type, results_file, elapsed_seconds):
    """
    Appends a log entry to 'log.csv' in the specified output directory.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Define the path to the log file
    log_file = os.path.join(output_dir, "log.csv")

    # Define the fields (keys of the log entry)
    # Add 'timestamp' and 'elapsed_seconds' as default fields
    fields = ["timestamp", "step", "model", "prompt_type", "results_file", "elapsed_seconds"]

    # Check if the file exists
    file_exists = os.path.isfile(log_file)

    # Open the file in append mode
    with open(log_file, mode="a", newline="", encoding="utf-8") as csvfile:
        # Initialize the CSV writer
        writer = csv.DictWriter(csvfile, fieldnames=fields)

        # Write the header if the file does not exist
        if not file_exists:
            writer.writeheader()

        # Create the log entry
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": elapsed_seconds,
            "step": step,
            "model": model,
            "prompt_type": prompt_type,
            "results_file": results_file,
            "elapsed_seconds": elapsed_seconds,
        }

        # Write the log entry
        writer.writerow(log_entry)


def output_file_exists(step, model, prompt_type, input_filename, output_dir):
    output_path = create_output_path(step, model, prompt_type=prompt_type, input_filename=input_filename, output_dir=output_dir)
    if os.path.exists(output_path):
        print(f"""Output file already exists on path "{output_path}". Skipping this part.""")
        return True
    else:
        return False


def output_analysis_result(result, output):
    if output == "stdout":
        print(json.dumps(result, indent = 4))
    elif output.endswith(".json"):
        with open(output, 'w') as f:
            f.write(json.dumps(result, indent=4))
    elif output.endswith(".csv"):
        if isinstance(result, list):
            pd.DataFrame(result).to_csv(output)
        else:
            pd.DataFrame([result]).to_csv(output)
    else:
        raise(ValueError(f"Invalid argument '--output': {output}"))


METRIC_LABELS = {
    "recall": "Sensitivity",
    "specificity": "Specificity",
    "p_correct_total": "Accuracy (NA=wrong)",
    "p_correct": "Accuracy (excl. NA)",
    "p_na": "NA",
    "p_C77_79": "C77-C79 answers",
    "p_all_diagnoses_found_in_snippets_with_diagnoses": "All diagnoses found",
    "p_no_other_diagnoses_found_for_snippet" : "No incorrect diagnosis",
    "p_snippets_correct": "Snippet correct"
}

def get_metric_label(metric_col):
    return METRIC_LABELS[metric_col]


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
    "levenshtein-regex": "Levenshtein/Regex heuristics",
}

def get_model_label(model_id):
    if model_id in MODEL_LABELS:
        return MODEL_LABELS[model_id]
    else:
        return model_id


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