import pandas as pd
import re
import argparse
import numpy as np
import os
import glob
from functools import reduce
from results_helper import (normalize_icd10_answers, count_correct, output_analysis_result,
                            extract_prompt_type_from_filename, count_dict)

def normalize_year(year):
    if year < 100:
        if year > 30:
            return year + 1900
        else:
            return year + 2000
    else:
        return year


def normalize_date_str(date_str):
    if not date_str:
        return None
    if re.match(pattern=r"^\d{4}(-\d{2}(-\d{2})?)?$", string=date_str):
        return date_str # already normalized
    # month or year precision
    match = re.search(pattern=r"(?P<day>\d{1,2})\.(?P<month>\d{1,2})\.(?P<year>\d{4})", string=date_str)
    if match:
        return f"""{normalize_year(int(match.group("year")))}-{int(match.group("month")):02d}-{int(match.group("day")):02d}"""
    else:
        match = re.search(pattern=r"\d*/?\d+", string=date_str)
        if match:
            date_str = match.group(0)
            if "/" in date_str:
                # format date in the format YYYY-MM
                month, year = [int(str) for str in date_str.split("/")]
                if month < 1 or month > 12:
                    return None
                year = normalize_year(year)
                return f"{year}-{month:02d}"
            else:
                # pure year, simply return it
                return date_str
        else:
            return None


# answer may be a date or yes no
def normalize_date_answer(text):
    if not text:
        return None
    date_str = normalize_date_str(text)
    if date_str:
        return date_str
    else:
        if re.match(r"\s*[Nn]ein", text) or re.search(r"\s+not\s+", text):
            return "0"
        else:
            return None


def normalized_date_answer_to_bool(row):
    if row.model_answer_normalized:
        return row.model_answer_normalized == row.expected_answer
    else:
        return np.nan

def yes_no_answer_to_bool(answer):
    answer = answer.lstrip()
    if re.match(pattern=r'"*[jJ]a', string=answer):
        return True
    elif re.match(pattern=r'"*[Nn]ein', string=answer):
        return False
    else:
        return np.nan

# expects normalized dates of form YYYY or YYYY-MM or YYYY-MM-DD
def find_common_date_part(date_str1, date_str2):
    if date_str1[0:4] == date_str2[0:4]: # year overlap
        if len(date_str1) > 4 and len(date_str2) > 4 and date_str1[5:7] == date_str2[5:7]:
            if len(date_str1) > 7 and len(date_str2) > 7 and date_str1 == date_str2:
                return date_str1 # day, month and year identical
            else:
                return date_str1[0:7] # month and year identical, not more
        else: # only years overlap
            return date_str1[0:4]
    else:
        return ""


def combine_yes_no_answer_to_date_answer(group):
    # Assert that all 'expected_answer' values in the group are the same
    assert group["expected_answer"].nunique() == 1, "Expected answers are not the same within the group"

    # Determine the value for 'model_answer_combined'
    true_entries = group.loc[group["model_answer_bool"] == True]
    if len(true_entries) == 0:
        # If all answers are "Nein"/no, the combined answer is that there is no date for the diagnosis
        model_answer_combined = "0"
    elif len(true_entries) == 1:
        # If exactly one answer is "Ja"/yes, the corresponding date is the combined answer.
        model_answer_combined = normalize_date_str(true_entries["queried_date"].values[0])
    else:
        # If there is more than one answer with "yes",
        # check if there is overlap between the dates. In this case, take the more specific one
        normalized_dates = true_entries["queried_date"].apply(normalize_date_str)
        date_overlap = reduce(find_common_date_part, normalized_dates)
        if not date_overlap:
            # multiple non overlapping dates for the same diagnosis lead to no combined answer at all
            model_answer_combined = None
        else: # all dates have some overlap, use most detailed (longest) one
            # (The implementation could be more sophisticated for real-world application
            # and handle more possible cases but it should suffice for the purpose here.)
            index_longest_date = normalized_dates.apply(len).idxmax()
            model_answer_combined = normalized_dates.loc[index_longest_date]

    # Add the model_answer_combined to the first row of the group
    group["model_answer_combined"] = model_answer_combined
    group["model_answer"] = " + ".join(group["model_answer"])

    return group.iloc[0]


def combine_answers_of_yes_no_prompts(results_df):
    results_df = normalize_icd10_answers(results_df, model_answer_column="model_answer_step2")
    results_df.rename(columns={"model_answer_step2_normalized": "diagnosis_code"}, inplace=True)
    results_df["model_answer_bool"] = results_df["model_answer"].apply(yes_no_answer_to_bool)
    grouped = results_df.groupby(["snippet_id", "diagnosis_code"])

    results_df = grouped[results_df.columns].apply(combine_yes_no_answer_to_date_answer).reset_index(level=[0, 1], drop=True)
    return results_df


def run_analysis(file_path):
    results_df = pd.read_csv(file_path, dtype={"expected_answer": pd.StringDtype(), "model_answer": pd.StringDtype()})

    # remove values which could not be matched to a diagnosis code
    # (will not be needed in final versions as these are excluded when running the evaluation)
    results_df.dropna(subset=["expected_answer"], inplace=True)

    prompt_type = extract_prompt_type_from_filename(os.path.basename(file_path))
    if prompt_type.endswith("-verify"):
        results_df = combine_answers_of_yes_no_prompts(results_df)
        results_df["model_answer_normalized"] = results_df["model_answer_combined"]
    else:
        results_df["model_answer_normalized"] = results_df["model_answer"].apply(normalize_date_answer)

    results_df["model_answer_bool"] = results_df.apply(normalized_date_answer_to_bool, axis=1)

    wrong_results = results_df[results_df["model_answer_bool"] == False]
    wrong_results = dict(zip(wrong_results["expected_answer"], wrong_results["model_answer"]))

    ret = {
        "results_file": os.path.splitext(os.path.basename(file_path))[0],
        **count_correct(results_df["model_answer_bool"]),
        "wrong_results": wrong_results,
        "na_values": count_dict(results_df[results_df['model_answer_bool'].isna()]['model_answer'])
    }
    return ret


def run_analyses_on_dir(dir_path):
    step3files = glob.glob(os.path.join(dir_path, "step3_*.csv"))
    results = [run_analysis(file) for file in step3files]
    return results


def analyze_step3_results(results_path, output):
    if os.path.isdir(results_path):
        result = run_analyses_on_dir(results_path)
    else:
        result = run_analysis(results_path)
    output_analysis_result(result, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze results from evaluation step 3")
    parser.add_argument("path", help="The path to a file containing the results of step 3")
    parser.add_argument("--output", help="Output file. May end with .json or .csv", default="stdout")

    args = parser.parse_args()

    analyze_step3_results(results_path=args.path, output=args.output)