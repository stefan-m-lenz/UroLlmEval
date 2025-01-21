import pandas as pd
import argparse
import re
import copy
import numpy as np
import time
from inference_helper import execute_queries, INFERENCE_FRAMEWORKS
from results_helper import (normalize_icd10_answers, create_results_file_and_log_time,
                            ensure_directory, get_evaldata, output_file_exists, no_linebreaks)
from find_dates_heuristics import fuzzy_find_dates

DATE_EXPLANATION = no_linebreaks("""Die Datumsangabe kann tagesgenau, monatsgenau oder nur eine Jahresangabe sein.
Antworten Sie kurz mit dem Datum, wie es im Text angegeben ist.
Achten Sie darauf, dass Sie nur ein Datum angeben, wenn es sich eindeutig auf die Erstdiagnose des Tumors bezieht.
Datumsangaben zu Rezidiven, Metastasen oder zur Therapie des Tumors sind hier nicht relevant.
Wenn es kein passendes Erstdiagnosedatum im Text gibt, antworten Sie mit "Nein".
""")

def build_step3_zero_shot_prompt(snippet, diagnosis_text):
    prompt = (f"""[Diagnosetext]:
{snippet}
[Ende des Diagnosetexts]
""" +
no_linebreaks(f"""Sie haben eine Diagnose im Diagnosetext identifiziert:
"{diagnosis_text}"
Finden Sie eine Datumsangabe zur Erstdiagnose (ED) dieser Tumorerkankung im Text?
{DATE_EXPLANATION}
"""))
    return prompt


# extracts the date for the diagnosis with the specified ICD-10-code in the snippet
def get_date_for_icd10_answer(snippet_id, icd10Code):
    snippets = get_evaldata()
    labels_for_icd10Code = [label for label in snippets[snippet_id]["label"] if label["diagnosis"] == icd10Code]
    if len(labels_for_icd10Code) > 1:
        raise RuntimeError(f"Multiple labels found for snippet_id '{snippet_id}' and icd10Code '{icd10Code}'")
    elif len(labels_for_icd10Code) == 0:
        return ""

    date = labels_for_icd10Code[0]["date"]
    if not date:
        raise RuntimeError(f"""Could not map ICD-10 code "{icd10Code}" to a date in snippet with id "{snippet_id}" """)
    return date


def extract_date_strings_from_text(text):
    date_pattern = r"\b(\d{4}|\d{1,2}/\d{4}|\d{1,2}/\d{2}|\d{1,2}\.\d{1,2}\.\d{4})\b"
    return list(dict.fromkeys(re.findall(pattern=date_pattern, string=text)))



def prompt_question_verify_date_zero_shot_template(snippet, diagnosis_text):
    return (f"""[Diagnosetext]:
{snippet}
[Ende des Diagnosetexts]
""" +
no_linebreaks(f"""Im Diagnosetext haben Sie die Diagnose "{diagnosis_text}" identifiziert.
Zudem ist im Diagnosetext folgende Datumsangabe zu finden: [DATE].
Ist dies das Datum der Erstdiagnose der Tumorerkrankung "{diagnosis_text}"?
Antworten Sie mit "Ja" oder "Nein".
Das Erstdiagnosedatum wird oft mit der Abkürzung "ED" vermerkt. Dann ist es klar, dass es sich um das Erstdiagnosedatum handelt.
Es kann auch sein, dass das Datum ohne die Angabe "ED" angegeben ist, aber ein klarer Bezug zum ersten Diagnose der Tumorerkankung gegeben ist.
In diesem Fall antworten Sie mit "Ja".
Achten Sie aber darauf, dass Sie nur dann mit "Ja" antworten, wenn das Datum "[DATE]" sich eindeutig auf die Erstdiagnose zur Tumordiagnose "{diagnosis_text}" bezieht.
Datumsangaben, die sich ausschließlich auf Rezidive, Metastasen die Therapie des Tumors beziehen, sind hier nicht relevant.
Wenn das Datum "[DATE]" sich nicht klar auf die Zeit (Tag, Monat oder Jahr) der Erstdiagnose bezieht, antworten Sie mit "Nein".
"""))


FEW_SHOT_VERIFICATION_START_PROMPT = no_linebreaks("""Sie erhalten Diagnosetexte und müssen bestimmen, ob ein bestimmtes Datum das Erstdiagnosedatum einer Tumorerkrankung ist.
Antworten Sie mit "Ja" oder "Nein".
Das Erstdiagnosedatum wird oft mit der Abkürzung "ED" vermerkt. Dann ist es klar, dass es sich um das Erstdiagnosedatum handelt.
Es kann auch sein, dass das Datum ohne die Angabe "ED" angegeben ist, aber ein klarer Bezug zur ersten Diagnose der Tumorerkankung gegeben ist.
In diesem Fall antworten Sie mit "Ja".
Achten Sie aber darauf, dass Sie nur dann mit "Ja" antworten, wenn das genannte Datum sich eindeutig auf die Erstdiagnose zur Tumordiagnose bezieht.
Datumsangaben, die sich ausschließlich auf Rezidive, Metastasen die Therapie des Tumors beziehen, sind hier nicht relevant.
Wenn das Datum sich nicht klar auf die Zeit (Tag, Monat oder Jahr) der Erstdiagnose bezieht, antworten Sie mit "Nein".
""")

def prompt_question_verify_date_four_shot_template(snippet, diagnosis_text):
    return [
        {
            "role": "user",
            "content": FEW_SHOT_VERIFICATION_START_PROMPT
        },
        {
            "role": "assistant",
            "content": """OK, geben Sie mir die Texte und die Datumsangaben. Ich werde mit "Ja" antworten, wenn das Datum die Erstdiagnosedatum zur Diagnose beschreibt, ansonsten antworte ich mit "Nein"."""
        },
        {
            "role": "user",
            "content": """Ist "1999" die Datumsangabe für die Erstdiagnose der Tumorerkrankung "Prostatakarzinom" im Text? Text:
Z.n. Prostatakarzinom ED 1999
Z.n. Nierentransplantation 1980 links"""
        },
        {
            "role": "assistant",
            "content": """Ja. 1999 ist eindeutig als Erstdiagnosedatum (ED) bei der Diagnose "Prostatakarzinom" angegeben."""
        },
        {
            "role": "user",
            "content": """Ist "07/2017" die Datumsangabe für die Erstdiagnose der Tumorerkrankung "Urothelkarzinom" im Text? Text:
- TUR-B-NR mit dem Nachweis eines Urothelkarzinoms der Harnblase pT2G3
- Prostatakarzinom ED 07/2017"""
        },
        {
            "role": "assistant",
            "content": """Nein. Es gibt kein Datum zum Urothelkarzinom, die Datumsangabe "07/2017" bezieht sich auf das Prostatakarzinom."""
        },
        {
            "role": "user",
            "content": """Ist "10/2011" die Datumsangabe für die Erstdiagnose der Tumorerkrankung "Mammakarzinonm" im Text? Text:
09/2011: Invasives lobuläres Mammakarzinom links, Stadium II
10/2011: Durchführung einer brusterhaltenden Therapie (BET)
11/2011: Beginn der adjuvanten Strahlentherapie, abgeschlossen 12/2011
""",
        },
        {
            "role": "assistant",
            "content": """Nein. Die Datumsangabe "10/2011" bezieht sich auf eine Therapie des Mammakarzinoms, nicht auf die Erstdiagnose der Tumorerkrankung."""
        },
        {
            "role": "user",
            "content": """Ist "09/2011" die Datumsangabe für die Erstdiagnose der Tumorerkrankung "Mammakarzinom" im Text? Text:
09/2011: Invasives lobuläres Mammakarzinom links, Stadium II
10/2011: Durchführung einer brusterhaltenden Therapie (BET)
11/2011: Beginn der adjuvanten Strahlentherapie, abgeschlossen 12/2011
""",
        },
        {
            "role": "assistant",
            "content": """Ja. Hier ist anzunehmen, dass sich die Datumsangabe auf die Erstdiagnose des Mammakarzinoms bezieht, auch wenn die Angabe nicht explizit mit "ED" gkennzeichnet ist."""
        },
        {
            "role": "user",
            "content": f"""Ist "[DATE]" die Datumsangabe für die Erstdiagnose der Tumorerkrankung "{diagnosis_text}" im Text? Text:
{snippet}"""
        }
    ]


def create_date_suggestions(snippet, diagnosis_text):
    dates_in_snippet = extract_date_strings_from_text(snippet)
    if len(dates_in_snippet) == 0:
        # if there are no dates in the text, this will not be used for a real query later
        # These snippets will be filtered out in run_evaluation_step3
        return ""
    elif len(dates_in_snippet) == 1:
        return no_linebreaks(f"""Im Diagnosetext ist folgende Datumsangabe zu finden: {dates_in_snippet[0]}.
Ist dies das Datum der Erstdiagnose zur Tumordiagnose "{diagnosis_text}"?""")
    else: # len > 1
        return no_linebreaks(f"""Folgende Datumsangaben sind im Text zu finden: {", ".join(dates_in_snippet)}.
Ist eine der Datumsangaben das Datum der Erstdiagnose zur Tumordiagnose "{diagnosis_text}"?""")


def prompt_question_with_dates(snippet, diagnosis_text):
    date_part = create_date_suggestions(snippet, diagnosis_text)
    return (f"""[Diagnosetext]:
{snippet}
[Ende des Diagnosetexts]
""" +
no_linebreaks(f"""
Im Diagnosetext haben Sie die Diagnose "{diagnosis_text}" identifiziert.
{date_part}
Antworten Sie mit dem Datum, wenn es die Zeit der Erstdiagnose der Tumordiagnose "{diagnosis_text}" beschreibt, oder mit "Nein".
"""))


THREE_SHOT_START_PROMPT = [
        {
            "role": "user",
            "content": no_linebreaks(f"""Finden Sie eine Datumsangabe zur Erstdiagnose (ED) einer gegebenen Tumorerkankung?
{DATE_EXPLANATION}""")
        },
        {
            "role": "assistant",
            "content": no_linebreaks("""OK, geben Sie mir die Texte und Diagnosen. Ich werde mit dem Datum antworten, wie es im Text angegeben ist, wenn ich ein Erstdiagnosedatum zur Diagnose finde.
Wenn es kein Datum dazu gibt, antworte ich mit "Nein".""")
        }
]


def qa_with_dates(snippet, diagnosis_text, answer):
    return [
        {
           "role": "user",
           "content": prompt_question_with_dates(snippet, diagnosis_text)
        },
        {
            "role": "assistant",
            "content": answer
        }
    ]


def build_query_messages(prompt_type, snippet, diagnosis_text):
    if prompt_type == "0":
        return [
            {
                "role": "user",
                "content": build_step3_zero_shot_prompt(snippet, diagnosis_text),
            }
        ]
    elif prompt_type == "0-dates":
        return [
            {
                "role": "user",
                "content": (
                    build_step3_zero_shot_prompt(snippet, diagnosis_text) + "\n" +
                    create_date_suggestions(snippet, diagnosis_text)
                )
            }
        ]
    elif prompt_type == "0-verify":
        return [
            {
                "role": "user",
                "content": prompt_question_verify_date_zero_shot_template(snippet, diagnosis_text)
            }
        ]
    elif prompt_type == "4-verify":
        return prompt_question_verify_date_four_shot_template(snippet, diagnosis_text)

    elif prompt_type == "3":
        return THREE_SHOT_START_PROMPT + [
            {
                "role": "user",
                "content": """Gibt es ein Erstdiagnosedatum zur Diagnose "Prostatakarzinom" in folgendem Text? Text:
Z.n. Prostatakarzinom ED 1999
Z.n. Nierentransplantation 1980 links"""
            },
            {
                "role": "assistant",
                "content": "1999"
            },
            {
                "role": "user",
                "content": """Gibt es ein Erstdiagnosedatum zur Diagnose "Urothelkarzinom" in folgendem Text? Text:
- TUR-B-NR mit dem Nachweis eines Urothelkarzinoms der Harnblase pT2G3
- Prostatakarzinom ED 07/2017"""
            },
            {
                "role": "assistant",
                "content": "Nein (Es gibt kein Datum zum Urothelkarzinom, nur eines für das Prostatakarzinom.)"
            },
            {
                "role": "user",
                "content": """Gibt es ein Erstdiagnosedatum zur Diagnose "Mammakarzinom" in folgendem Text? Text:
09/2011: Invasives lobuläres Mammakarzinom links, Stadium II
10/2011: Durchführung einer brusterhaltenden Therapie (BET)
11/2011: Beginn der adjuvanten Strahlentherapie, abgeschlossen 12/2011
""",
            },
            {
                "role": "assistant",
                "content": "09/2011"
            },
            {
                "role": "user",
                "content": f"""Gibt es ein Erstdiagnosedatum zur Diagnose "{diagnosis_text}" in folgendem Text? Text:\n{snippet}"""
            }
        ]
    elif prompt_type == "3-dates":
        return THREE_SHOT_START_PROMPT + (
            qa_with_dates(snippet="""Z.n. Prostatakarzinom ED 1999
Z.n. Nierentransplantation 1980 links""", diagnosis_text="Prostatakarzinom", answer="1999") +
            qa_with_dates(snippet="""- TUR-B-NR mit dem Nachweis eines Urothelkarzinoms der Harnblase pT2G3
- Prostatakarzinom ED 07/2017""",
                          diagnosis_text="Urothelkarzinom", answer="Nein (Es gibt kein Datum zum Urothelkarzinom, nur eines für das Prostatakarzinom)") +
            qa_with_dates(snippet="""09/2011: Invasives lobuläres Mammakarzinom links, Stadium II
10/2011: Brusterhaltende Therapie (BET)
11/2011: Adjuvante Strahlentherapie, abgeschlossen 12/2011""", diagnosis_text="Mammakarzinom", answer = "09/2011")
        ) + [
            {
                "role": "user",
                "content": prompt_question_with_dates(snippet, diagnosis_text)
            }
        ]


def build_step3_query_for_step1_answer(model, prompt_type, snippet, diagnosis_text):
    if prompt_type == "regex-match-same-line":
        return {} # no model queried here

    query = {
        "model": model,
        "stream": False,
        "options": {
            "seed": 112233,
            "temperature": 0,
            "num_predict": 50
        }
    }

    query["messages"] = build_query_messages(prompt_type=prompt_type, snippet=snippet, diagnosis_text=diagnosis_text)
    return query


def extract_dates_from_snippets():
    snippets = get_evaldata()
    snippet_ids = range(0, len(snippets))
    dates = [extract_date_strings_from_text(snippet["text"]) for snippet in snippets]
    return pd.DataFrame({"snippet_id": snippet_ids, "regex_matched_dates": dates})


REGEX_MATCH_TYPES = {
    "regex-match-same-line" : 0,
    "regex-match-line-dist-1": 1,
    "regex-match-line-dist-2": 2
}


def regex_match_heuristics_result(snippet_text, diagnosis, max_line_dist):
    dates = fuzzy_find_dates(snippet_text, diagnosis, max_line_dist)
    if dates:
        return dates[0]
    else:
        return "Nein"


def regex_match_heuristics_results(results_df, max_line_dist):
    results_df["model_answer"] = results_df.apply(lambda row: regex_match_heuristics_result(row.snippet, row.diagnosis, max_line_dist), axis = 1)
    return results_df


def change_queries_to_incorporate_dates(results_df, max_line_dist=2):
    results_df["dates"] = results_df.apply(
        # deterministic order of dates with dict.fromkeys
        lambda row: list(dict.fromkeys(fuzzy_find_dates(row.snippet, row.diagnosis, max_line_dist=max_line_dist))),
        axis=1
    )

    # Handling rows with no dates: Set answer to "Nein" and use an empty query
    results_df['query'] = results_df.apply(
        lambda row: {} if len(row['dates']) == 0 else row['query'],
        axis=1
    )
    results_df["model_answer"] = ""
    results_df.loc[results_df["dates"].apply(len) == 0, "model_answer"] = "Nein"

    def updated_query(query_template, date):
        query = copy.deepcopy(query_template)
        # modifying only the last message works for zero-shot and few-shot queries:
        query["messages"][-1]["content"] = query["messages"][-1]["content"].replace("[DATE]", date)
        return query

    # Expanding rows with dates into multiple rows, one for each date
    def expand_dates(row):
        if row["dates"]:
            expanded = [
                {**row, "query": updated_query(row["query"], date), "queried_date" : date}
                for date in row["dates"]
            ]
            return pd.DataFrame(expanded)  # Create a DataFrame from the list of dictionaries
        else:
            return pd.DataFrame([row])

    # Apply expand_dates and concatenate all DataFrames created from each row
    expanded_rows = pd.concat(results_df.apply(expand_dates, axis=1).tolist(), ignore_index=True)
    expanded_rows.drop(columns=["dates"], inplace=True)  # Optionally remove 'dates' column if no longer needed

    return expanded_rows


def evaluate_step3_model_prompt(model, prev_step_results, prompt_type, inference_framework, output_dir=".", test_only=False, show_progress=True):

    if output_file_exists(model=model, step=3, input_filename=prev_step_results, prompt_type=prompt_type, output_dir=output_dir):
        return

    step2_results_df = pd.read_csv(prev_step_results, dtype={"model_answer": pd.StringDtype()})
    step2_results_df = step2_results_df.dropna(subset=['snippet_id'])
    step2_results_df = normalize_icd10_answers(step2_results_df)
    step2_results_df = step2_results_df.dropna(subset="model_answer_normalized")

    results_df = [{
            "snippet_id": row.snippet_id,
            "snippet": row.snippet,
            "query": build_step3_query_for_step1_answer(model, prompt_type, row.snippet, row.diagnosis),
            "diagnosis": row.diagnosis, # part of answer from step 1
            "model_answer_step2": row.model_answer, # answer from step 2, given the diagnosis returned from the model in step 1
            "expected_answer": get_date_for_icd10_answer(row.snippet_id, row.model_answer_normalized)
        } for row in step2_results_df.itertuples()]

    results_df = pd.DataFrame(results_df)

    # remove values which could not be matched to a diagnosis code
    results_df = results_df.loc[results_df["expected_answer"] != ""]

    results_df = results_df.merge(extract_dates_from_snippets(), on="snippet_id", how="left")

    if prompt_type in ["0-dates", "3-dates"]:
        rows_with_dates = results_df["regex_matched_dates"].apply(bool)
        results_df_without_dates = results_df[~rows_with_dates]
        results_df = results_df[rows_with_dates] # query LLM only for rows with regex matched dates in this scenario
    elif prompt_type.endswith("-verify"):
        results_df = change_queries_to_incorporate_dates(results_df)

    if test_only:
        results_df = results_df.head(test_only).copy()

    start_time = time.time()

    if prompt_type in REGEX_MATCH_TYPES:
        results_df = regex_match_heuristics_results(results_df, REGEX_MATCH_TYPES[prompt_type])
        model = "levenshtein-regex"
    else: #actual LLM queries
        results_df = execute_queries(query_df=results_df, inference_framework=inference_framework,
                                     show_progress=show_progress)

    if prompt_type in ["0-dates", "3-dates"]:
        # do as if the model had answered no for snippets without regex matched dates:
        results_df_without_dates["model_answer"] = "Nein"
        results_df_without_dates["query_execution_time"] = np.nan
        results_df = pd.concat([results_df, results_df_without_dates], axis=0)

    create_results_file_and_log_time(df=results_df, model=model, step=3,
                                     start_time = start_time,
                                     input_filename=prev_step_results, prompt_type=prompt_type,
                                     output_dir=output_dir)

PROMPT_TYPES = ["0", "0-dates", "3", "3-dates",
                "0-verify", "4-verify"] + list(REGEX_MATCH_TYPES.keys())


def run_evaluation_step3(model, prev_step_results, prompt_type=None, inference_framework="ollama", output_dir="output", show_progress=True, test_only = None):
    ensure_directory(output_dir)
    if prompt_type:
        evaluate_step3_model_prompt(model=model,
                                    prev_step_results=prev_step_results,
                                    prompt_type=prompt_type,
                                    inference_framework=inference_framework, output_dir=output_dir,
                                    show_progress=show_progress,
                                    test_only=test_only)
    else:
        for prompt_type in PROMPT_TYPES:
            print(f"""Evaluating prompt type {prompt_type}""")
            evaluate_step3_model_prompt(model=model,
                                        prev_step_results=prev_step_results,
                                        prompt_type=prompt_type,
                                        inference_framework=inference_framework,
                                        output_dir=output_dir,
                                        show_progress=show_progress,
                                        test_only=test_only)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation step 3 based on step 2")
    parser.add_argument("--model", type=str, help="Model name to use.", default="mistral")
    parser.add_argument("--prev-step-results", type=str, help="File with results from step 2", required=True)
    parser.add_argument("--prompt-type", choices=PROMPT_TYPES, help="""The type of the prompt that is used for querying the LLM.
If not specified, all prompt types are used.""", default=None)
    parser.add_argument("--inference-framework", choices=INFERENCE_FRAMEWORKS, default="ollama",
                        help="""The inference framework: "ollama" (default) or "transformers".""")
    parser.add_argument("--test-only", type=int, help="Number of prompts to use (for testing the script)", default=None)
    parser.add_argument("--output-dir", type=str, default="output",
                        help="A directory for creating the results files (will be created if it doesn't exist).")

    args = parser.parse_args()

    run_evaluation_step3(model=args.model, prev_step_results=args.prev_step_results,
                         prompt_type=args.prompt_type,
                         inference_framework=args.inference_framework,
                         output_dir=args.output_dir,
                         test_only=args.test_only)