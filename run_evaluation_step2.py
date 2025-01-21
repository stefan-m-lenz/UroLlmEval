import pandas as pd
import argparse
import time
from inference_helper import execute_queries, INFERENCE_FRAMEWORKS
from results_helper import (extract_diagnosis_texts, create_results_file_and_log_time,
                            ensure_directory, get_evaldata, output_file_exists, no_linebreaks)

C77_C79_EXPLANATION = """Die Metastasierung muss bei der Kodierung nicht berücksichtigt werden, das heißt die Codes C77, C78 und C79 für sekundäre Neubildungen sollten nicht verwendet werden. Die Antwort sollte also niemals "C77", "C78" oder "C79" sein."""

def build_step2_zero_shot_prompt(prompt_type, snippet, diagnosis_text):
    if prompt_type == "0-uro":
        prompt = no_linebreaks(f"""Was ist der ICD-10-Code für die Diagnose "{diagnosis_text}"? Antworten Sie nur kurz mit dem 3-stelligen ICD-10-Code.
Wenn es sich bei der Diagnose "{diagnosis_text}" um ein Prostatakarzinom handelt,
antworten Sie zum Beispiel mit "C61", bei einem Urothelkarzinom der Harnblase mit "C67".
{C77_C79_EXPLANATION}
""")
    elif prompt_type == "0-gyn":
        prompt = no_linebreaks(f"""Was ist der ICD-10-Code für die Diagnose "{diagnosis_text}"? Antworten Sie nur kurz mit dem 3-stelligen ICD-10-Code.
Wenn es sich bei der Diagnose "{diagnosis_text}" um ein Mammakarzinom handelt,
antworten Sie zum Beispiel mit "C50", bei einem Ovarialkarzinom mit "C56".
{C77_C79_EXPLANATION}
""")
    elif prompt_type == "0-uro-ctx":
        prompt = (f"""[Diagnosetext]:
{snippet}
[Ende des Diagnosetexts]
""" +
no_linebreaks(f"""
Sie haben eine Diagnose im Diagnosetext identifiziert:
"{diagnosis_text}"
Was ist der ICD-10-Code für diese Diagnose? Antworten Sie nur kurz mit dem 3-stelligen ICD-10-Code.
Wenn es sich bei der Diagnose "{diagnosis_text}" um ein Prostatakarzinom handelt,
antworten Sie zum Beispiel mit "C61", bei einem Urothelkarzinom der Harnblase mit "C67".
{C77_C79_EXPLANATION}
"""))
    elif prompt_type == "0-gyn-ctx":
        prompt = (f"""[Diagnosetext]:
{snippet}
[Ende des Diagnosetexts]
""" +
no_linebreaks(f"""Sie haben eine Diagnose im Diagnosetext identifiziert:
"{diagnosis_text}"
Was ist der ICD-10-Code für diese Diagnose? Antworten Sie nur kurz mit dem 3-stelligen ICD-10-Code.
Wenn es sich bei der Diagnose "{diagnosis_text}" um ein Mammakarzinom handelt,
antworten Sie zum Beispiel mit "C50", bei einem Ovarialkarzinom mit "C56".
{C77_C79_EXPLANATION}
"""))
    return prompt


QAS = {}
QAS["2-uro"] = [
    {
        "role": "user",
        "content": "Was ist der ICD-10-Code für diese Diagnose?\nDiagnose: Metastasiertes Prostatakarzinom"
    },
    {
        "role": "assistant",
        "content": "C61"
    },
    {
        "role": "user",
        "content": "Was ist der 3-stellige ICD-10-Code für diese Diagnose?\nDiagnose: Urothelkarzinom der Harnblase"
    },
    {
        "role": "assistant",
        "content": "C67"
    }
]
QAS["2-uro-ctx"] = QAS["2-uro"]

QAS["2-gyn"] = [
        {
        "role": "user",
        "content": "Was ist der ICD-10-Code für diese Diagnose?\nDiagnose: Metastasiertes Mammakarzinom"
    },
    {
        "role": "assistant",
        "content": "C50"
    },
    {
        "role": "user",
        "content": "Was ist der 3-stellige ICD-10-Code für diese Diagnose?\nDiagnose: Ovarialkarzinom"
    },
    {
        "role": "assistant",
        "content": "C56"
    }
]
QAS["2-gyn-ctx"] = QAS["2-gyn"]


def build_query_messages(prompt_type, snippet, diagnosis_text):
    if prompt_type.startswith("0"):
        return [
            {
                "role": "user",
                "content": build_step2_zero_shot_prompt(prompt_type, snippet, diagnosis_text)
            }
        ]
    elif prompt_type.startswith("2"):
        chat_start = [
            {
                "role": "user",
                "content": no_linebreaks(f"""Ihnen werden Diagnosetexte vorgelegt, zu denen Sie ICD-10-Codes zuordnen sollen.
Die Frage ist jeweils: Was ist der ICD-10-Code für diese Diagnose?
Antworten Sie darauf nur kurz mit dem 3-stelligen ICD-10-Code.
{C77_C79_EXPLANATION}"""),
            },
            {
                "role": "assistant",
                "content": "Ich werde auf die folgenden Diagnosetexte kurz mit dem ICD-10-Code antworten."
            }
        ]
        final_question = f"""Was ist der 3-stellige ICD-10-Code für diese Diagnose?\nDiagnose: "{diagnosis_text}"."""
        if prompt_type.endswith("-ctx"):
            final_question = f"{final_question} Die Diagnose stammt aus folgendem längeren Text: {snippet}"
        return chat_start + QAS[prompt_type] + [
            {
                "role": "user",
                "content": final_question
            }
        ]
    else:
        return ValueError(f"""Unknown prompt type "{prompt_type}" """)

def build_step2_query_for_step1_answer(model, prompt_type, snippet, diagnosis_text):
    query = {
        "model": model,
        "stream": False,
        "options": {
            "seed": 112233,
            "temperature": 0,
            "num_predict": 50
        }
    }
    query["messages"] = build_query_messages(prompt_type, snippet, diagnosis_text)
    return query


def read_expected_answers_from_data():
    snippets = get_evaldata()
    return {idx: [label["diagnosis"] for label in snippet["label"]] for idx, snippet in enumerate(snippets)}


def evaluate_step2_model_prompt(model, prev_step_results, prompt_type, inference_framework, output_dir=".", test_only=False, show_progress=True):

    if output_file_exists(model=model, step=2, input_filename=prev_step_results, prompt_type=prompt_type, output_dir=output_dir):
        return

    all_queries = []
    step1_results_df = pd.read_csv(prev_step_results, dtype={"model_answer": pd.StringDtype()})
    step1_results_df = step1_results_df.dropna(subset=['snippet_id'])
    expected_answers = read_expected_answers_from_data()

    for row in step1_results_df.itertuples():
        diagnosis_texts = extract_diagnosis_texts(row.model_answer)
        for diagnosis_text in diagnosis_texts:
            all_queries.append({
                "snippet_id": row.snippet_id,
                "snippet": row.snippet,
                "query": build_step2_query_for_step1_answer(model, prompt_type, row.snippet, diagnosis_text),
                "diagnosis": diagnosis_text,
                "expected_answer": expected_answers[row.snippet_id]
            })
    all_queries = pd.DataFrame(all_queries)
    if test_only:
        all_queries = all_queries.head(test_only).copy()

    start_time = time.time()
    results_df = execute_queries(query_df=all_queries, inference_framework=inference_framework, show_progress=show_progress)
    create_results_file_and_log_time(df=results_df, model=model, step=2,
                                     start_time=start_time, input_filename=prev_step_results, prompt_type=prompt_type,
                                     output_dir=output_dir)

PROMPT_TYPES = ["0-uro", "0-uro-ctx",
                "0-gyn", "0-gyn-ctx",
                "2-uro", "2-uro-ctx",
                "2-gyn", "2-gyn-ctx"]


def run_evaluation_step2(model, prev_step_results, prompt_type=None, inference_framework="ollama", output_dir="output", show_progress=True, test_only = None):
    ensure_directory(output_dir)
    if prompt_type:
        evaluate_step2_model_prompt(model=model,
                                    prev_step_results=prev_step_results,
                                    prompt_type=prompt_type,
                                    inference_framework=inference_framework, output_dir=output_dir,
                                    show_progress=show_progress,
                                    test_only=test_only)
    else:
        for prompt_type in PROMPT_TYPES:
            print(f"""Evaluating prompt type {prompt_type}""")
            evaluate_step2_model_prompt(model=model,
                                        prev_step_results=prev_step_results,
                                        prompt_type=prompt_type,
                                        inference_framework=inference_framework,
                                        output_dir=output_dir,
                                        show_progress=show_progress,
                                        test_only=test_only)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation step 2 based on step 1")
    parser.add_argument("--model", type=str, help="Model name to use.", default="mistral")
    parser.add_argument("--prev-step-results", type=str, help="File with results from step 1", required=True)
    parser.add_argument("--prompt-type", choices=PROMPT_TYPES, help="""The type of the prompt that is used for querying the LLM.
If not specified, all prompt types are used.""", default=None)
    parser.add_argument("--inference-framework", choices=INFERENCE_FRAMEWORKS, default="ollama",
                        help="""The inference framework: "ollama" (default) or "transformers".""")
    parser.add_argument("--test-only", type=int, help="Number of prompts to use (for testing the script)", default = None)
    parser.add_argument("--output-dir", type=str, default="output",
                        help="A directory for creating the results files (will be created if it doesn't exist).")

    args = parser.parse_args()

    run_evaluation_step2(model=args.model, prev_step_results=args.prev_step_results,
                         prompt_type=args.prompt_type,
                         inference_framework=args.inference_framework,
                         output_dir=args.output_dir,
                         test_only=args.test_only)

