
import pandas as pd
import argparse
import time
from results_helper import create_results_file_and_log_time, ensure_directory, get_evaldata, output_file_exists
from inference_helper import execute_queries, INFERENCE_FRAMEWORKS


INITIAL_PROMPT = """Gibt es eine oder mehrere Tumordiagnosen in folgendem Diagnosetext?
Als Tumordiagnose zählen hier Diagnosen, die im Kapitel II (Neubildungen/neoplasms) der International Classification of Diseases 10th revision (ICD-10) in den Kategorien C00-D48 beschrieben sind.
Eine Tumorerkrankung in diesem Sinn ist eine Neubildung abnormen Gewebes aus körpereigenen Zellen im Körper des Patienten.
Es ist entscheidend, nur Diagnosen als Tumordiagnosen zu werten, bei denen klar eine solche Tumorerkrankung beschrieben wird.
Aussagen über Symptome, Therapien oder andere Erkrankungen sollten nicht als Tumordiagnosen interpretiert werden.
Denken Sie nach, ob es sich bei den Diagnosen wirklich um Tumordiagnosen handelt und geben Sie nur Tumordiagnosen zurück, keine anderen Diagnosen.
Antworten Sie mit einem JSON-Array, das die Tumordiagnosen als Strings beinhaltet oder mit einem leeren Array, falls es keine eindeutigen Tumordiagnosen gibt.
Geben Sie im Antwort-Array keine Diagnosen an, die keine Tumordiagnosen sind."""
INITIAL_PROMPT = INITIAL_PROMPT.replace("\n", " ")


def qa(diagnosis_text, answer):
    return [
        {
            "role": "user",
            "content": f"Gibt es eine oder mehrere Tumordiagnosen in folgendem Diagnosetext?\nDiagnosetext: {diagnosis_text}"
        },
        {
            "role": "assistant",
            "content": answer
        }
    ]

QAS = {}
QAS["2-uro"] = (
    qa("Prostata-Karzinom pT1. Ausschluss Lebermetastasen", '["Prostata-Karzinom"]') +
    qa("Harnstauung links. Ausschluss eines Urothelkarzinoms", "[] (Das Urothelkarzinom wird ausgeschlossen.)")
)

QAS["4-uro"] = (
    qa("Prostata-Karzinom pT1. Ausschluss Lebermetastasen", '["Prostata-Karzinom"]') +
    qa("Miktionsbeschwerden. Ausschluss PCA. Z.n. Nierentransplantation 1980 links",
        "[] (Die Diagnose Prostatakarzinom/PCA ist hier ausgeschlossen.)") +
    qa("Harnstauung links. Ausschluss eines Urothelkarzinoms",
        "[] (Das Urothelkarzinom wird ausgeschlossen.)") +
    qa("Z.n. TUR-B-NR mit dem Nachweis eines Urothelkarzinoms der Harnblase pT2G3",
        '["Urothelkarzinom der Harnblase"]')
)


QAS["6-uro"] = QAS["4-uro"] + (
    qa("""Digital-rektale Untersuchung zeigt eine vergrößerte, aber gleichmäßig geformte Prostata.
PSA-Wert im Normbereich. Diagnose: Benigne Prostatahyperplasie (BPH) ohne Hinweise auf Prostatakarzinom.""",
    """[] (Der Verdacht auf die erwähnte Tumorerkrankung, hier "Prostatakarzinom", wird nicht bestätigt.
Die Diagnose "Benigne Prostatahyperplasie" (ICD-10-Code N40) zählt nicht als Tumordiagnose.)""") +
    qa("""Z.n. Adenomkarzinom der Prostata pT2b. ED 2015.""", '["Adenomkarzinom der Prostata"]')
)

QAS["2-gyn"] = (
    qa("""2005: Invasives lobuläres Mammakarzinom links, Stadium II
Sekundärbefunde: Verdacht auf axilläre Lymphknotenmetastasen links
BRCA1-Genmutation positiv""", '["Mammakarzinom"]') +
    qa("""CA-125-Wert im Normbereich, unauffällige Sonographie. Kein Anhalt für ein Ovarialkarzinoms.""",
        '[] (Der Verdacht auf eine erwähnte Tumorerkrankung, hier "Ovarialkarzinom", wird nicht bestätigt.)')
)

QAS["4-gyn"] = (
    qa("""2005: Invasives lobuläres Mammakarzinom links, Stadium II
Sekundärbefunde: Verdacht auf axilläre Lymphknotenmetastasen links
BRCA1-Genmutation positiv""", '["Mammakarzinom"]') +
    qa("""Die Patientin berichtet über tastbare Veränderung in der rechten Brust.
Die durchgeführte Mammographie und der Brustultraschall zeigen keine Anomalien,
und die Veränderung entspricht einer gutartigen zystischen Läsion.
MRT der Brust bestätigt das Fehlen maligner Merkmale.
Ausschluss eines Mammakarzinoms""", "[] (Mammakarzinom wird ausgeschlossen)") +
    qa("""CA-125-Wert im Normbereich, unauffällige Sonographie. Kein Anhalt für ein Ovarialkarzinoms.""",
            '[] (Die Tumorerkrankung "Ovarialkarzinom" wird nicht bestätigt.)') +
    qa("""Seröses Ovarialkarzinom des linken Ovars, keine Anzeichen für Fernmetastasen""", '["Ovarialkarzinom"]')
)

QAS["6-gyn"] = QAS["4-gyn"] + (
    qa("""Ultraschall der Brust zeigt ein dichtes, aber homogenes Brustgewebe mit mehreren kleinen, nicht-suspekten Zysten.
Kein Anhalt für Malignität.
Diagnose: Fibrozystische Mastopathie.""",
    """[] (Es wurde keine Tumorerkrankung bei der Patientin festgestellt.
Die hier gestellte Diagnose "Fibrozystische Mastopathie" (ICD-10-Code N60) zählt nicht als Tumordiagnose.)""") +
    qa("""Duktales In-situ-Karzinom der rechten Brust.
Keine Anzeichen für Fernmetastasen.
Hormonrezeptorstatus zeigt Östrogenrezeptor-Positivität. HER2/neu-Status negativ.""",
    '["Duktales In-situ-Karzinom der rechten Brust"]')
)


def build_query_messages(prompt_type, snippet_text):
    if prompt_type == "0":
        return [
            {
                "role": "user",
                "content": f"""{INITIAL_PROMPT}
Gibt es eine oder mehrere Tumordiagnosen in folgendem Diagnosetext? Diagnosetext:
{snippet_text}""",
            },
        ]
    else: # few-shot-prompting
        chat_start = [
            {
                "role": "user",
                "content": INITIAL_PROMPT,
            },
            {
                "role": "assistant",
                "content": "Ich werde auf die folgenden Diagnosetexte kurz mit einem JSON-Array antworten und nur Tumordiagnosen berücksichtigen."
            }
        ]
        final_question = [
                {
                    "role": "user",
                    "content": f"Gibt es eine oder mehrere Tumordiagnosen in folgendem Diagnosetext?\nDiagnosetext: {snippet_text}",
                }
            ]

        return chat_start + QAS[prompt_type] + final_question


def build_query(model, prompt_type, snippet_text):
    query = {
        "model": model,
        "stream": False,
        "options": {
            "seed": 112233,
            "temperature": 0,
            "num_predict": 80
        }
    }
    query["messages"] = build_query_messages(prompt_type=prompt_type, snippet_text=snippet_text)
    return query


def evaluate_step1_model_prompt(model, prompt_type, inference_framework, output_dir=".", show_progress=True, test_only = None):

    if output_file_exists(model=model, step=1, input_filename=None, prompt_type=prompt_type, output_dir=output_dir):
        return

    evaldata = get_evaldata()

    all_queries = []

    for idx, snippet in enumerate(evaldata):
        snippet_text = snippet["text"]
        query = build_query(model=model, prompt_type=prompt_type, snippet_text=snippet_text)

        all_queries.append({
            "snippet_id": idx,
            "snippet": snippet["text"],
            "query": query,
            "expected_answer": bool(snippet["label"]), # any diagnosis labeled/existing?
        })

    all_queries = pd.DataFrame(all_queries)
    if test_only:
        all_queries = all_queries.head(test_only).copy()
    start_time = time.time()
    results_df = execute_queries(query_df=all_queries, inference_framework=inference_framework, show_progress=show_progress)
    create_results_file_and_log_time(df=results_df, model=model, step=1, start_time=start_time,
                                     input_filename=None, prompt_type=prompt_type,
                                     output_dir=output_dir)


PROMPT_TYPES = ["0"] + list(QAS.keys())


def run_evaluation_step1(model, prompt_type=None, inference_framework="ollama", output_dir=".", show_progress=True, test_only = None):
    ensure_directory(output_dir)
    if prompt_type:
        evaluate_step1_model_prompt(model=model, prompt_type=prompt_type,
                             inference_framework=inference_framework, output_dir=output_dir,
                             show_progress=show_progress,
                             test_only=test_only)
    else:
        for prompt_type in PROMPT_TYPES:
            print(f"""Evaluating prompt type {prompt_type}""")
            evaluate_step1_model_prompt(model=model, prompt_type=prompt_type,
                                 inference_framework=inference_framework,
                                 output_dir=output_dir,
                                 show_progress=show_progress,
                                 test_only=test_only)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation with specified step and model.")
    parser.add_argument("--model", type=str, help="Model name to use.", default="mistral")
    parser.add_argument("--test-only", type=int, help="Number of prompts to use (for testing the script)", default=None)
    parser.add_argument("--prompt-type", choices=PROMPT_TYPES, help="""The type of the prompt that is used for querying the LLM.
If not specified, all prompt types are used.""", default=None)
    parser.add_argument("--inference-framework", choices=INFERENCE_FRAMEWORKS, default="ollama",
                        help="""The inference framework: "ollama" (default) or "transformers".""")
    parser.add_argument("--output-dir", type=str, default="output",
                        help="A directory for creating the results files (will be created if it doesn't exist).")

    args = parser.parse_args()

    run_evaluation_step1(model=args.model, prompt_type=args.prompt_type,
                         inference_framework=args.inference_framework, output_dir=args.output_dir,
                         test_only=args.test_only)