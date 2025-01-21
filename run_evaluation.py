import os
import time
import traceback
from pathlib import Path
from analyze_step1_results import analyze_step1_results
from analyze_step2_results import analyze_step2_results
from analyze_step3_results import analyze_step3_results
from find_best_result import find_best_result
from create_tables import create_summary_tables
from model_config import MODEL_CONFIG
from run_helper import run_script, send_exception_email, send_notification_email
from results_helper import extract_model_name_from_filename

# Configuration
inference_framework = "transformers"
output_dir = "output"
test_only = None
model_list = MODEL_CONFIG.keys()

# Map steps to functions
step_analysis = {
    1: analyze_step1_results,
    2: analyze_step2_results,
    3: analyze_step3_results,
}

# Resolve the directory where the script is located
main_dir = Path(os.getcwd()) if "__file__" not in globals() else Path(__file__).resolve().parent

# Timer
start_time = time.time()

def print_time(step_name):
    elapsed = time.time() - start_time
    hours, rem = divmod(elapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    formatted_time = f"{int(hours)}:{int(minutes):02}:{int(seconds):02}"
    print(f"Elapsed time after {step_name}: {formatted_time} (hh:mm:ss)")


best_models = {}

# run steps 2 and 3
def run_complete_step(step):
    prev_step = step - 1
    prev_analysis_file = os.path.join(output_dir, f"analysis_step{prev_step}.csv")

    # Find best result overall from the previous step
    best_prev_result = find_best_result(analysis_file=prev_analysis_file, models=best_models)
    print(f"Building on best result from previous steps: {best_prev_result}")
    best_models[prev_step] = extract_model_name_from_filename(filename=best_prev_result, step=prev_step)
    best_prev_result_path = f"{output_dir}/{best_prev_result}.csv"

    for model in model_list:
        print(f"Starting evaluation of {model}...")
        # Run step using current model and best result from all models in previous step.
        # (The evaluation is run in a separate process to ensure that the GPU RAM has been cleaned.)
        run_script(
            f"run_evaluation_step{step}.py",
            model=model,
            inference_framework=inference_framework,
            prev_step_results=best_prev_result_path,
            output_dir=output_dir,
            test_only=test_only,
        )

        # Find best result where this specific model is used in all previous steps
        best_prev_result_model = find_best_result(analysis_file=prev_analysis_file,
                                                  models={i: model for i in range(1, step)})

        if best_prev_result_model != best_prev_result:
            best_prev_result_model_path = f"{output_dir}/{best_prev_result_model}.csv"
            run_script(
                f"run_evaluation_step{step}.py",
                model=model,
                inference_framework=inference_framework,
                prev_step_results=best_prev_result_model_path,
                output_dir=output_dir,
                test_only=test_only,
            )

    # Create analysis summary file
    step_analysis[step](
        results_path=output_dir, output=f"{output_dir}/analysis_step{step}.csv"
    )
    print_time(f"Step {step}")


try:
    # Step 1
    print(" === Step 1 === ")
    for model in model_list:
        print(f"Starting evaluation of {model}...")
        run_script(
            f"run_evaluation_step1.py",
            model=model,
            inference_framework=inference_framework,
            output_dir=output_dir,
            test_only=test_only,
        )

    analyze_step1_results(results_path=output_dir, output=f"{output_dir}/analysis_step1.csv")
    print_time("Step 1")

    # Step 2
    print(" === Step 2 === ")
    run_complete_step(2)

    # Step 3
    print(" === Step 3 === ")
    run_complete_step(3)

    # End
    print(" === End === ")
    print_time("all steps")

    # Summarize results in HTML tables
    create_summary_tables(input=output_dir, output_dir=f"{output_dir}/summary")

    send_notification_email(subject="UroLlmEval completed succesfully!")
except Exception as e:
    traceback.print_exc()
    send_exception_email(e)
